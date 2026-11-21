"""R1 protocol: compare original carried against R1a and R1b on Qasper.

Pre-registered comparison (review):
  baseline  state_update=additive  (original JM-R)
  R1a       state_update=norm      (normalise the accumulated state)
  R1b       state_update=bounded   (learned interpolation, so a new chunk keeps
                                    a fixed share of the state)

All three share: same data split, same block count, same training budget,
same evaluation path. The original checkpoint and the original code path are
kept (additive stays the default), so nothing is overwritten.

What is reported (review asked for all four layers):

  representation : block-pair cosine, block rep std, state norm over position
  task           : validation loss, test acc@1 / MRR
  mechanism      : carried vs reset vs shuffled on the SAME checkpoint
  generalisation : one fixed split for screening; multi-split validation is a
                   separate step (multiple seeds are NOT multiple datasets)

Usage:
    python scripts/train_r1.py --mode norm --seed 0 --epochs 3 --out ...
"""
import os as _os
PROJECT_ROOT = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

REPO = fr"{PROJECT_ROOT}\code\repo"
sys.path.insert(0, REPO)
from jaume.device import resolve_device, has_amp, make_scaler, empty_cache, max_memory_allocated, reset_peak_memory_stats, total_memory, seed_all
sys.path.insert(0, os.path.join(REPO, "scripts"))

from jaume.inference.encode import JaumeEmbedding  # noqa: E402
import qasper_batch as QB  # noqa: E402

MODES = ("additive", "norm", "bounded")


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def make_model(init_from, dev, mode):
    rm = JaumeEmbedding.from_pretrained(init_from)
    base = rm.model
    from jaume.modeling.memory import build_memory_pooling
    base.config.pooling = "recurrent_memory_gate"
    base.config.memory_output = "new_slots"
    base.config.num_slots = 8
    base.config.slot_heads = 8
    base.config.state_update = mode
    base.pooling = build_memory_pooling(base.config)
    base.to(dev)
    src = os.path.join(init_from, "pytorch_model.bin")
    if os.path.exists(src):
        sd = torch.load(src, map_location="cpu", weights_only=True)
        ms = base.state_dict()
        reused = 0
        for k, v in ms.items():
            if k in sd and tuple(sd[k].shape) == tuple(v.shape):
                v.copy_(sd[k])
                reused += 1
        print(f"[R1] reused {reused}/{len(ms)} tensors (mode={mode})")
    return rm


def trim(rows, max_blocks):
    return [r for r in rows if r["evidence_idx"] < min(len(r["blocks"]), max_blocks)]


def diag_representations(rm, rows, max_blocks, max_length, n=6):
    """Representation-level diagnostics: collapse, scale growth."""
    coss, stds, norms = [], [], []
    for r in rows[:n]:
        reps, s = [], None
        with torch.no_grad():
            for b in r["blocks"][:max_blocks]:
                e = rm.tokenizer([b], padding=True, truncation=True,
                                 max_length=max_length, return_tensors="pt").to(rm.device)
                _, _, s = rm.model.encode(e["input_ids"], e["attention_mask"], s)
                reps.append(F.normalize(rm.model.projection(s.mean(dim=1)), dim=-1))
                norms.append(float(s.norm()))
        V = torch.cat(reps)
        sims = V @ V.T
        k = V.shape[0]
        off = sims - torch.eye(k, device=sims.device)
        coss.append(float(off.sum() / (k * (k - 1))))
        stds.append(float(V.std(dim=0).mean()))
    return {"block_cos": sum(coss) / len(coss),
            "block_std": sum(stds) / len(stds),
            "state_norm_first": norms[0] if norms else 0.0,
            "state_norm_last": norms[-1] if norms else 0.0}


def evaluate(rm, rows, cond, max_blocks, max_length):
    a1 = mrr = 0.0
    strata = {}
    n = 0
    B = 8
    with torch.no_grad():
        for s in range(0, len(rows), B):
            batch = rows[s : s + B]
            reps = QB.run_batch_states(
                rm, batch, cond in ("carried", "shuffled"), max_length, max_blocks)
            if cond == "shuffled":
                reps = reps[::-1]
            qs = QB.question_embs(rm, [r["question"] for r in batch])
            golds = [r["evidence_idx"] for r in batch]
            A, M, ranks = QB.rank_all(reps, qs, golds)
            a1 += A * len(batch)
            mrr += M * len(batch)
            for r, rk in zip(batch, ranks):
                d = r["distance"]
                k = "d=0" if d == 0 else ("d=1-3" if d <= 3 else
                                          ("d=4-7" if d <= 7 else "d>=8"))
                st = strata.setdefault(k, [0, 0, 0])
                st[0] += 1 if rk == 1 else 0
                st[1] += 1.0 / rk
                st[2] += 1
            n += len(batch)
    out = {"acc1": a1 / n, "mrr": mrr / n, "n": n}
    out["strata"] = {k: {"acc1": v[0] / v[2], "mrr": v[1] / v[2], "n": v[2]}
                     for k, v in strata.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--data", default=fr"{PROJECT_ROOT}\data\qasper_blocks")
    ap.add_argument("--init-from", default=fr"{PROJECT_ROOT}\350m-mlm")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--papers-per-step", type=int, default=4)
    ap.add_argument("--max-blocks", type=int, default=12)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--eval-limit", type=int, default=120)
    args = ap.parse_args()

    dev = resolve_device()
    torch.manual_seed(args.seed)

    tr = trim(load(os.path.join(args.data, "train.jsonl")), args.max_blocks)
    va = trim(load(os.path.join(args.data, "val.jsonl")), args.max_blocks)
    te = trim(load(os.path.join(args.data, "test.jsonl")), args.max_blocks)
    print(f"[R1:{args.mode}] train={len(tr)} val={len(va)} test={len(te)} "
          f"blocks<={args.max_blocks}", flush=True)

    rm = make_model(args.init_from, dev, args.mode)
    base = rm.model
    base.train()
    opt = torch.optim.AdamW(base.parameters(), lr=args.lr, weight_decay=0.01)

    log = []
    t0 = time.time()
    for ep in range(args.epochs):
        perm = torch.randperm(len(tr))
        tot, nb = 0.0, 0
        opt.zero_grad()
        for s in range(0, len(tr), args.papers_per_step):
            idx = perm[s : s + args.papers_per_step].tolist()
            batch = [tr[i] for i in idx]
            reps = QB.run_batch_states(rm, batch, True, args.max_length, args.max_blocks)
            qs = QB.question_embs(rm, [r["question"] for r in batch])
            loss = 0.0
            for rep, q, r in zip(reps, qs, batch):
                logits = (F.normalize(rep, dim=-1) @ q) * 8.0
                loss = loss + F.cross_entropy(
                    logits.unsqueeze(0),
                    torch.tensor([r["evidence_idx"]], device=dev))
            loss = loss / len(batch)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            tot += float(loss)
            nb += 1

        base.eval()
        d = diag_representations(rm, va, args.max_blocks, args.max_length)
        v = evaluate(rm, va[: args.eval_limit], "carried",
                     args.max_blocks, args.max_length)
        rec = {"epoch": ep, "loss": tot / max(1, nb), **d,
               "val_acc1": v["acc1"], "val_mrr": v["mrr"]}
        log.append(rec)
        print(f"  ep{ep} loss {rec['loss']:.4f} cos {d['block_cos']:.4f} "
              f"std {d['block_std']:.4f} norm {d['state_norm_first']:.1f}"
              f"->{d['state_norm_last']:.1f} valacc {v['acc1']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        base.train()

    base.eval()
    final = {}
    for cond in ("carried", "reset", "shuffled"):
        final[cond] = evaluate(rm, te, cond, args.max_blocks, args.max_length)
        print(f"  TEST {cond:9s} acc@1 {final[cond]['acc1']:.4f} "
              f"MRR {final[cond]['mrr']:.4f}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    torch.save(base.state_dict(), os.path.join(args.out, "pytorch_model.bin"))
    rm.tokenizer.save_pretrained(args.out)
    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(base.config.to_dict(), f, indent=2)
    with open(os.path.join(args.out, "r1_results.json"), "w", encoding="utf-8") as f:
        json.dump({"mode": args.mode, "seed": args.seed,
                   "log": log, "test": final,
                   "config": {"max_blocks": args.max_blocks,
                              "papers_per_step": args.papers_per_step,
                              "epochs": args.epochs,
                              "max_length": args.max_length}},
                  f, indent=2)
    print(f"[R1:{args.mode}] saved -> {args.out}")


if __name__ == "__main__":
    main()
