"""Qasper long-document memory experiment (batched).

Task: evidence-block selection. All blocks of a paper are read through the
memory mechanism, then the question must be matched to the block holding the
answer evidence. Metric: accuracy@1 / MRR over that paper's blocks.

Conditions (block representations differ only in how the state flows):

    carried   state passed between blocks           -> rep_i summarises 0..i
    reset     state never passed                    -> rep_i is block i alone
    shuffled  state from ANOTHER paper              -> excludes generic bias
    concat    every block encoded independently,    -> same blocks as carried,
              then aggregated (mean)                   no state at all

`concat` is NOT a literal text concatenation: Qasper papers are median 5169
tokens over ~21 blocks and 99.7% exceed the encoder window, so a literal concat
would be truncated for nearly every paper and would be an unfair reference
(review rule 4). Independent-encoding-plus-aggregation sees identical content.

Distance for S3: d = n_blocks - 1 - evidence_idx, i.e. how many blocks follow
the evidence. Sequential readers must carry the evidence further as d grows.

Usage:
    python scripts/train_qasper.py --seed 0 --epochs 3 --out ...
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


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def make_model(init_from, dev):
    rm = JaumeEmbedding.from_pretrained(init_from)
    base = rm.model
    from jaume.modeling.memory import build_memory_pooling
    base.config.pooling = "recurrent_memory_gate"
    base.config.memory_output = "new_slots"
    base.config.num_slots = 8
    base.config.slot_heads = 8
    base.pooling = build_memory_pooling(base.config)
    base.to(dev)
    src = os.path.join(init_from, "pytorch_model.bin")
    if os.path.exists(src):
        sd = torch.load(src, map_location="cpu", weights_only=True)
        ms = base.state_dict()
        for k, v in ms.items():
            if k in sd and tuple(sd[k].shape) == tuple(v.shape):
                v.copy_(sd[k])
    return rm


def trim(rows, max_blocks):
    """Keep only rows whose evidence fits inside the block cap."""
    out = []
    for r in rows:
        if r["evidence_idx"] < min(len(r["blocks"]), max_blocks):
            out.append(r)
    return out


def evaluate(rm, rows, cond, max_blocks, max_length, other_reps=None):
    a1 = mrr = 0.0
    strata = {}
    n = 0
    B = 8
    with torch.no_grad():
        for s in range(0, len(rows), B):
            batch = rows[s : s + B]
            if cond == "concat":
                reps = QB.run_batch_states(rm, batch, False, max_length, max_blocks)
            elif cond == "reset":
                reps = QB.run_batch_states(rm, batch, False, max_length, max_blocks)
            else:
                reps = QB.run_batch_states(rm, batch, True, max_length, max_blocks)
            if cond == "shuffled" and other_reps is not None:
                reps = [other_reps[(s + j) % len(other_reps)]
                        for j in range(len(batch))]
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
    ap.add_argument("--data", default=fr"{PROJECT_ROOT}\data\qasper_blocks")
    ap.add_argument("--init-from", default=fr"{PROJECT_ROOT}\350m-mlm")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--papers-per-step", type=int, default=6)
    ap.add_argument("--max-blocks", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--eval-limit", type=int, default=120)
    args = ap.parse_args()

    dev = resolve_device()
    torch.manual_seed(args.seed)

    tr = trim(load(os.path.join(args.data, "train.jsonl")), args.max_blocks)
    va = trim(load(os.path.join(args.data, "val.jsonl")), args.max_blocks)
    te = trim(load(os.path.join(args.data, "test.jsonl")), args.max_blocks)
    print(f"[q] train={len(tr)} val={len(va)} test={len(te)} "
          f"max_blocks={args.max_blocks} bps={args.papers_per_step}", flush=True)

    rm = make_model(args.init_from, dev)
    base = rm.model
    base.train()
    opt = torch.optim.AdamW(base.parameters(), lr=args.lr, weight_decay=0.01)

    t0 = time.time()
    best = -1.0
    for ep in range(args.epochs):
        perm = torch.randperm(len(tr))
        tot, nb = 0.0, 0
        opt.zero_grad()
        for s in range(0, len(tr), args.papers_per_step):
            idx = perm[s : s + args.papers_per_step].tolist()
            batch = [tr[i] for i in idx]
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=has_amp(dev)):
                reps = QB.run_batch_states(rm, batch, True,
                                           args.max_length, args.max_blocks)
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
            if nb % 30 == 0:
                print(f"    [q-s{args.seed}] ep{ep} step {nb} "
                      f"loss {tot/nb:.4f} ({time.time()-t0:.0f}s)", flush=True)

        base.eval()
        v = evaluate(rm, va[: args.eval_limit], "carried",
                     args.max_blocks, args.max_length)
        print(f"  [q-s{args.seed}] ep{ep} loss {tot/max(1,nb):.4f} "
              f"val carried acc@1 {v['acc1']:.4f} mrr {v['mrr']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        base.train()
        if v["acc1"] > best:
            best = v["acc1"]
            os.makedirs(args.out, exist_ok=True)
            torch.save(base.state_dict(), os.path.join(args.out, "pytorch_model.bin"))
            rm.tokenizer.save_pretrained(args.out)
            with open(os.path.join(args.out, "config.json"), "w",
                      encoding="utf-8") as f:
                json.dump(base.config.to_dict(), f, indent=2)

    with open(os.path.join(args.out, "train_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump({"seed": args.seed, "epochs": args.epochs,
                   "train": len(tr), "best_val_acc1": best,
                   "seconds": round(time.time() - t0, 1),
                   "max_blocks": args.max_blocks,
                   "papers_per_step": args.papers_per_step}, f, indent=2)
    print(f"[q] saved -> {args.out}")


if __name__ == "__main__":
    main()
