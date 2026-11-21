"""Nine-run protocol: 3 fixed paper-level splits x 3 fixed seeds.

Frozen configuration (memory-preregistration.md section 3.2). This script runs
exactly ONE (split, seed) cell per invocation so the driver cannot silently
change the grid, and writes a result file per cell.

Conditions evaluated on the SAME checkpoint:
    carried   state passed between blocks
    static    state never passed  (identical to `concat` in this task coding)
    shuffled  state permuted across items, fixed seed

Primary metric: set-level F1 over the gold evidence blocks.
    pred = top-k blocks by cosine(question, block_repr), k = |gold|
    F1   = 2*|pred & gold| / (|pred| + |gold|)

Usage:
    python scripts/run_nine.py --split 0 --seed 0
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

# ---- FROZEN CONFIG (do not edit between runs) -----------------------------
MAX_BLOCKS = 12
PAPERS_PER_STEP = 4
EPOCHS = 3
LR = 1e-4
MAX_LENGTH = 128
BASE = fr"{PROJECT_ROOT}\350m-mlm"
# ---------------------------------------------------------------------------


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def make_model(dev):
    rm = JaumeEmbedding.from_pretrained(BASE)
    base = rm.model
    from jaume.modeling.memory import build_memory_pooling
    base.config.pooling = "recurrent_memory_gate"
    base.config.memory_output = "new_slots"
    base.config.num_slots = 8
    base.config.slot_heads = 8
    base.config.state_update = "additive"
    base.pooling = build_memory_pooling(base.config)
    base.to(dev)
    src = os.path.join(BASE, "pytorch_model.bin")
    sd = torch.load(src, map_location="cpu", weights_only=True)
    ms = base.state_dict()
    for k, v in ms.items():
        if k in sd and tuple(sd[k].shape) == tuple(v.shape):
            v.copy_(sd[k])
    return rm


def set_f1(pred, gold):
    p, g = set(pred), set(gold)
    inter = len(p & g)
    if not p or not g:
        return 0.0
    return 2 * inter / (len(p) + len(g))


def eval_conditions(rm, rows):
    """Return per-item set-F1 for each condition, keyed by item id."""
    out = {c: {} for c in ("carried", "static", "shuffled")}
    B = 8
    with torch.no_grad():
        for s in range(0, len(rows), B):
            batch = rows[s : s + B]
            rep_c = QB.run_batch_states(rm, batch, True, MAX_LENGTH, MAX_BLOCKS)
            rep_s = QB.run_batch_states(rm, batch, False, MAX_LENGTH, MAX_BLOCKS)
            g = torch.Generator().manual_seed(0)
            perm = torch.randperm(len(batch), generator=g)
            rep_sh = [rep_c[i] for i in perm.tolist()]
            qs = QB.question_embs(rm, [r["question"] for r in batch])
            for i, r in enumerate(batch):
                k = len(r["gold_blocks"])
                for name, reps in (("carried", rep_c), ("static", rep_s),
                                   ("shuffled", rep_sh)):
                    sc = F.normalize(reps[i], dim=-1) @ qs[i]
                    pred = torch.topk(sc, k).indices.tolist()
                    out[name][r["id"]] = set_f1(pred, r["gold_blocks"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data", default=fr"{PROJECT_ROOT}\data\qasper_comp_clean")
    ap.add_argument("--outroot", default=fr"{PROJECT_ROOT}\output\comp_nine")
    args = ap.parse_args()

    dev = resolve_device()
    torch.manual_seed(args.seed)

    d = os.path.join(args.data, f"split{args.split}")
    tr = load(os.path.join(d, "train.jsonl"))
    va = load(os.path.join(d, "val.jsonl"))
    te = load(os.path.join(d, "test.jsonl"))
    tr = [r for r in tr if len(r["blocks"]) >= 2][:520]
    print(f"[nine] split={args.split} seed={args.seed} "
          f"train={len(tr)} val={len(va)} test={len(te)}", flush=True)

    rm = make_model(dev)
    base = rm.model
    base.train()
    opt = torch.optim.AdamW(base.parameters(), lr=LR, weight_decay=0.01)

    def target_index(r):
        return r["gold_blocks"][0] if r["gold_blocks"] else 0

    t0 = time.time()
    for ep in range(EPOCHS):
        perm = torch.randperm(len(tr))
        tot, nb = 0.0, 0
        opt.zero_grad()
        for s in range(0, len(tr), PAPERS_PER_STEP):
            idx = perm[s : s + PAPERS_PER_STEP].tolist()
            batch = [tr[i] for i in idx]
            reps = QB.run_batch_states(rm, batch, True, MAX_LENGTH, MAX_BLOCKS)
            qs = QB.question_embs(rm, [r["question"] for r in batch])
            loss = 0.0
            for rep, q, r in zip(reps, qs, batch):
                logits = (F.normalize(rep, dim=-1) @ q) * 8.0
                tgt = torch.tensor([target_index(r)], device=dev)
                loss = loss + F.cross_entropy(logits.unsqueeze(0), tgt)
            loss = loss / len(batch)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            tot += float(loss)
            nb += 1
        print(f"  [s{args.split}k{args.seed}] ep{ep} loss {tot/max(1,nb):.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    base.eval()
    res = eval_conditions(rm, te)

    def mean(dd):
        return sum(dd.values()) / len(dd)

    summary = {c: mean(res[c]) for c in res}
    print(f"[nine] split={args.split} seed={args.seed} "
          f"carried={summary['carried']:.4f} static={summary['static']:.4f} "
          f"shuffled={summary['shuffled']:.4f}", flush=True)

    os.makedirs(args.outroot, exist_ok=True)
    out = os.path.join(args.outroot, f"split{args.split}_seed{args.seed}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"split": args.split, "seed": args.seed,
                   "config": {"max_blocks": MAX_BLOCKS,
                              "papers_per_step": PAPERS_PER_STEP,
                              "epochs": EPOCHS, "lr": LR,
                              "max_length": MAX_LENGTH},
                   "summary": summary, "per_item": res,
                   "train_size": len(tr), "test_size": len(te),
                   "seconds": round(time.time() - t0, 1)}, f)
    print(f"[nine] wrote {out}")


if __name__ == "__main__":
    main()
