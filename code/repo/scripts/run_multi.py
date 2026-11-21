"""Fix the ill-posed training target: multi-label set supervision.

Diagnosis (diagnose_composition_learning.py) showed:
  * the pretrained encoder ALREADY separates the gold block from others
    (cos 0.593 vs 0.559, gold mean rank 3.42 vs chance 6.0, 27.5% rank-1)
  * but training used a single-label CE on gold_blocks[0], while 100% of items
    have more than one correct block -- so 55% of correct targets were being
    pushed DOWN by the loss
  * after training the model scored 0.19 while an untrained BM25 scored 0.32

That means the loss was actively destroying the signal that fine-tuning should
be building on. The fix is a multi-label objective:

    loss = BCEWithLogits(score_i, 1.0 if i in gold_blocks else 0.0)

over all blocks of the paper. This rewards every correct block and penalises
only the genuinely wrong ones.

Everything else (data, splits, budget) is held fixed so this isolates the
training target. The comparison is reported against the ORIGINAL nine-run
protocol so nothing is overwritten.

Usage:
    python scripts/run_multi.py --split 0 --seed 0
"""
import os as _os
PROJECT_ROOT = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import argparse
import json
import math
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

# ---- SAME frozen config as the nine-run protocol --------------------------
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
    sd = torch.load(os.path.join(BASE, "pytorch_model.bin"),
                    map_location="cpu", weights_only=True)
    ms = base.state_dict()
    for k, v in ms.items():
        if k in sd and tuple(sd[k].shape) == tuple(v.shape):
            v.copy_(sd[k])
    return rm


def set_f1(pred, gold):
    p, g = set(pred), set(gold)
    inter = len(p & g)
    return 2 * inter / (len(p) + len(g)) if p and g else 0.0


def eval_conditions(rm, rows):
    out = {c: {} for c in ("carried", "static", "shuffled")}
    B = 8
    with torch.no_grad():
        for s in range(0, len(rows), B):
            batch = rows[s : s + B]
            rep_c = QB.run_batch_states(rm, batch, True, MAX_LENGTH, MAX_BLOCKS)
            rep_s = QB.run_batch_states(rm, batch, False, MAX_LENGTH, MAX_BLOCKS)
            g = torch.Generator().manual_seed(0)
            perm = torch.randperm(len(batch), generator=g).tolist()
            rep_sh = [rep_c[i] for i in perm]
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
    ap.add_argument("--outroot", default=fr"{PROJECT_ROOT}\output\comp_multi")
    args = ap.parse_args()

    dev = resolve_device()
    torch.manual_seed(args.seed)
    d = os.path.join(args.data, f"split{args.split}")
    tr = load(os.path.join(d, "train.jsonl"))[:520]
    te = load(os.path.join(d, "test.jsonl"))
    print(f"[multi] split={args.split} seed={args.seed} "
          f"train={len(tr)} test={len(te)}", flush=True)

    rm = make_model(dev)
    base = rm.model
    base.train()
    opt = torch.optim.AdamW(base.parameters(), lr=LR, weight_decay=0.01)

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
                # MULTI-LABEL: every gold block is positive
                gold = set(r["gold_blocks"])
                y = torch.tensor(
                    [1.0 if j in gold else 0.0 for j in range(len(logits))],
                    device=dev)
                # weight positives up so the 2-3 gold blocks are not swamped
                w = torch.where(y > 0, torch.full_like(y, 4.0),
                                torch.ones_like(y))
                loss = loss + F.binary_cross_entropy_with_logits(
                    logits, y, weight=w)
            loss = loss / len(batch)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            tot += float(loss)
            nb += 1
        base.eval()
        tr_eval = eval_conditions(rm, tr[:64])
        te_eval = eval_conditions(rm, te)
        a_tr = sum(tr_eval["carried"].values()) / len(tr_eval["carried"])
        a_te = sum(te_eval["carried"].values()) / len(te_eval["carried"])
        print(f"  [s{args.split}k{args.seed}] ep{ep} loss {tot/max(1,nb):.4f} "
              f"train_f1 {a_tr:.4f} test_f1(carried) {a_te:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        base.train()

    base.eval()
    res = eval_conditions(rm, te)
    summary = {c: sum(res[c].values()) / len(res[c]) for c in res}
    print(f"[multi] split={args.split} seed={args.seed} "
          f"carried={summary['carried']:.4f} static={summary['static']:.4f} "
          f"shuffled={summary['shuffled']:.4f}", flush=True)

    os.makedirs(args.outroot, exist_ok=True)
    # save the trained weights: without them the run cannot be re-analysed,
    # and the earlier version of this script omitted them entirely
    ckpt_dir = os.path.join(args.outroot, f"ckpt_split{args.split}_seed{args.seed}")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(base.state_dict(), os.path.join(ckpt_dir, "pytorch_model.bin"))
    rm.tokenizer.save_pretrained(ckpt_dir)
    with open(os.path.join(ckpt_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(base.config.to_dict(), f, indent=2)
    out = os.path.join(args.outroot, f"split{args.split}_seed{args.seed}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"split": args.split, "seed": args.seed,
                   "target": "multi_label_bce",
                   "summary": summary, "per_item": res,
                   "train_size": len(tr), "test_size": len(te),
                   "seconds": round(time.time() - t0, 1)}, f)
    print(f"[multi] wrote {out}")


if __name__ == "__main__":
    main()
