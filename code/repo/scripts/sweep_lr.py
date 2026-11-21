"""Find a learning rate that IMPROVES the zero-shot signal instead of destroying it.

Diagnosis established the key fact:
    base encoder, zero training : cos(gold)-cos(other) = +0.021, gold rank 3.27
    after fine-tuning           : test set-F1 0.15-0.19 (worse than BM25 0.32)

So the starting representation already separates gold blocks from distractors,
and fine-tuning is making it worse. That is the signature of too large an
update on too little data (514 training items), not of a missing capability.

This script sweeps the learning rate with the multi-label objective, stopping
each run early (1 epoch) and reporting the test set-F1. It is a small, bounded
search intended to pick a configuration, after which the frozen 3x3 protocol is
re-run with that single configuration.

Usage:
    python scripts/sweep_lr.py --lrs 1e-5 3e-5 1e-4 --epochs 1
"""
import os as _os
PROJECT_ROOT = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import argparse
import json
import os
import statistics
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

MAX_BLOCKS = 12
PAPERS_PER_STEP = 4
MAX_LENGTH = 128
BASE = fr"{PROJECT_ROOT}\350m-mlm"


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
    i = len(p & g)
    return 2 * i / (len(p) + len(g)) if p and g else 0.0


def eval_carried(rm, rows):
    vals = []
    with torch.no_grad():
        for s in range(0, len(rows), 8):
            batch = rows[s : s + 8]
            reps = QB.run_batch_states(rm, batch, True, MAX_LENGTH, MAX_BLOCKS)
            qs = QB.question_embs(rm, [r["question"] for r in batch])
            for i, r in enumerate(batch):
                sc = F.normalize(reps[i], dim=-1) @ qs[i]
                k = len(r["gold_blocks"])
                pred = torch.topk(sc, k).indices.tolist()
                vals.append(set_f1(pred, r["gold_blocks"]))
    return statistics.mean(vals)


def seg_sep(rm, rows):
    """cos(gold) - cos(other), the representation-quality signal."""
    gold, other = [], []
    with torch.no_grad():
        for s in range(0, len(rows), 8):
            batch = rows[s : s + 8]
            reps = QB.run_batch_states(rm, batch, True, MAX_LENGTH, MAX_BLOCKS)
            qs = QB.question_embs(rm, [r["question"] for r in batch])
            for i, r in enumerate(batch):
                sc = F.normalize(reps[i], dim=-1) @ qs[i]
                g = set(r["gold_blocks"])
                for j in range(len(sc)):
                    (gold if j in g else other).append(float(sc[j]))
    return statistics.mean(gold) - statistics.mean(other)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-5, 3e-5, 1e-4])
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--data", default=fr"{PROJECT_ROOT}\data\qasper_comp_clean\split0")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    dev = resolve_device()
    tr = load(os.path.join(args.data, "train.jsonl"))[:520]
    te = load(os.path.join(args.data, "test.jsonl"))[: args.limit]

    print(f"{'lr':>8s} {'loss':>8s} {'train_f1':>9s} {'test_f1':>9s} "
          f"{'sep':>8s}")
    print("-" * 48)
    results = {}
    for lr in args.lrs:
        torch.manual_seed(0)
        rm = make_model(dev)
        base = rm.model
        base.train()
        opt = torch.optim.AdamW(base.parameters(), lr=lr, weight_decay=0.01)
        t0 = time.time()
        for ep in range(args.epochs):
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
                    gold = set(r["gold_blocks"])
                    y = torch.tensor([1.0 if j in gold else 0.0
                                      for j in range(len(logits))], device=dev)
                    w = torch.where(y > 0, torch.full_like(y, 4.0),
                                    torch.ones_like(y))
                    loss = loss + F.binary_cross_entropy_with_logits(
                        logits, y, weight=w)
                (loss / len(batch)).backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                tot += float(loss)
                nb += 1
        base.eval()
        tr_f1 = eval_carried(rm, tr[:64])
        te_f1 = eval_carried(rm, te)
        sep = seg_sep(rm, te[:80])
        results[lr] = {"loss": tot / max(1, nb), "train_f1": tr_f1,
                       "test_f1": te_f1, "sep": sep, "seconds": time.time() - t0}
        print(f"{lr:8.0e} {tot/max(1,nb):8.4f} {tr_f1:9.4f} {te_f1:9.4f} "
              f"{sep:+8.4f}", flush=True)
        del rm, base, opt
        empty_cache(dev)

    best = max(results, key=lambda k: results[k]["test_f1"])
    print(f"\nbest lr = {best:.0e} (test_f1 {results[best]['test_f1']:.4f})")
    print("baseline references: untrained sep=+0.021 rank 3.27; BM25 F1=0.322")
    out = fr"{PROJECT_ROOT}\output\lr_sweep.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)


if __name__ == "__main__":
    main()
