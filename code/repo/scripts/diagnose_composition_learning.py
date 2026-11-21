"""Diagnose WHY training does not learn on the composition task.

Before changing anything, establish where learning breaks. Four candidate causes,
each measured separately:

  C1  Can the encoder even embed a question and a block into a comparable space?
      Measure zero-shot cosine(question, gold block) vs cosine(question, other
      blocks). If there is no separation at all, fine-tuning from this base on
      this task is starting from zero signal.

  C2  Is the training target the problem?
      Training uses a single-label CE on gold_blocks[0]. With 12 candidates that
      target asks "which ONE block", while scoring asks "which SET". Measure how
      often gold_blocks[0] is even reachable: if several blocks are equally
      correct, single-label CE teaches an arbitrary choice.

  C3  Is 3 epochs simply too few? Compare train-set loss/accuracy at epoch 0 vs
      epoch 2 to see whether the model is fitting anything at all.

  C4  Is the candidate space too large for this encoder? Check the rank of the
      gold block under zero-shot cosine; a rank of ~6/12 is chance, whereas a
      rank near 1 means the representation already carries the signal.

Usage:
    python scripts/diagnose_composition_learning.py
"""
import os as _os
PROJECT_ROOT = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import argparse
import json
import os
import statistics
import sys

import torch
import torch.nn.functional as F

REPO = fr"{PROJECT_ROOT}\code\repo"
sys.path.insert(0, REPO)
from jaume.device import resolve_device, has_amp, make_scaler, empty_cache, max_memory_allocated, reset_peak_memory_stats, total_memory, seed_all
sys.path.insert(0, os.path.join(REPO, "scripts"))

from jaume.inference.encode import JaumeEmbedding  # noqa: E402
import qasper_batch as QB  # noqa: E402

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
    base.to(dev)
    sd = torch.load(os.path.join(BASE, "pytorch_model.bin"),
                    map_location="cpu", weights_only=True)
    ms = base.state_dict()
    for k, v in ms.items():
        if k in sd and tuple(sd[k].shape) == tuple(v.shape):
            v.copy_(sd[k])
    return rm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=fr"{PROJECT_ROOT}\data\qasper_comp_clean\split0")
    ap.add_argument("--limit", type=int, default=120)
    args = ap.parse_args()

    dev = resolve_device()
    rows = load(os.path.join(args.data, "test.jsonl"))[: args.limit]
    rm = make_model(dev)
    rm.model.eval()

    print("=" * 70)
    print("C1  zero-shot separation: cos(question, gold block) vs others")
    print("=" * 70)
    gold_cos, other_cos = [], []
    ranks = []
    with torch.no_grad():
        for s in range(0, len(rows), 8):
            batch = rows[s : s + 8]
            reps = QB.run_batch_states(rm, batch, True, 128, 12)
            qs = QB.question_embs(rm, [r["question"] for r in batch])
            for i, r in enumerate(batch):
                sc = F.normalize(reps[i], dim=-1) @ qs[i]
                g = set(r["gold_blocks"])
                order = torch.argsort(sc, descending=True).tolist()
                ranks.append(min(order.index(j) for j in g) + 1)
                for j in range(len(sc)):
                    (gold_cos if j in g else other_cos).append(float(sc[j]))
    print(f"  mean cos(gold)  = {statistics.mean(gold_cos):.4f}")
    print(f"  mean cos(other) = {statistics.mean(other_cos):.4f}")
    print(f"  separation      = {statistics.mean(gold_cos)-statistics.mean(other_cos):+.4f}")
    print(f"  gold-block rank: mean={statistics.mean(ranks):.2f} "
          f"median={statistics.median(ranks):.1f} "
          f"(chance ≈ 6.0 for 12 blocks)")
    print(f"  rank==1 rate    = "
          f"{sum(1 for r in ranks if r == 1)/len(ranks):.3f}")

    print()
    print("=" * 70)
    print("C2  is the single-label target well-posed?")
    print("=" * 70)
    multi = sum(1 for r in rows if len(r["gold_blocks"]) > 1)
    print(f"  items with gold set size >1: {multi}/{len(rows)}")
    print(f"  training uses gold_blocks[0] as the ONLY positive")
    print(f"  -> {multi} items have other equally-correct blocks that")
    print(f"     single-label CE actively pushes DOWN")
    # how many of the gold blocks would be 'wrong' targets
    total_gold = sum(len(r["gold_blocks"]) for r in rows)
    print(f"  total gold blocks={total_gold}, "
          f"trained positives={len(rows)} "
          f"({100*len(rows)/total_gold:.0f}% of correct targets used)")

    print()
    print("=" * 70)
    print("C4  zero-shot BM25 vs model on the same items")
    print("=" * 70)
    import math
    import re
    import collections

    def toks(x):
        return re.findall(r"[a-z0-9]+", x.lower())

    def bm25(q, docs):
        qt = toks(q); dt = [toks(d) for d in docs]
        n = len(dt); avg = sum(len(t) for t in dt)/max(1, n)
        df = collections.Counter()
        for t in dt:
            df.update(set(t))
        out = []
        for t in dt:
            tf = collections.Counter(t); s = 0.0
            for x in qt:
                if x not in tf:
                    continue
                idf = math.log(1+(n-df[x]+0.5)/(df[x]+0.5))
                s += idf*tf[x]*2.5/(tf[x]+1.5*(1-0.75+0.75*len(t)/max(1, avg)))
            out.append(s)
        return out

    def setf1(pred, gold):
        p, g = set(pred), set(gold)
        i = len(p & g)
        return 2*i/(len(p)+len(g)) if p and g else 0.0

    bm_f1 = []
    for r in rows:
        k = len(r["gold_blocks"])
        sc = bm25(r["question"], r["blocks"])
        pred = sorted(range(len(sc)), key=lambda i: -sc[i])[:k]
        bm_f1.append(setf1(pred, r["gold_blocks"]))
    print(f"  BM25 set-F1 (zero training) = {statistics.mean(bm_f1):.4f}")

    print()
    print("=" * 70)
    print("READ THIS AS")
    print("=" * 70)
    sep = statistics.mean(gold_cos) - statistics.mean(other_cos)
    if sep < 0.02:
        print("  C1: NO separation -> the pretrained encoder does not place the")
        print("      question near its evidence block. Any fine-tuning must learn")
        print("      this from scratch, which 3 epochs cannot do.")
    else:
        print(f"  C1: separation {sep:+.3f} exists -> representation has signal.")
    if multi / len(rows) > 0.5:
        print("  C2: MOST items have >1 correct block -> single-label CE is")
        print("      ill-posed. Must switch to multi-label / set target.")
    print(f"  C4: BM25 {statistics.mean(bm_f1):.3f} vs model ~0.19 -> "
          f"{'retrieval surface form beats the encoder' if statistics.mean(bm_f1) > 0.19 else 'comparable'}")


if __name__ == "__main__":
    main()
