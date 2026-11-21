"""Data-consistency check: can a lexical matcher tell the evidence block from a
distractor block of the same paper?

The audit showed the trained model scores 0.525 on that 2-way choice (chance
0.500), i.e. it does not use block content. Two explanations remain:

  (a) the DATA is mis-constructed: the "evidence block" does not actually match
      the question better than any other block does -> fix the data;
  (b) the data is fine and the model simply has not learned the task.

If a pure lexical matcher (no training, no GPU) scores clearly above 0.5 on the
same pairs, the evidence blocks really do relate to their questions and the
problem is the model. If it also sits at ~0.5, the data construction is at
fault and no amount of training will help.

Usage:
    python scripts/check_evidence_lexical.py --data D:/Jaume/data/qasper_blocks
"""
import os as _os
PROJECT_ROOT = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import argparse
import json
import math
import os
import random
import re
import statistics


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def toks(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def bm25_scores(query_terms, docs, k1=1.5, b=0.75):
    """Simple BM25 over a small doc list."""
    doc_toks = [toks(d) for d in docs]
    n = len(doc_toks)
    avgdl = sum(len(t) for t in doc_toks) / max(1, n)
    df = {}
    for t in doc_toks:
        for w in set(t):
            df[w] = df.get(w, 0) + 1
    out = []
    for t in doc_toks:
        tf = {}
        for w in t:
            tf[w] = tf.get(w, 0) + 1
        s = 0.0
        for q in query_terms:
            if q not in tf:
                continue
            idf = math.log(1 + (n - df.get(q, 0) + 0.5) / (df.get(q, 0) + 0.5))
            denom = tf[q] + k1 * (1 - b + b * len(t) / max(1, avgdl))
            s += idf * tf[q] * (k1 + 1) / denom
        out.append(s)
    return out


def overlap_acc(query, a, b):
    qa, qb = set(toks(query)), None
    q = qa
    ta, tb = set(toks(a)), set(toks(b))
    sa = len(q & ta) / max(1, len(q | ta))
    sb = len(q & tb) / max(1, len(q | tb))
    return sa > sb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=fr"{PROJECT_ROOT}\data\qasper_blocks")
    ap.add_argument("--max-blocks", type=int, default=12)
    ap.add_argument("--limit", type=int, default=122)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    te = [r for r in load(os.path.join(args.data, "test.jsonl"))
          if r["evidence_idx"] < min(len(r["blocks"]), args.max_blocks)]
    if args.limit:
        te = te[: args.limit]
    rng = random.Random(args.seed)

    ov_ok = bm_ok = 0
    n = 0
    # also: does the evidence block beat the MEAN of all other blocks?
    ov_rank = []
    for r in te:
        blocks = r["blocks"][: args.max_blocks]
        ei = r["evidence_idx"]
        others = [j for j in range(len(blocks)) if j != ei]
        if not others:
            continue
        j = rng.choice(others)
        n += 1
        if overlap_acc(r["question"], blocks[ei], blocks[j]):
            ov_ok += 1
        sc = bm25_scores(toks(r["question"]), [blocks[ei], blocks[j]])
        if sc[0] > sc[1]:
            bm_ok += 1

        # full-document ranking by BM25
        allsc = bm25_scores(toks(r["question"]), blocks)
        order = sorted(range(len(blocks)), key=lambda k: -allsc[k])
        ov_rank.append(1.0 / (order.index(ei) + 1))

    print(f"items={n}  chance(2-way)=0.500")
    print(f"  lexical overlap 2-way      = {ov_ok/n:.3f}")
    print(f"  BM25 2-way                 = {bm_ok/n:.3f}")
    print(f"  BM25 full-document MRR     = {statistics.mean(ov_rank):.3f} "
          f"(model carried MRR was ~0.27)")
    print()
    lex = max(ov_ok / n, bm_ok / n)
    if lex > 0.65:
        print("VERDICT: the evidence block IS lexically tied to its question.")
        print("  -> the data is usable; the model has simply not learned the")
        print("     task yet. A longer training run is justified.")
    elif lex > 0.55:
        print("VERDICT: weak but non-trivial lexical signal.")
        print("  -> data is probably fine, but the task may be hard for a")
        print("     lexical-style encoder. Boundary case.")
    else:
        print("VERDICT: evidence blocks are NOT distinguishable from")
        print("  distractors even by exact word overlap -> DATA CONSTRUCTION")
        print("  is at fault. Fix the data before any training.")


if __name__ == "__main__":
    main()
