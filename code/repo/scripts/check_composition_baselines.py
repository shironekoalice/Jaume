"""Pre-training gate: can NON-memory baselines already solve the composition task?

Review's entry criterion #2: if BM25 or a single-block baseline can already solve
it at high accuracy, the dataset must not be used as the main memory benchmark.

Task: select the SET of blocks that together carry the evidence. Because the
answer is a set, the metric is set-level:

    precision = |pred ∩ gold| / |pred|
    recall    = |pred ∩ gold| / |gold|
    f1        = harmonic mean

Baselines evaluated (all zero-training):
  B1  BM25 top-k blocks          (pure retrieval, no composition)
  B2  lexical overlap top-k
  B3  random k blocks            (sanity floor)
  B4  single best block only     (can one block suffice?)
  B5  union of top-k with k=|gold| (the strongest retrieval-style attempt)

If B1/B5 reach high F1, the "composition" framing is not warranted: retrieval
finds the blocks anyway, and a memory mechanism would have nothing to add.

Usage:
    python scripts/check_composition_baselines.py --data D:/Jaume/data/qasper_comp
"""
import os as _os
PROJECT_ROOT = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import argparse
import collections
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


def bm25_scores(query, docs):
    qt = toks(query)
    dt = [toks(d) for d in docs]
    n = len(dt)
    avgdl = sum(len(t) for t in dt) / max(1, n)
    df = collections.Counter()
    for t in dt:
        df.update(set(t))
    out = []
    for t in dt:
        tf = collections.Counter(t)
        s = 0.0
        for q in qt:
            if q not in tf:
                continue
            idf = math.log(1 + (n - df[q] + 0.5) / (df[q] + 0.5))
            s += idf * tf[q] * 2.5 / (
                tf[q] + 1.5 * (1 - 0.75 + 0.75 * len(t) / max(1, avgdl)))
        out.append(s)
    return out


def overlap_scores(query, docs):
    q = set(toks(query))
    return [len(q & set(toks(d))) / max(1, len(q | set(toks(d)))) for d in docs]


def set_f1(pred, gold):
    p, g = set(pred), set(gold)
    if not p or not g:
        return 0.0, 0.0, 0.0
    inter = len(p & g)
    prec = inter / len(p)
    rec = inter / len(g)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=fr"{PROJECT_ROOT}\data\qasper_comp")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    te = load(os.path.join(args.data, "test.jsonl"))
    rng = random.Random(args.seed)
    print(f"test items={len(te)}, avg gold set size="
          f"{statistics.mean(len(r['gold_blocks']) for r in te):.2f}\n")

    rows = collections.defaultdict(list)
    for r in te:
        docs = r["blocks"]
        gold = r["gold_blocks"]
        k = len(gold)

        bm = bm25_scores(r["question"], docs)
        ov = overlap_scores(r["question"], docs)

        preds = {
            "B1_bm25_topk": sorted(sorted(range(len(docs)),
                                          key=lambda i: -bm[i])[:k]),
            "B2_overlap_topk": sorted(sorted(range(len(docs)),
                                             key=lambda i: -ov[i])[:k]),
            "B3_random_k": sorted(rng.sample(range(len(docs)), k)),
            "B4_single_best": [max(range(len(docs)), key=lambda i: bm[i])],
        }
        for name, pred in preds.items():
            p, rec, f1 = set_f1(pred, gold)
            rows[name].append((p, rec, f1))

    print(f"{'baseline':18s} {'precision':>10s} {'recall':>8s} {'F1':>8s}")
    print("-" * 48)
    summary = {}
    for name, vals in rows.items():
        p = statistics.mean(v[0] for v in vals)
        rc = statistics.mean(v[1] for v in vals)
        f1 = statistics.mean(v[2] for v in vals)
        summary[name] = {"precision": p, "recall": rc, "f1": f1}
        print(f"{name:18s} {p:10.3f} {rc:8.3f} {f1:8.3f}")

    # exact-set accuracy: did the baseline pick exactly the gold set?
    exact = collections.Counter()
    for r in te:
        docs = r["blocks"]; gold = set(r["gold_blocks"]); k = len(gold)
        bm = bm25_scores(r["question"], docs)
        top = set(sorted(range(len(docs)), key=lambda i: -bm[i])[:k])
        exact["bm25_exact_set"] += int(top == gold)
    print(f"\nBM25 exact-set accuracy: "
          f"{exact['bm25_exact_set']}/{len(te)} = "
          f"{exact['bm25_exact_set']/len(te):.3f}")

    best = max(summary, key=lambda k: summary[k]["f1"])
    print("\n" + "=" * 60)
    print(f"Strongest baseline: {best} (F1 {summary[best]['f1']:.3f})")
    if summary[best]["f1"] > 0.6:
        print("VERDICT: a zero-training retrieval baseline already solves this")
        print("  task at F1>0.6 -> NOT suitable as the main memory benchmark.")
        print("  The task must be made harder or a different one used.")
    elif summary[best]["f1"] > 0.45:
        print("VERDICT: baselines are moderate. Usable, but any memory gain")
        print("  must be measured AGAINST these baselines, not against chance.")
    else:
        print("VERDICT: retrieval baselines are weak -> the task genuinely")
        print("  requires composition, and a memory comparison is meaningful.")

    out = os.path.join(args.data, "composition_baselines.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary,
                   "bm25_exact_set": exact["bm25_exact_set"] / len(te),
                   "n": len(te)}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
