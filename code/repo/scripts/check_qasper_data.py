"""Leakage and sanity checks for the Qasper block dataset.

Must pass before any training (same discipline as SciFact).

  Q1 paper-level split disjointness
  Q2 the answer text must not appear inside the QUESTION
  Q3 the evidence block must actually contain the evidence (re-verified)
  Q4 lexical baseline: can the evidence block be found by word overlap with the
     question alone? If a bag-of-words matcher already solves it, the task does
     not need memory and would repeat SciFact's ambiguity.
  Q5 distance strata are populated in the test split

Usage:
    python scripts/check_qasper_data.py --data D:/Jaume/data/qasper_blocks
"""
import os as _os
PROJECT_ROOT = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import argparse
import collections
import json
import os
import re


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def toks(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=fr"{PROJECT_ROOT}\data\qasper_blocks")
    args = ap.parse_args()

    tr = load(os.path.join(args.data, "train.jsonl"))
    va = load(os.path.join(args.data, "val.jsonl"))
    te = load(os.path.join(args.data, "test.jsonl"))
    ok = True

    # Q1
    def papers(rows):
        return {r["paper_id"] for r in rows}
    a, b, c = papers(tr), papers(va), papers(te)
    o = (len(a & b), len(a & c), len(b & c))
    print(f"Q1 paper overlap tr/va={o[0]} tr/te={o[1]} va/te={o[2]} "
          f"{'PASS' if o == (0, 0, 0) else 'FAIL'}")
    ok &= o == (0, 0, 0)

    # Q2: answer must not be inside the question
    bad = 0
    for r in te:
        ans = r["answer"].strip().lower()
        q = r["question"].strip().lower()
        if ans and len(ans) > 12 and ans in q:
            bad += 1
    print(f"Q2 answer not inside question    : {bad} "
          f"{'PASS' if bad == 0 else 'FAIL'}")
    ok &= bad == 0

    # Q3: evidence block really contains evidence-ish content
    bad = 0
    for r in te:
        blk = r["blocks"][r["evidence_idx"]].lower()
        ans_toks = toks(r["answer"])
        if ans_toks and len(ans_toks & toks(blk)) / len(ans_toks) < 0.15:
            bad += 1
    frac = bad / len(te)
    print(f"Q3 evidence block supports answer: {frac:.3f} unsupported "
          f"{'PASS' if frac < 0.25 else 'FAIL'}")
    ok &= frac < 0.25

    # Q4: question-vs-block lexical matcher
    hits = 0
    mrr = 0.0
    for r in te:
        q = toks(r["question"])
        sc = [len(q & toks(blk)) / max(1, len(q | toks(blk))) for blk in r["blocks"]]
        order = sorted(range(len(sc)), key=lambda i: -sc[i])
        if order[0] == r["evidence_idx"]:
            hits += 1
        rank = order.index(r["evidence_idx"]) + 1
        mrr += 1.0 / rank
    acc1 = hits / len(te)
    mrr /= len(te)
    chance = sum(1.0 / len(r["blocks"]) for r in te) / len(te)
    print(f"Q4 lexical baseline @1={acc1:.4f} MRR={mrr:.4f} "
          f"(chance@1={chance:.4f})")
    print(f"   {'PASS (task not solvable lexically)' if acc1 < chance + 0.15 else 'FAIL: lexical shortcut exists'}")
    ok &= acc1 < chance + 0.15

    # Q5
    print("\nQ5 distance strata (test):")
    s = collections.Counter()
    for r in te:
        d = r["distance"]
        s["d=0" if d == 0 else ("d=1-3" if d <= 3 else
                                ("d=4-7" if d <= 7 else "d>=8"))] += 1
    for k in ("d=0", "d=1-3", "d=4-7", "d>=8"):
        print(f"   {k:6s} {s.get(k,0):4d}")
    populated = all(s.get(k, 0) >= 15 for k in ("d=1-3", "d=4-7", "d>=8"))
    print(f"   {'PASS' if populated else 'FAIL: some stratum too small'}")
    ok &= populated

    print(f"\nQASPER DATA READY: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
