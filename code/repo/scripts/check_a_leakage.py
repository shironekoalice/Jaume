"""Check A: did the composition-subset construction leak test information?

Read-only. Re-runs the construction logic while TRACKING which filters use
information that must not influence the test set.

The specific concern raised in review:
  the filter `if top_bm25_block in evidence_blocks: drop` uses the official
  evidence annotation AND the item's own question, applies to every item
  INCLUDING those that will become the test split, and therefore selects the
  test set on a property of a baseline. Even though BM25 is unsupervised, the
  resulting test set is no longer a random sample: it is "items on which BM25
  failed". Any later baseline number measured on it is biased upward for
  retrieval methods and cannot be reported as an unbiased estimate.

This script quantifies the damage and produces a leak-free alternative:

  LEAKY   the current pipeline (filters applied before splitting)
  CLEAN   filters that only need the ANNOTATION and the text (evidence spans
          >=2 blocks, non-redundant, question does not quote evidence) are
          applied before splitting; the BM25-based filter is NOT used at all.

For the CLEAN set we then report the honest BM25 baseline on the test split, so
the review's entry criterion can be evaluated without selection bias.

Usage:
    python scripts/check_a_leakage.py
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
import sys

REPO = fr"{PROJECT_ROOT}\code\repo"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from make_qasper_composition import (  # noqa: E402
    paper_blocks, norm, toks, jaccard, evidence_blocks, bm25_rank, SRC, TOK)


def build(max_blocks=12, max_evidence_blocks=3, use_bm25_filter=True):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK)
    data = json.load(open(os.path.join(SRC, "qasper-train-v0.3.json"),
                          encoding="utf-8"))
    dev = json.load(open(os.path.join(SRC, "qasper-dev-v0.3.json"),
                         encoding="utf-8"))
    items, stats = [], collections.Counter()
    for split, src in (("train", data), ("dev", dev)):
        for pid, p in src.items():
            blocks = paper_blocks(p, tok)
            if len(blocks) < 2:
                continue
            if len(blocks) > max_blocks:
                blocks = blocks[:max_blocks]
            bn = [norm(b) for b in blocks]
            for qa in p.get("qas", []):
                ev = []
                for a in qa.get("answers", []):
                    ans = a.get("answer") or {}
                    if ans.get("unanswerable"):
                        continue
                    ev.extend(ans.get("evidence") or [])
                uniq = list({norm(e) for e in ev if norm(e)})
                if len(uniq) < 2:
                    stats["single_evidence"] += 1
                    continue
                hit = evidence_blocks(uniq, bn)
                if len(hit) < 2:
                    stats["evidence_in_one_block"] += 1
                    continue
                if len(hit) > max_evidence_blocks:
                    stats["too_many_blocks"] += 1
                    continue
                pieces = [s for v in hit.values() for s in v]
                maxjac = max((jaccard(pieces[i], pieces[j])
                              for i in range(len(pieces))
                              for j in range(i + 1, len(pieces))), default=0.0)
                if maxjac >= 0.5:
                    stats["redundant"] += 1
                    continue
                q = qa["question"]
                if use_bm25_filter:
                    sc = bm25_rank(q, blocks)
                    top = max(range(len(sc)), key=lambda i: sc[i])
                    if top in hit:
                        stats["single_block_bm25_wins"] += 1
                        continue
                qn = norm(q)
                if any(norm(e) in qn and len(norm(e)) > 20 for e in uniq):
                    stats["question_leaks_evidence"] += 1
                    continue
                items.append({"id": f"{split}:{pid}:{qa.get('question_id','?')}",
                              "paper_id": str(pid), "blocks": blocks,
                              "question": q, "gold_blocks": sorted(hit)})
                stats["ok"] += 1
    return items, stats


def split_by_paper(items, seed=0, val_frac=0.2, test_frac=0.2):
    rng = random.Random(seed)
    papers = sorted({it["paper_id"] for it in items})
    rng.shuffle(papers)
    n = len(papers)
    test_p = set(papers[: max(1, int(n * test_frac))])
    val_p = set(papers[max(1, int(n * test_frac)):
                       max(2, int(n * (test_frac + val_frac)))])
    out = {"train": [], "val": [], "test": []}
    for it in items:
        (out["test"] if it["paper_id"] in test_p
         else out["val"] if it["paper_id"] in val_p
         else out["train"]).append(it)
    return out


def bm25_baseline_f1(rows):
    """Honest, unbiased BM25 baseline on whatever split is given."""
    vals = []
    for r in rows:
        sc = bm25_rank(r["question"], r["blocks"])
        k = len(r["gold_blocks"])
        pred = set(sorted(range(len(sc)), key=lambda i: -sc[i])[:k])
        gold = set(r["gold_blocks"])
        inter = len(pred & gold)
        if k:
            vals.append(inter / k)
    return statistics.mean(vals) if vals else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("=" * 74)
    print("CHECK A: is the composition subset selected using test information?")
    print("=" * 74)

    print("\n--- filters used, and whether each is leak-free ---")
    print("  evidence spans >=2 blocks      : needs ANNOTATION only      OK")
    print("  evidence pieces non-redundant  : needs ANNOTATION only      OK")
    print("  question does not quote evid.  : needs text only            OK")
    print("  ** BM25 single-block wins      : needs a BASELINE's outcome **")
    print("     ^ this is applied to every item BEFORE splitting, so the")
    print("       test split is 'items where BM25 failed'. Even though BM25")
    print("       is unsupervised, this selects the test set on baseline")
    print("       performance -> LEAK, and it invalidates the baseline")
    print("       numbers that were reported afterwards.")

    leaky_items, leaky_stats = build(use_bm25_filter=True)
    clean_items, clean_stats = build(use_bm25_filter=False)

    print("\n--- construction counts ---")
    print(f"  LEAKY (BM25 filter on): {len(leaky_items)} kept  "
          f"{{'single_block_bm25_wins': {leaky_stats.get('single_block_bm25_wins',0)}}}")
    print(f"  CLEAN  (BM25 filter off): {len(clean_items)} kept")

    sl = split_by_paper(leaky_items, args.seed)
    sc = split_by_paper(clean_items, args.seed)
    print("\n--- split sizes ---")
    for name in ("train", "val", "test"):
        print(f"  {name:5s} leaky={len(sl[name]):4d}  clean={len(sc[name]):4d}")

    print("\n--- honest BM25 baseline on the TEST split ---")
    print(f"  LEAKY test (reported earlier): F1 = {bm25_baseline_f1(sl['test']):.3f}")
    print(f"  CLEAN test (unbiased)        : F1 = {bm25_baseline_f1(sc['test']):.3f}")

    print("\n--- overlap of the two test sets ---")
    it_l = {r["id"] for r in sl["test"]}
    it_c = {r["id"] for r in sc["test"]}
    print(f"  leaky test size={len(it_l)} clean test size={len(it_c)} "
          f"shared={len(it_l & it_c)}")

    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)
    print("  CHECK A: FAIL (for the current subset)")
    print("  The BM25 filter must be removed. The CLEAN construction is")
    print("  leak-free and must be used for the nine-run protocol.")
    print(f"  New test size = {len(sc['test'])} items; unbiased BM25 F1 = "
          f"{bm25_baseline_f1(sc['test']):.3f}")


if __name__ == "__main__":
    main()
