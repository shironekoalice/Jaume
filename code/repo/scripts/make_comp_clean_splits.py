"""Build the LEAK-FREE composition dataset with 3 FIXED paper-level splits.

Per the frozen protocol (memory-preregistration.md):

  * filters use ONLY the official annotation or the text -- the BM25-based
    "single block wins" filter is REMOVED because it selects the test set on a
    baseline's performance (check A FAIL)
  * three fixed splits, seeds 0/1/2, papers never straddle splits
  * test size is reported for each split

Output layout:
    data/qasper_comp_clean/split0/{train,val,test}.jsonl
    data/qasper_comp_clean/split1/...
    data/qasper_comp_clean/split2/...

Usage:
    python scripts/make_comp_clean_splits.py
"""
import os as _os
PROJECT_ROOT = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import argparse
import collections
import json
import os
import random
import sys

REPO = fr"{PROJECT_ROOT}\code\repo"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from check_a_leakage import build, bm25_baseline_f1  # noqa: E402


def split_by_paper(items, seed, val_frac=0.2, test_frac=0.2):
    """Paper-level split, no paper crosses boundaries."""
    rng = random.Random(seed)
    papers = sorted({it["paper_id"] for it in items})
    rng.shuffle(papers)
    n = len(papers)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    test_p = set(papers[:n_test])
    val_p = set(papers[n_test:n_test + n_val])
    out = {"train": [], "val": [], "test": []}
    for it in items:
        if it["paper_id"] in test_p:
            out["test"].append(it)
        elif it["paper_id"] in val_p:
            out["val"].append(it)
        else:
            out["train"].append(it)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=fr"{PROJECT_ROOT}\data\qasper_comp_clean")
    ap.add_argument("--max-blocks", type=int, default=12)
    ap.add_argument("--max-evidence-blocks", type=int, default=3)
    ap.add_argument("--split-seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    print("building LEAK-FREE composition items (no BM25 filter)...")
    items, stats = build(use_bm25_filter=False,
                         max_blocks=args.max_blocks,
                         max_evidence_blocks=args.max_evidence_blocks)
    print(f"  items={len(items)} stats={dict(stats)}")

    os.makedirs(args.out, exist_ok=True)
    meta = {"filters": {
        "evidence_spans_2plus_blocks": True,
        "redundancy_jaccard_lt_0.5": True,
        "question_does_not_quote_evidence": True,
        "bm25_single_block_filter": False,
    }, "stats": dict(stats), "splits": {}}

    for seed in args.split_seeds:
        parts = split_by_paper(items, seed)
        d = os.path.join(args.out, f"split{seed}")
        os.makedirs(d, exist_ok=True)
        for name, rows in parts.items():
            with open(os.path.join(d, f"{name}.jsonl"), "w",
                      encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        def papers(rows):
            return {r["paper_id"] for r in rows}
        a, b, c = papers(parts["train"]), papers(parts["val"]), papers(parts["test"])
        overlap = (len(a & b), len(a & c), len(b & c))
        bm = bm25_baseline_f1(parts["test"])
        ev = collections.Counter(len(r["gold_blocks"]) for r in parts["test"])
        meta["splits"][f"split{seed}"] = {
            "sizes": {k: len(v) for k, v in parts.items()},
            "paper_overlap": overlap,
            "test_bm25_f1_unbiased": bm,
            "test_evidence_blocks": dict(ev),
        }
        print(f"\n[split{seed}] train={len(parts['train'])} "
              f"val={len(parts['val'])} test={len(parts['test'])}")
        print(f"  papers: train={len(a)} val={len(b)} test={len(c)}  "
              f"overlap={overlap}")
        print(f"  test gold-set sizes={dict(ev)}  "
              f"unbiased BM25 F1={bm:.3f}")

    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
