"""Aggregate the cross-input probe JSONs into one decision table.

The point of this script is to answer, for every arm and every probe, the two
questions the plan cares about:

  1. Is the probe itself valid?  A probe is only informative if the memory
     column beats what surface matching already achieves. We report three
     floors: random, query-only similarity, and the chunk-overlap ceiling
     (bag-of-words matching against the last chunk). A probe whose
     `chunk_overlap_ceiling` is at or above the model's score measures nothing.
  2. Does carrying memory help?  `carried` must beat `reset`. If they are
     equal the memory state contributes nothing; if `carried` is below
     `reset` the state actively corrupts.

Usage:  python summarize_crossinput.py --root D:\\Jaume\\output\\mem
"""
import os as _os
PROJECT_ROOT = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import argparse
import json
import os

PROBES = ("far_retrieval", "aggregation", "conflict",
          "distractor", "isolation", "order")

# arms in the order they should be reported; the last four are the JM-R arms
ORDER = (
    "B0-mean-s1", "B2-attention-s1", "B2mlp-attnmlp-s1", "B3-latent-s1",
    "B1-cls-s1", "E1-slots4-s1", "E4-interact-s1",
    "C1-competition-s1", "C2-competition-iter3-s1", "C3-comp-iter3-div-s1",
    "C4-comp-iter3-16slot-s1", "C5-comp-attnpool-s1", "C6-comp-iter3-nogru-s1",
    "B5-recurrent-s1", "B5r-recurrent-comp-s1",
    "B6-recurrent-gate-s1", "B6r-recurrent-gcomp-s1",
)


def load(root, arm):
    path = os.path.join(root, arm, "final", "memory_crossinput.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=fr"{PROJECT_ROOT}\output\mem")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = {}
    for arm in ORDER:
        data = load(args.root, arm)
        if data is None:
            continue
        rows[arm] = data

    print(f"loaded {len(rows)} arms\n")

    # ---- probe validity: is a memory column better than the free baselines? ----
    print("=" * 100)
    print("PROBE VALIDITY  (carried vs the floors it must beat)")
    print("=" * 100)
    hdr = f"{'probe':16s} {'rand':>6s} {'query_only':>11s} {'chunk_ceil':>11s}"
    print(hdr)
    def mean(v):
        return sum(v) / len(v) if v else float("nan")

    # per-probe, per-arm validity: averages across arms would hide the fact
    # that a probe is trivially solvable on SOME arms and impossible on others
    for p in PROBES:
        parts = []
        for arm, data in rows.items():
            r = data["probes"].get(p)
            if not r:
                continue
            parts.append(f"{arm.split('-')[0]} qo={r.get('query_only', 0):.2f}/"
                         f"ceil={r.get('chunk_overlap_ceiling', 0):.2f}")
        print(f"{p:16s} " + "  ".join(parts))

    # ---- the actual question: does carrying beat resetting? ----
    print()
    print("=" * 100)
    print("MEMORY CONTRIBUTION  (carried - reset; >0 means the state helps)")
    print("=" * 100)
    head = f"{'probe':16s}"
    for arm in rows:
        head += f" {arm.split('-')[0]:>7s}"
    print(head)
    for p in PROBES:
        line = f"{p:16s}"
        for arm, data in rows.items():
            r = data["probes"].get(p)
            if not r:
                line += f" {'-':>7s}"
                continue
            line += f" {r['carried'] - r['reset']:>+7.3f}"
        print(line)

    print()
    print("=" * 100)
    print("INDEPENDENT vs CARRIED  (does turning memory on destroy a working score?)")
    print("=" * 100)
    print(head)
    for p in PROBES:
        line = f"{p:16s}"
        for arm, data in rows.items():
            r = data["probes"].get(p)
            if not r:
                line += f" {'-':>7s}"
                continue
            line += f" {r['carried'] - r['independent']:>+7.3f}"
        print(line)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
