"""Aggregate the concat/dual experiment across splits and seeds.

Question: does offering the state ALONGSIDE the block vector beat the block
vector alone? That is the test of whether the state carries complementary
information, as opposed to simply being a bad replacement for it.

Reports per run, then per split, then overall, with the paired deltas that
matter:
    dual - static   (does the state add anything?)
    carried - static (confirm the earlier "bad replacement" finding)
    dual - carried  (is the dual wiring strictly better than state-only?)

Usage:
    python scripts/aggregate_concat.py
"""
import os as _os
PROJECT_ROOT = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import argparse
import glob
import json
import os
import statistics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=fr"{PROJECT_ROOT}\output\comp_concat")
    args = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(os.path.join(args.root, "split*_seed*.json"))):
        rows.append(json.load(open(f, encoding="utf-8")))
    if not rows:
        print("no runs found")
        return

    print(f"{'split':>5s} {'seed':>4s} {'static':>8s} {'carried':>8s} "
          f"{'dual':>8s} {'d-static':>9s} {'c-static':>9s} {'d-carried':>10s}")
    print("-" * 72)
    by_split = {}
    ds, cs, dc = [], [], []
    for d in rows:
        s = d["summary"]
        a = s["dual"] - s["static"]
        b = s["carried"] - s["static"]
        c = s["dual"] - s["carried"]
        ds.append(a); cs.append(b); dc.append(c)
        by_split.setdefault(d["split"], []).append((a, b, c))
        print(f"{d['split']:5d} {d['seed']:4d} {s['static']:8.4f} "
              f"{s['carried']:8.4f} {s['dual']:8.4f} {a:+9.4f} {b:+9.4f} "
              f"{c:+10.4f}")
    print("-" * 72)

    print("\nper-split means:")
    for sp in sorted(by_split):
        v = by_split[sp]
        print(f"  split{sp}: dual-static={statistics.mean(x[0] for x in v):+.4f}  "
              f"carried-static={statistics.mean(x[1] for x in v):+.4f}  "
              f"dual-carried={statistics.mean(x[2] for x in v):+.4f}  "
              f"(n={len(v)})")

    print(f"\nALL {len(rows)} runs:")
    print(f"  dual - static    = {statistics.mean(ds):+.4f} "
          f"(range {min(ds):+.4f}..{max(ds):+.4f}, "
          f"positive {sum(1 for x in ds if x>0)}/{len(ds)})")
    print(f"  carried - static = {statistics.mean(cs):+.4f} "
          f"(range {min(cs):+.4f}..{max(cs):+.4f}, "
          f"negative {sum(1 for x in cs if x<0)}/{len(cs)})")
    print(f"  dual - carried   = {statistics.mean(dc):+.4f} "
          f"(range {min(dc):+.4f}..{max(dc):+.4f}, "
          f"positive {sum(1 for x in dc if x>0)}/{len(dc)})")

    print("\n" + "=" * 66)
    ok = all(x > 0 for x in ds) and all(x < 0 for x in cs)
    if ok:
        print("VERDICT: the state DOES add complementary information when given")
        print("  its own channel (dual > static in every run), while remaining")
        print("  a poor REPLACEMENT for the block vector (carried < static).")
        print("  -> The JM-R state is useful; the earlier negative results came")
        print("     from the wiring, not from the mechanism.")
    elif statistics.mean(ds) > 0:
        print("VERDICT: dual > static on average but not in every run.")
        print("  Suggestive, not conclusive. Report as a trend only.")
    else:
        print("VERDICT: the state adds nothing even with its own channel.")
        print("  The earlier negative conclusion stands for this task.")

    out = os.path.join(args.root, "aggregate.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"n": len(rows), "dual_minus_static": statistics.mean(ds),
                   "carried_minus_static": statistics.mean(cs),
                   "dual_minus_carried": statistics.mean(dc),
                   "per_run": {f"{d['split']}_{d['seed']}": d["summary"]
                               for d in rows}}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
