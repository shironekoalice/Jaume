"""Aggregate the nine runs exactly as the frozen protocol specifies.

Protocol section 3.4:
  1. per (split, seed): paired F1 differences over TEST ITEMS
  2. per split: mean over its 3 seeds (report each seed too)
  3. across splits: mean of the 3 split-level means, with a bootstrap CI that
     resamples PAPERS (the clustering unit), not items and not runs
  4. report per-split, per-seed and paper-clustered paired statistics

Success (section 3.3) requires ALL of:
  S1 mean(carried - static)   > 0 on every split
  S2 mean(carried - shuffled) > 0 on every split
  S3 the across-split 95% CI excludes 0 for both comparisons
  S4 the conclusion holds over all nine runs (not just the best seed)

Usage:
    python scripts/aggregate_nine.py
"""
import os as _os
PROJECT_ROOT = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import argparse
import glob
import json
import os
import random
import statistics

DATA = fr"{PROJECT_ROOT}\data\qasper_comp_clean"
OUT = fr"{PROJECT_ROOT}\output\comp_nine"


def load_items(split):
    p = os.path.join(DATA, f"split{split}", "test.jsonl")
    return {json.loads(l)["id"]: json.loads(l)
            for l in open(p, encoding="utf-8") if l.strip()}


def bootstrap_paper_ci(diffs_by_paper, n_boot=4000, seed=0):
    """95% CI of the mean difference, resampling PAPERS."""
    rng = random.Random(seed)
    papers = list(diffs_by_paper)
    n = len(papers)
    means = []
    for _ in range(n_boot):
        pick = [papers[rng.randrange(n)] for _ in range(n)]
        vals = [diffs_by_paper[p] for p in pick]
        means.append(sum(vals) / len(vals))
    means.sort()
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--data", default=DATA)
    args = ap.parse_args()

    runs = {}
    for f in sorted(glob.glob(os.path.join(args.out, "split*_seed*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        runs[(d["split"], d["seed"])] = d
    print(f"found {len(runs)} runs (protocol requires 9)\n")

    splits = sorted({s for s, _ in runs})
    per_split = {}
    print(f"{'split':6s} {'seed':4s} {'carried':>8s} {'static':>8s} "
          f"{'shuffled':>9s} {'c-static':>9s} {'c-shuf':>8s}")
    print("-" * 62)
    for sp in splits:
        ds_cs, ds_csh = [], []
        for sd in sorted(k[1] for k in runs if k[0] == sp):
            r = runs[(sp, sd)]
            c = r["summary"]["carried"]
            st = r["summary"]["static"]
            sh = r["summary"]["shuffled"]
            ds_cs.append(c - st)
            ds_csh.append(c - sh)
            print(f"{sp:<6d} {sd:<4d} {c:8.4f} {st:8.4f} {sh:9.4f} "
                  f"{c-st:+9.4f} {c-sh:+8.4f}")
        per_split[sp] = {"c_minus_static": statistics.mean(ds_cs),
                         "c_minus_shuffled": statistics.mean(ds_csh),
                         "seeds_static": ds_cs, "seeds_shuffled": ds_csh,
                         "n": len(ds_cs)}
        print(f"{'':6s} {'mean':4s} {'':8s} {'':8s} {'':9s} "
              f"{per_split[sp]['c_minus_static']:+9.4f} "
              f"{per_split[sp]['c_minus_shuffled']:+8.4f}")
        print("-" * 62)

    # paper-clustered bootstrap on the split-level means
    print("\nchecking success criteria ...")
    S1 = all(per_split[sp]["c_minus_static"] > 0 for sp in splits)
    S2 = all(per_split[sp]["c_minus_shuffled"] > 0 for sp in splits)
    print(f"  S1 carried>static on every split : {'PASS' if S1 else 'FAIL'}")
    print(f"  S2 carried>shuffled on every split: {'PASS' if S2 else 'FAIL'}")

    # aggregate across splits with paper-level resampling
    for key, label in (("c_minus_static", "carried-static"),
                       ("c_minus_shuffled", "carried-shuffled")):
        vals = [per_split[sp][key] for sp in splits]
        m = statistics.mean(vals)
        # resample splits at the paper level: aggregate per-paper deltas
        by_paper = {}
        for sp in splits:
            items = load_items(sp)
            # average the per-item difference over seeds, then over that
            # paper's items
            acc = {}
            for sd in sorted(k[1] for k in runs if k[0] == sp):
                d = runs[(sp, sd)]["per_item"]
                ca = d["carried"]
                other = "static" if key == "c_minus_static" else "shuffled"
                ob = d[other]
                for iid in ca:
                    acc.setdefault(iid, []).append(ca[iid] - ob[iid])
            for iid, ds in acc.items():
                pid = items[iid]["paper_id"]
                by_paper.setdefault(pid, []).append(sum(ds) / len(ds))
        flat = {p: sum(v) / len(v) for p, v in by_paper.items()}
        lo, hi = bootstrap_paper_ci(flat)
        excl = lo > 0 or hi < 0
        print(f"  {label:18s} cross-split mean={m:+.4f}  "
              f"paper-clustered 95%CI=[{lo:+.4f}, {hi:+.4f}]  "
              f"{'excludes 0' if excl else 'INCLUDES 0'}")
        if key == "c_minus_static":
            S3a = excl
        else:
            S3b = excl
    S3 = S3a and S3b
    print(f"  S3 both CIs exclude 0            : {'PASS' if S3 else 'FAIL'}")

    alls = [runs[k]["summary"]["carried"] - runs[k]["summary"]["static"]
            for k in runs]
    S4 = all(v > 0 for v in alls)
    print(f"  S4 conclusion holds on all 9 runs: "
          f"{'PASS' if S4 else 'FAIL'} "
          f"({sum(1 for v in alls if v > 0)}/{len(alls)} runs positive)")

    print("\n" + "=" * 64)
    if len(runs) < 9:
        print(f"INCOMPLETE: {len(runs)}/9 runs. No verdict may be issued.")
    elif S1 and S2 and S3 and S4:
        print("VERDICT: meets all pre-registered criteria -> preliminary")
        print("  mechanistic evidence on a real cross-block composition task.")
    else:
        print("VERDICT: does NOT meet the pre-registered criteria.")
        print("  Do not claim the mechanism works; do not claim it fails.")
        print("  Report as: no reproducible benefit detected under this protocol.")

    out = os.path.join(args.out, "aggregate.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"runs": len(runs), "per_split": {str(k): v
                                                    for k, v in per_split.items()},
                   "S1": S1, "S2": S2, "S3": S3, "S4": S4}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
