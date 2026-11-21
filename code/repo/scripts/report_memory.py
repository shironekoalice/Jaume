"""Aggregates the Jaume-Memory sweep into the report table (plan §3: mean±std
over >=3 seeds; §9: judge against the parameter-matched B2 control).

Usage:
    python scripts/report_memory.py --root D:/Jaume/output/mem --tasks sickr
"""
import argparse
import glob
import json
import os
import statistics


def load(root, arm):
    """Collects per-seed holdout + cost numbers for one arm prefix."""
    rows = []
    for d in sorted(glob.glob(os.path.join(root, f"{arm}-s*"))):
        res_path = os.path.join(d, "final", "holdout_results.json")
        sum_path = os.path.join(d, "train_summary.json")
        if not os.path.exists(res_path):
            continue
        try:
            res = json.load(open(res_path, encoding="utf-8"))
            sm = json.load(open(sum_path, encoding="utf-8")) if os.path.exists(sum_path) else {}
        except Exception:
            continue
        rows.append({"seed": d.rsplit("-s", 1)[-1], "results": res["tasks"], "cost": sm})
    return rows


def fmt(v):
    return f"{v:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="D:/Jaume/output/mem")
    ap.add_argument("--arms", default="B0-mean,B1-cls,B2-attention,B2mlp-attnmlp,B3-latent")
    ap.add_argument("--task", default="sickr")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    print(f"{'arm':<18}{'n':>3}  {args.task} (mean±std over seeds)   {'params(M)':>10} {'pairs/s':>8} {'VRAM(MB)':>9}")
    print("-" * 82)

    table = {}
    for arm in arms:
        rows = load(args.root, arm)
        if not rows:
            print(f"{arm:<18}{0:>3}  -")
            continue
        vals = [r["results"][args.task]["value"] for r in rows if args.task in r["results"]]
        if not vals:
            print(f"{arm:<18}{len(rows):>3}  (no {args.task} result)")
            continue
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        params = [r["cost"].get("params_M") for r in rows if r["cost"].get("params_M")]
        sp = [r["cost"].get("pairs_per_s") for r in rows if r["cost"].get("pairs_per_s")]
        vm = [r["cost"].get("peak_vram_mb") for r in rows if r["cost"].get("peak_vram_mb")]
        table[arm] = {"mean": mean, "std": std, "n": len(vals), "vals": vals}
        print(f"{arm:<18}{len(vals):>3}  {mean:.4f} ± {std:.4f}   "
              f"{statistics.mean(params) if params else 0:>10.2f} "
              f"{statistics.mean(sp) if sp else 0:>8.1f} "
              f"{statistics.mean(vm) if vm else 0:>9.1f}")
        print(f"{'':<18}     seeds: {[round(v, 4) for v in vals]}")

    # plan §9 verdict: JM-S vs the parameter-matched control
    print()
    if "B3-latent" in table and "B2mlp-attnmlp" in table:
        b3, b2 = table["B3-latent"], table["B2mlp-attnmlp"]
        d = b3["mean"] - b2["mean"]
        pooled = (b3["std"] ** 2 + b2["std"] ** 2) ** 0.5
        print(f"[verdict] B3(latent) - B2mlp(param-matched) = {d:+.4f}   seed-pooled sd = {pooled:.4f}")
        print(f"[verdict] {'JM-S ahead' if d > 0 else 'JM-S behind'} of the parameter-matched "
              f"attention-pooling control by {abs(d):.4f} "
              f"({'within' if abs(d) < pooled else 'beyond'} ~1 sd of seed noise)")
    if "B3-latent" in table and "B2-attention" in table:
        b3, b2 = table["B3-latent"], table["B2-attention"]
        d = b3["mean"] - b2["mean"]
        print(f"[verdict] B3(latent) - B2(attention, 1K params) = {d:+.4f}")
    if "B3-latent" in table and "B0-mean" in table:
        d = table["B3-latent"]["mean"] - table["B0-mean"]["mean"]
        print(f"[verdict] B3(latent) - B0(mean) = {d:+.4f}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(table, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
