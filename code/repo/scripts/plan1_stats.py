"""Proper statistics for the plan-1 result, as the review requested.

The earlier report divided the mean effect by the between-seed standard
deviation and called it "27 sigma". That mixes three different quantities:

  * between-seed variation (training randomness)
  * per-test-item sampling error (n=1000)
  * the paired within-seed difference (what we actually care about)

This script reports, for each seed separately:
  1. the paired difference carried-reset and carried-shuffled
  2. a bootstrap 95% CI over TEST ITEMS for that paired difference
  3. per-class / per-item correctness so the aggregate is not the only view

and then the spread of the three paired differences across seeds.

Usage:
    python scripts/plan1_stats.py --tags P1b-s0 P1b-s1 P1b-s2
"""
import os as _os
PROJECT_ROOT = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import argparse
import json
import os
import random
import statistics
import sys

import torch

REPO = fr"{PROJECT_ROOT}\code\repo"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from jaume.inference.encode import JaumeEmbedding  # noqa: E402
import g3_memory as G  # noqa: E402
import eval_plan1 as E  # noqa: E402


def correctness(model, rep, items):
    """Per-item correctness vector (1/0), so differences can be paired."""
    rep = torch.nn.functional.normalize(rep, dim=-1)
    flat = [c for it in items for c in it["candidates"]]
    nm = []
    with torch.no_grad():
        for i in range(0, len(flat), 64):
            e = model.tokenizer(flat[i : i + 64], padding=True, truncation=True,
                                max_length=64, return_tensors="pt").to(model.device)
            emb, _ = model.model(e["input_ids"], e["attention_mask"])
            nm.append(emb.float().cpu())
    nm = torch.nn.functional.normalize(torch.cat(nm, dim=0), dim=-1)
    out = []
    ptr = 0
    for i, it in enumerate(items):
        k = len(it["candidates"])
        block = nm[ptr : ptr + k]
        ptr += k
        out.append(1.0 if int((rep[i] @ block.T).argmax()) == it["gold"] else 0.0)
    return torch.tensor(out)


def bootstrap_ci(diff, n_boot=2000, seed=0):
    """95% CI of the MEAN paired difference, resampling test items."""
    rng = random.Random(seed)
    n = diff.numel()
    vals = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        vals.append(float(diff[idx].mean()))
    vals.sort()
    lo = vals[int(0.025 * len(vals))]
    hi = vals[int(0.975 * len(vals))]
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=fr"{PROJECT_ROOT}\data\synth_v3")
    ap.add_argument("--root", default=fr"{PROJECT_ROOT}\output\mem")
    ap.add_argument("--tags", nargs="+", default=["P1b-s0", "P1b-s1", "P1b-s2"])
    args = ap.parse_args()

    items = G.load_items(os.path.join(args.data, "test.jsonl"))
    chance = sum(1.0 / len(it["candidates"]) for it in items) / len(items)
    print(f"items={len(items)} chance={chance:.4f}")
    print("Bootstrap resamples TEST ITEMS; seeds are reported separately.\n")

    per_seed = {}
    for tag in args.tags:
        arm = os.path.join(args.root, tag, "final")
        if not os.path.exists(arm):
            arm = os.path.join(args.root, tag)
        if not os.path.exists(arm):
            print(f"MISSING {arm}")
            continue
        m = JaumeEmbedding.from_pretrained(arm)
        m.model.eval()
        a_txt = [G.split_chunks(it)[0] for it in items]
        b_txt = [G.split_chunks(it)[1] for it in items]
        s_a = E.first_step_states(m, a_txt)

        c_car = correctness(m, E.step2_repr(m, b_txt, s_a), items)
        c_res = correctness(m, E.step2_repr(m, b_txt, None), items)
        g = torch.Generator().manual_seed(0)
        perm = torch.randperm(s_a.size(0), generator=g)
        c_shu = correctness(m, E.step2_repr(m, b_txt, s_a[perm]), items)
        both = [f"{a} {b}" for a, b in zip(a_txt, b_txt)]
        c_con = correctness(m, E.step2_repr(m, both, None), items)

        d_res = c_car - c_res
        d_shu = c_car - c_shu
        d_con = c_car - c_con
        lo_r, hi_r = bootstrap_ci(d_res)
        lo_s, hi_s = bootstrap_ci(d_shu)
        lo_c, hi_c = bootstrap_ci(d_con)

        per_seed[tag] = {
            "carried": float(c_car.mean()), "reset": float(c_res.mean()),
            "shuffled": float(c_shu.mean()), "concat_full": float(c_con.mean()),
            "d_reset": float(d_res.mean()), "d_reset_ci": [lo_r, hi_r],
            "d_shuffled": float(d_shu.mean()), "d_shuffled_ci": [lo_s, hi_s],
            "d_concat": float(d_con.mean()), "d_concat_ci": [lo_c, hi_c],
        }
        print(f"{tag}")
        print(f"  carried={float(c_car.mean()):.4f} reset={float(c_res.mean()):.4f} "
              f"shuffled={float(c_shu.mean()):.4f} concat={float(c_con.mean()):.4f}")
        print(f"  carried-reset    {float(d_res.mean()):+.4f}  "
              f"95%CI [{lo_r:+.4f}, {hi_r:+.4f}]  "
              f"{'excludes 0' if lo_r > 0 else 'INCLUDES 0'}")
        print(f"  carried-shuffled {float(d_shu.mean()):+.4f}  "
              f"95%CI [{lo_s:+.4f}, {hi_s:+.4f}]  "
              f"{'excludes 0' if lo_s > 0 else 'INCLUDES 0'}")
        print(f"  carried-concat   {float(d_con.mean()):+.4f}  "
              f"95%CI [{lo_c:+.4f}, {hi_c:+.4f}]  "
              f"{'excludes 0' if lo_c > 0 else 'INCLUDES 0'}")
        print()

    if len(per_seed) > 1:
        print("ACROSS SEEDS (paired differences):")
        for key in ("d_reset", "d_shuffled", "d_concat"):
            vals = [per_seed[t][key] for t in per_seed]
            same_sign = all(v > 0 for v in vals) or all(v < 0 for v in vals)
            print(f"  {key:12s} values={[round(v,4) for v in vals]} "
                  f"mean={statistics.mean(vals):+.4f} "
                  f"range={max(vals)-min(vals):.4f} "
                  f"same_sign={same_sign}")

    out = os.path.join(args.data, "plan1_stats.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"chance": chance, "n": len(items),
                   "per_seed": per_seed}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
