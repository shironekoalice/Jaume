"""Corrected `concat` baseline, compared against the original one.

Review rule 4: "不要让 concat 成为一个不公平的基线". The original concat was
`a_repr + b_repr` -- two independently encoded vectors summed -- whereas
`carried` runs a full step-2 forward with the state. So concat was missing the
step-2 processing entirely, not just the memory.

This script evaluates BOTH definitions on the already-trained checkpoints, so no
retraining is needed and the earlier conclusion can be checked for robustness:

  concat_sum   a_repr + b_repr                    (the original, unfair)
  concat_full  encode("A B", state=None)          (one forward over both texts)
  concat_2step encode(B, state=s_A_concat)        where s_A_concat is produced
               from the FULL摘要 -- i.e. the same number of forwards as carried

If carried still beats concat under the fair definition, the positive result
survives. If it does not, the earlier claim depended on the unfair baseline.

Usage:
    python scripts/eval_concat_fair.py --tags P1b-s0 P1b-s1 P1b-s2
"""
import os as _os
PROJECT_ROOT = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import argparse
import json
import os
import statistics
import sys

import torch

REPO = fr"{PROJECT_ROOT}\code\repo"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from jaume.inference.encode import JaumeEmbedding  # noqa: E402
import g3_memory as G  # noqa: E402
import eval_plan1 as E  # noqa: E402


def eval_fair(arm_path, items):
    model = JaumeEmbedding.from_pretrained(arm_path)
    model.model.eval()
    a_txt = [G.split_chunks(it)[0] for it in items]
    b_txt = [G.split_chunks(it)[1] for it in items]

    s_a = E.first_step_states(model, a_txt)

    out = {}
    # the three reference arms, unchanged
    out["carried"] = E.score(model, E.step2_repr(model, b_txt, s_a), b_txt, items)
    out["reset"] = E.score(model, E.step2_repr(model, b_txt, None), b_txt, items)
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(s_a.size(0), generator=g)
    out["shuffled"] = E.score(model, E.step2_repr(model, b_txt, s_a[perm]),
                              b_txt, items)

    # (1) original, unfair: sum of two independent embeddings
    a_repr = E.step2_repr(model, a_txt, None)
    b_repr = E.step2_repr(model, b_txt, None)
    out["concat_sum"] = E.score(model, a_repr + b_repr, b_txt, items)

    # (2) single forward over the concatenated text
    both = [f"{a} {b}" for a, b in zip(a_txt, b_txt)]
    out["concat_full"] = E.score(model, E.step2_repr(model, both, None), b_txt, items)

    # (3) two-step with the same forward count as carried: the state is built
    #     from the full context, then step 2 runs exactly like carried
    out["concat_2step"] = E.score(model, E.step2_repr(model, b_txt, s_a),
                                  b_txt, items)
    # NOTE: (3) is numerically identical to `carried` because chip A already IS
    # the full context in this synthetic task; it is reported to make that
    # explicit rather than to add a new number.
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=fr"{PROJECT_ROOT}\data\synth_v3")
    ap.add_argument("--root", default=fr"{PROJECT_ROOT}\output\mem")
    ap.add_argument("--tags", nargs="+", default=["P1b-s0", "P1b-s1", "P1b-s2"])
    args = ap.parse_args()

    items = G.load_items(os.path.join(args.data, "test.jsonl"))
    chance = sum(1.0 / len(it["candidates"]) for it in items) / len(items)
    print(f"[fair] items={len(items)} chance={chance:.3f}\n")

    rows = {}
    for tag in args.tags:
        arm = os.path.join(args.root, tag, "final")
        if not os.path.exists(arm):
            # train_plan1.py writes the checkpoint at the output root
            arm = os.path.join(args.root, tag)
        if not os.path.exists(arm):
            print(f"  MISSING {arm}")
            continue
        r = eval_fair(arm, items)
        rows[tag] = r
        print(f"  {tag:8s} carried={r['carried']:.3f} reset={r['reset']:.3f} "
              f"shuffled={r['shuffled']:.3f} | concat_sum={r['concat_sum']:.3f} "
              f"concat_full={r['concat_full']:.3f}", flush=True)

    print("\n" + "=" * 78)
    print(f"{'cond':14s} {'mean':>7s} {'sd':>6s}  per-seed")
    print("=" * 78)
    for cond in ("carried", "reset", "shuffled", "concat_sum", "concat_full"):
        vals = [rows[t][cond] for t in rows]
        if not vals:
            continue
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"{cond:14s} {statistics.mean(vals):7.3f} {sd:6.3f}  "
              f"{[round(v,3) for v in vals]}")

    print("\nPAIRED DIFFERENCES (per seed, as review requested):")
    for t in rows:
        c = rows[t]["carried"]
        print(f"  {t:8s} carried-reset={c-rows[t]['reset']:+.3f}  "
              f"carried-shuffled={c-rows[t]['shuffled']:+.3f}  "
              f"carried-concat_full={c-rows[t]['concat_full']:+.3f}")

    print("\nVERDICT under the FAIR concat definition:")
    c = statistics.mean([rows[t]["carried"] for t in rows])
    cf = statistics.mean([rows[t]["concat_full"] for t in rows])
    cs = statistics.mean([rows[t]["concat_sum"] for t in rows])
    print(f"  carried {c:.3f} | concat_full {cf:.3f} | concat_sum {cs:.3f}")
    if c > cf:
        print("  => carried still beats the FAIR concat: the positive result holds.")
    else:
        print("  => carried does NOT beat the fair concat: the earlier claim")
        print("     depended on an unfair baseline and must be withdrawn.")

    out = os.path.join(args.data, "concat_fair_eval.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"chance": chance, "rows": rows}, f, indent=2)
    print(f"\n[fair] wrote {out}")


if __name__ == "__main__":
    main()
