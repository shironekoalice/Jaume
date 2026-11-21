"""Experiment A evaluation: same four conditions, same eval code, new weights.

The training already happened (train_memory.py with --memory-output carry vs
new_slots). This script ONLY evaluates, using g3_memory's condition functions so
the measurement protocol is identical to the earlier G3 run.

Conditions (unchanged from the pre-registered set):
    concat    both chunks in one input, no state      (upper reference)
    reset     sequential, state never passed
    carried   sequential, state passed
    shuffled  sequential, state passed but from another item

Verdict rule, fixed in advance:
    memory helps <=> carried > reset AND carried > shuffled
reported per seed, plus mean and range across seeds (not a single number).

Usage:
    python scripts/eval_experiment_a.py --tag A-B6-carry --seeds 0 1 2
"""
import os as _os
PROJECT_ROOT = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import argparse
import json
import os
import statistics
import sys

REPO = fr"{PROJECT_ROOT}\code\repo"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import g3_memory as G  # noqa: E402
from jaume.inference.encode import JaumeEmbedding  # noqa: E402


def eval_one(arm_path, test_items):
    model = JaumeEmbedding.from_pretrained(arm_path)
    out = {}
    for cond in ("concat", "reset", "carried"):
        out[cond] = G.eval_condition(model, test_items, cond)
    out["shuffled"] = G.eval_condition(model, test_items, "carried", shuffle=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=fr"{PROJECT_ROOT}\data\synth_v3")
    ap.add_argument("--root", default=fr"{PROJECT_ROOT}\output\mem")
    ap.add_argument("--tags", nargs="+", default=["A-B6-carry", "A-B6-base"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    test_items = G.load_items(os.path.join(args.data, "test.jsonl"))
    chance = sum(1.0 / len(it["candidates"]) for it in test_items) / len(test_items)
    print(f"[A-eval] test={len(test_items)} chance={chance:.3f}\n")

    results = {}
    for tag in args.tags:
        rows = {}
        for sd in args.seeds:
            arm = os.path.join(args.root, f"{tag}-s{sd}", "final")
            if not os.path.exists(arm):
                print(f"  MISSING {arm}")
                continue
            r = eval_one(arm, test_items)
            rows[sd] = r
            print(f"  {tag} seed {sd}: concat={r['concat']:.3f} "
                  f"reset={r['reset']:.3f} carried={r['carried']:.3f} "
                  f"shuffled={r['shuffled']:.3f}", flush=True)
        results[tag] = rows

    print("\n" + "=" * 86)
    print(f"{'tag':14s} {'cond':9s} {'mean':>7s} {'sd':>6s} "
          f"{'min':>6s} {'max':>6s}  per-seed")
    print("=" * 86)
    for tag, rows in results.items():
        for cond in ("concat", "reset", "carried", "shuffled"):
            vals = [rows[sd][cond] for sd in sorted(rows)]
            if not vals:
                continue
            sd_ = statistics.stdev(vals) if len(vals) > 1 else 0.0
            print(f"{tag:14s} {cond:9s} {statistics.mean(vals):7.3f} {sd_:6.3f} "
                  f"{min(vals):6.3f} {max(vals):6.3f}  "
                  f"{[round(v,3) for v in vals]}")
        print("-" * 86)

    print("\nVERDICT (per tag, using means across seeds):")
    for tag, rows in results.items():
        if not rows:
            continue
        c = statistics.mean([rows[s]["carried"] for s in rows])
        r = statistics.mean([rows[s]["reset"] for s in rows])
        sh = statistics.mean([rows[s]["shuffled"] for s in rows])
        co = statistics.mean([rows[s]["concat"] for s in rows])
        helps = (c > r) and (c > sh)
        print(f"  {tag:14s} carried={c:.3f} reset={r:.3f} shuffled={sh:.3f} "
              f"concat={co:.3f} -> {'MEMORY HELPS' if helps else 'no memory benefit'}")
        print(f"       carried-reset={c-r:+.3f}  carried-shuffled={c-sh:+.3f}")

    out = os.path.join(args.data, "experiment_a_eval.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"chance": chance,
                   "results": {k: {str(s): v for s, v in r.items()}
                               for k, r in results.items()}}, f, indent=2)
    print(f"\n[A-eval] wrote {out}")


if __name__ == "__main__":
    main()
