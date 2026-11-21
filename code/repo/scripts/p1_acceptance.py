"""P1 acceptance test: can a memoryless model solve the synthetic task?

If B0 (mean pooling, no state) or B3 (latent slots, no cross-chunk state) score
above chance, the task is NOT a cross-chunk task and must be rebuilt before any
350M training budget is spent.

We also run the *stateful* arms (B5/B6) under `carried`, because a task on which
even they fail is unlearnable rather than merely hard -- and the distinction
matters for deciding whether to train.

Strategies are aligned with the P0 audit lesson: `independent` here means
encode each chunk separately and average (memoryless), `carried` means pass the
state across chunks (only differs for recurrent arms).
"""
import os as _os
PROJECT_ROOT = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import argparse
import json
import os
import sys

import torch

REPO = fr"{PROJECT_ROOT}\code\repo"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import eval_memory_crossinput as emc  # noqa: E402
from jaume.inference.encode import JaumeEmbedding  # noqa: E402


def load_items(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=fr"{PROJECT_ROOT}\data\synth_crossblock\test.jsonl")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--arms", default=None,
                    help="comma list of arm dirs; default = the four JM-R arms")
    args = ap.parse_args()

    items = load_items(args.data)[: args.n]
    for i, it in enumerate(items):
        it["_id"] = i
    n = len(items)
    avg_cand = sum(len(it["candidates"]) for it in items) / n
    print(f"[p1] {args.data}")
    print(f"[p1] items={n} avg_candidates={avg_cand:.2f} "
          f"random={sum(1.0/len(it['candidates']) for it in items)/n:.3f}")
    print()

    if args.arms:
        arms = [a.strip() for a in args.arms.split(",")]
    else:
        root = fr"{PROJECT_ROOT}\output\mem"
        arms = [os.path.join(root, a, "final") for a in
                ("B0-mean-s1", "B3-latent-s1", "B5-recurrent-s1",
                 "B6-recurrent-gate-s1")]

    results = {}
    for arm_path in arms:
        if not os.path.exists(arm_path):
            print(f"[p1] MISSING {arm_path}")
            continue
        name = os.path.basename(os.path.dirname(arm_path))
        model = JaumeEmbedding.from_pretrained(arm_path)
        pooling = model.model.config.pooling
        rec = pooling in emc.RECURRENT_ARMS

        row = {}
        cache = {}
        table = emc.precompute_chunks(model, items)
        for st in ("independent", "reset", "carried", "concat"):
            try:
                acc = emc.run_probe(model, items, st, "memory_only", cache, table)
            except Exception as exc:  # noqa: BLE001
                acc = float("nan")
                print(f"  {name} {st} raised {type(exc).__name__}: {str(exc)[:60]}")
            row[st] = round(acc, 4)
        row["query_only"] = round(emc.run_probe(model, items, "independent",
                                                "query_only", cache, table), 4)
        results[name] = row
        print(f"  {name:28s} pooling={pooling:22s} recurrent={rec!s:5s} "
              f"indep={row['independent']:.3f} reset={row['reset']:.3f} "
              f"carried={row['carried']:.3f} concat={row['concat']:.3f} "
              f"q_only={row['query_only']:.3f}")

    out = os.path.join(os.path.dirname(args.data), "p1_acceptance.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[p1] wrote {out}")

    print()
    print("INTERPRETATION")
    print("  memoryless arms (B0/B3) near chance + stateful arms (B5/B6) also near")
    print("  chance  -> task is NOT solvable by any current model: rebuild or train.")
    print("  memoryless near chance + stateful clearly above -> the task is the")
    print("  right instrument; training can be justified.")
    print("  memoryless ABOVE chance -> the task leaks; do NOT train.")


if __name__ == "__main__":
    main()
