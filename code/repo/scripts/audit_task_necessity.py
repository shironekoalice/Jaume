"""E1-E3: does the Qasper evidence-block task actually REQUIRE cross-block memory?

Read-only audit. No training, no model changes.

The question this answers (review): before spending GPU time on a 5-10 epoch
run, establish whether the task can be solved WITHOUT any recurrent state. If
independently encoding each block and matching it to the question already
works, then a memory mechanism has nothing to contribute and the task must be
replaced rather than the gate rewritten.

E1  distance stratification
      For each distance stratum, compare reset / concat / carried.
      If `reset` does not DEGRADE as distance grows, then distant evidence is
      still reachable by independent encoding -> the task does not need memory.

E2  evidence-block counterfactual
      Remove or replace the block that actually holds the evidence and see how
      much each condition's prediction changes. A model that ignores the
      evidence (position shortcut) barely moves.

E3  cross-block necessity
      Compare: full document / evidence block only / no-evidence blocks only /
      shuffled block order. If "evidence block only" already suffices, the task
      reduces to retrieval, not integration.

Usage:
    python scripts/audit_task_necessity.py --ckpt D:/Jaume/output/mem/R1-smoke-additive
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
import torch.nn.functional as F

REPO = fr"{PROJECT_ROOT}\code\repo"
sys.path.insert(0, REPO)
from jaume.device import resolve_device, has_amp, make_scaler, empty_cache, max_memory_allocated, reset_peak_memory_stats, total_memory, seed_all
sys.path.insert(0, os.path.join(REPO, "scripts"))

import train_r1 as R  # noqa: E402
import qasper_batch as QB  # noqa: E402
import test_gate_value as TGV  # noqa: E402


def score_blocks(rm, blocks, question, max_length=128):
    """Score one question against a list of blocks; return the score vector."""
    reps = QB.run_batch_states(rm, [{"blocks": blocks, "evidence_idx": 0,
                                     "question": question, "distance": 0}],
                              True, max_length, len(blocks))[0]
    q = QB.question_embs(rm, [question])
    sc = F.normalize(reps, dim=-1) @ q.squeeze(0)
    return sc.cpu()


def predict(rm, row, cond, max_blocks, max_length):
    """Return (predicted index among the given block list, full score vector)."""
    blocks = row["blocks"][:max_blocks]
    if cond == "concat":
        # concat in this experiment = independent encoding per block
        reps = QB.run_batch_states(rm, [dict(row, blocks=blocks)], False,
                                   max_length, max_blocks)[0]
        q = QB.question_embs(rm, [row["question"]])
        sc = (F.normalize(reps, dim=-1) @ q.squeeze(0)).cpu()
    else:
        reps = QB.run_batch_states(rm, [dict(row, blocks=blocks)],
                                   cond == "carried", max_length, max_blocks)[0]
        q = QB.question_embs(rm, [row["question"]])
        sc = (F.normalize(reps, dim=-1) @ q.squeeze(0)).cpu()
    return int(sc.argmax()), sc


def acc_at1(rm, rows, cond, max_blocks, max_length):
    ok = 0
    for r in rows:
        p, _ = predict(rm, r, cond, max_blocks, max_length)
        if p == r["evidence_idx"]:
            ok += 1
    return ok / len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=fr"{PROJECT_ROOT}\output\mem\R1-smoke-additive")
    ap.add_argument("--data", default=fr"{PROJECT_ROOT}\data\qasper_blocks")
    ap.add_argument("--max-blocks", type=int, default=12)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--limit", type=int, default=122)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    te = R.trim(R.load(os.path.join(args.data, "test.jsonl")), args.max_blocks)
    if args.limit:
        te = te[: args.limit]
    rm = R.make_model(args.ckpt, resolve_device(), "additive")
    rm.model.eval()
    chance = sum(1.0 / len(r["blocks"][: args.max_blocks]) for r in te) / len(te)
    print(f"checkpoint={args.ckpt}\nitems={len(te)} chance≈{chance:.4f}\n")

    # ---------------------------------------------------------------- E1
    print("=" * 72)
    print("E1  distance stratification  (does `reset` degrade with distance?)")
    print("=" * 72)
    strata = {"d=1-3": [], "d=4-7": [], "d>=8": []}
    for r in te:
        d = r["distance"]
        if d <= 0:
            continue
        k = "d=1-3" if d <= 3 else ("d=4-7" if d <= 7 else "d>=8")
        strata[k].append(r)

    print(f"{'stratum':8s} {'n':>4s} {'reset':>7s} {'concat':>7s} {'carried':>8s} "
          f"{'carried-reset':>14s}")
    e1 = {}
    for k, rows in strata.items():
        if len(rows) < 5:
            print(f"{k:8s} {len(rows):4d}   (too few)")
            continue
        a_reset = acc_at1(rm, rows, "reset", args.max_blocks, args.max_length)
        a_concat = acc_at1(rm, rows, "concat", args.max_blocks, args.max_length)
        a_carried = acc_at1(rm, rows, "carried", args.max_blocks, args.max_length)
        e1[k] = {"n": len(rows), "reset": a_reset, "concat": a_concat,
                 "carried": a_carried, "diff": a_carried - a_reset}
        print(f"{k:8s} {len(rows):4d} {a_reset:7.3f} {a_concat:7.3f} "
              f"{a_carried:8.3f} {a_carried - a_reset:+14.3f}")

    keys = [k for k in ("d=1-3", "d=4-7", "d>=8") if k in e1]
    if len(keys) >= 2:
        first, last = keys[0], keys[-1]
        drop = e1[last]["reset"] - e1[first]["reset"]
        print(f"\n  reset accuracy change from {first} to {last}: {drop:+.3f}")
        print(f"  -> {'reset DEGRADES with distance (task may need memory)' if drop < -0.05 else 'reset does NOT degrade -> independent encoding suffices'}")

    # ---------------------------------------------------------------- E2
    print()
    print("=" * 72)
    print("E2  evidence-block counterfactual  (does the model USE the evidence?)")
    print("=" * 72)
    rng = random.Random(args.seed)
    changed = 0
    n = 0
    for r in te:
        blocks = r["blocks"][: args.max_blocks]
        ei = r["evidence_idx"]
        pred0, sc0 = predict(rm, r, "carried", args.max_blocks, args.max_length)
        # replace the evidence block with an unrelated block from another paper
        other = te[rng.randrange(len(te))]
        if other["paper_id"] == r["paper_id"] or not other["blocks"]:
            continue
        swapped = list(blocks)
        swapped[ei] = other["blocks"][0]
        r2 = dict(r, blocks=swapped, evidence_idx=ei)
        pred1, sc1 = predict(rm, r2, "carried", args.max_blocks, args.max_length)
        n += 1
        if pred0 != pred1:
            changed += 1
    print(f"  replacing the true evidence block changed the prediction in "
          f"{changed}/{n} = {changed/max(1,n):.3f} of cases")
    print(f"  -> {'model IS evidence-sensitive' if changed/max(1,n) > 0.8 else 'model may rely on position/shortcut, not evidence'}")

    # ---------------------------------------------------------------- E3
    print()
    print("=" * 72)
    print("E3  cross-block necessity  (does the task need integration?)")
    print("=" * 72)
    res = {"full(reset)": [], "full(carried)": [], "evidence_vs_others": [],
           "no_evidence": [], "shuffled_order": []}
    for r in te:
        blocks = r["blocks"][: args.max_blocks]
        ei = r["evidence_idx"]

        # evidence vs. distractors: score the evidence block against a
        # randomly chosen non-evidence block from the SAME paper. This is a
        # genuine 2-way choice, unlike the previous version which scored a
        # single-block list (where argmax is trivially 0 and the "accuracy" is
        # an artefact -- that measurement has been removed).
        others = [j for j in range(len(blocks)) if j != ei]
        if not others:
            continue
        j = rng.choice(others)
        pair = [blocks[ei], blocks[j]]
        reps = QB.run_batch_states(rm, [dict(r, blocks=pair, evidence_idx=0)],
                                   False, args.max_length, 2)[0]
        q = QB.question_embs(rm, [r["question"]])
        sc = (F.normalize(reps, dim=-1) @ q.squeeze(0)).cpu()
        res["evidence_vs_others"].append(1.0 if int(sc.argmax()) == 0 else 0.0)

        # no-evidence: remove the evidence block, ask for the best remaining
        rest = [b for jj, b in enumerate(blocks) if jj != ei]
        if not rest:
            continue
        reps = QB.run_batch_states(rm, [dict(r, blocks=rest, evidence_idx=0)],
                                   True, args.max_length, len(rest))[0]
        q = QB.question_embs(rm, [r["question"]])
        sc = (F.normalize(reps, dim=-1) @ q.squeeze(0)).cpu()
        res["no_evidence"].append(float(sc.max()))

        # shuffled order: the evidence moves position, content unchanged
        perm = list(range(len(blocks)))
        rng.shuffle(perm)
        new_ei = perm.index(ei)
        shuf = [blocks[j] for j in perm]
        rr = dict(r, blocks=shuf, evidence_idx=new_ei)
        p, _ = predict(rm, rr, "carried", args.max_blocks, args.max_length)
        res["shuffled_order"].append(1.0 if p == new_ei else 0.0)

    print(f"  evidence vs a same-paper distractor (2-way) = "
          f"{statistics.mean(res['evidence_vs_others']):.3f}  "
          f"(chance 0.500; >0.5 means the block content IS discriminative)")
    full_reset = acc_at1(rm, te, "reset", args.max_blocks, args.max_length)
    full_carried = acc_at1(rm, te, "carried", args.max_blocks, args.max_length)
    print(f"  full document, reset          = {full_reset:.3f}")
    print(f"  full document, carried        = {full_carried:.3f}")
    print(f"  full document, shuffled order = "
          f"{statistics.mean(res['shuffled_order']):.3f}  "
          f"(order sensitivity)")

    print()
    print("=" * 72)
    print("DECISION (pre-registered branches)")
    print("=" * 72)
    feats = []
    if len(keys) >= 2:
        feats.append(("reset degrades with distance", e1[keys[-1]]["reset"] < e1[keys[0]]["reset"] - 0.05))
    feats.append(("model is evidence-sensitive", changed / max(1, n) > 0.8))
    feats.append(("carried > reset overall", full_carried > full_reset + 0.05))
    for name, v in feats:
        print(f"  {name:34s}: {'YES' if v else 'NO'}")

    if not any(v for _, v in feats[:2]) or not feats[2][1]:
        print("\n  => The task does NOT demonstrably require cross-block memory.")
        print("     Switch tasks rather than training longer or rewriting the gate.")
    else:
        print("\n  => Evidence supports a memory-dependent task; a longer")
        print("     training diagnostic is justified.")

    out = os.path.join(args.data, "task_necessity_audit.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"chance": chance, "e1": e1,
                   "e2_changed": changed / max(1, n),
                   "e3": {"full_reset": full_reset, "full_carried": full_carried,
                          "evidence_vs_distractor": statistics.mean(res["evidence_vs_others"]),
                          "shuffled_order": statistics.mean(res["shuffled_order"])}},
                  f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

