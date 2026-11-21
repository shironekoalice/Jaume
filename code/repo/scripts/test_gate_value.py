"""Discriminating test: is the gate VALUE the bottleneck?

The diagnosis showed the trained gate sits at ~0.52 with sd 0.006, and that
swapping a block changes the read-out 6x more than it changes the state. Two
competing explanations remain:

  (a) the update magnitude is the bottleneck -> forcing a LARGER g (more of the
      new block) should improve accuracy
  (b) the gate is not the root cause           -> varying g changes nothing

This test needs NO training: it takes a trained checkpoint, overrides the gate
output with a fixed constant, and re-scores the evidence-block task. It is a
pure inference-time intervention, so all other factors (weights, data,
question encoding) are held fixed.

Usage:
    python scripts/test_gate_value.py --ckpt D:/Jaume/output/mem/R1-smoke-additive
"""
import os as _os
PROJECT_ROOT = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import argparse
import os
import sys

import torch
import torch.nn.functional as F

REPO = fr"{PROJECT_ROOT}\code\repo"
sys.path.insert(0, REPO)
from jaume.device import resolve_device, has_amp, make_scaler, empty_cache, max_memory_allocated, reset_peak_memory_stats, total_memory, seed_all
sys.path.insert(0, os.path.join(REPO, "scripts"))

import train_r1 as R  # noqa: E402
import qasper_batch as QB  # noqa: E402


def run_with_fixed_gate(rm, rows, g_value, max_blocks, max_length):
    """Same as carried, but G2 returns a constant gate."""
    pool = rm.model.pooling
    orig_forward = pool.gate.forward

    class FixedGate:
        def __call__(self, prev, new):
            return g_value * new + (1.0 - g_value) * prev

    pool.gate.forward = lambda prev, new: g_value * new + (1.0 - g_value) * prev
    try:
        res = R.evaluate(rm, rows, "carried", max_blocks, max_length)
    finally:
        pool.gate.forward = orig_forward
    return res


def pearson(xs, ys):
    n=len(xs); mx=sum(xs)/n; my=sum(ys)/n
    num=sum((a-mx)*(b-my) for a,b in zip(xs,ys))
    dx=sum((a-mx)**2 for a in xs)**0.5; dy=sum((b-my)**2 for b in ys)**0.5
    return num/(dx*dy) if dx*dy>1e-12 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=fr"{PROJECT_ROOT}\output\mem\R1-smoke-additive")
    ap.add_argument("--data", default=fr"{PROJECT_ROOT}\data\qasper_blocks")
    ap.add_argument("--max-blocks", type=int, default=12)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    te = R.trim(R.load(os.path.join(args.data, "test.jsonl")), args.max_blocks)
    if args.limit:
        te = te[: args.limit]
    rm = R.make_model(args.ckpt, resolve_device(), "additive")
    rm.model.eval()
    chance = sum(1.0 / len(r["blocks"][: args.max_blocks]) for r in te) / len(te)
    print(f"test items={len(te)}  chance acc@1 ~ {chance:.4f}\n")

    print(f"{'gate g':>8s} {'acc@1':>8s} {'MRR':>8s}   note")
    results = {}
    for g in (0.10, 0.30, 0.50, 0.70, 0.90, 0.99):
        r = run_with_fixed_gate(rm, te, g, args.max_blocks, args.max_length)
        results[g] = r
        note = ""
        if abs(g - 0.52) < 0.03:
            note = "<- trained value is ~0.52"
        if g == 0.99:
            note = "almost pure new block (no history)"
        print(f"{g:8.2f} {r['acc1']:8.4f} {r['mrr']:8.4f}   {note}")

    vals = [results[g]["acc1"] for g in results]
    spread = max(vals) - min(vals)
    n = len(te)
    se = (chance * (1 - chance) / n) ** 0.5

    # A real "more new-block weight helps" effect must be MONOTONE in g, not
    # merely have a large spread: with n=122 a couple of lucky items move the
    # spread by 0.03. Rank correlation with g is the honest test.
    gs = list(results.keys())
    r = pearson(gs, vals)
    print(f"\nacc@1 spread across g = {spread:.4f}")
    print(f"binomial SE at n={n}  = {se:.4f}")
    print(f"corr(g, acc@1)        = {r:+.4f}")
    print(f"all values within chance ± 2SE: "
          f"{'YES' if all(abs(v - chance) < 2 * se for v in vals) else 'NO'}")
    print("\nVERDICT")
    within_noise = all(abs(v - chance) < 2 * se for v in vals)
    if within_noise or abs(r) < 0.5:
        print("  No monotone, above-noise effect of the gate value.")
        print("  -> the gate is NOT established as the bottleneck. Do not")
        print("     rewrite the update formula on this evidence.")
    else:
        best = max(results, key=lambda g: results[g]["acc1"])
        print(f"  gate value shows a systematic effect: corr={r:+.2f}, "
              f"best g={best:.2f} (acc@1 {results[best]['acc1']:.4f})")
        print("  -> changing how g is computed is justified.")


if __name__ == "__main__":
    main()

