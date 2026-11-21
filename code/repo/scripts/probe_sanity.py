"""Sanity check for the state probe itself.

The probe reported identical R^2 for pooled_A and state_A_mean. In B5 that is
expected (pool_slots is the linear projection of the slot mean), but "expected"
and "a bug that copies one tensor into the other" look identical from the
numbers alone. This checks:

  1. are the two tensors literally equal, or merely similarly predictive?
  2. is the probe able to recover a fact that IS linearly present? Plant a
     known signal into random noise and confirm R^2 ~ 1.
  3. does a constant target give R^2 ~ 0 (no leakage through standardisation)?

If (2) fails the probe is broken and no conclusion may be drawn.
"""
import os as _os
PROJECT_ROOT = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import os
import sys

import torch

REPO = fr"{PROJECT_ROOT}\code\repo"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import probe_state_facts as P  # noqa: E402
from jaume.inference.encode import JaumeEmbedding  # noqa: E402


def main():
    items = P.load_items(fr"{PROJECT_ROOT}\data\synth_v3\test.jsonl", 200)
    model = JaumeEmbedding.from_pretrained(P.ARMS["B5"])
    reps = P.extract(model, items)

    pooled = reps["pooled_A"]
    sa = reps["state_A_mean"]
    print("=== 1. are pooled_A and state_A_mean the same tensor? ===")
    print(f"  shapes {tuple(pooled.shape)} vs {tuple(sa.shape)}")
    diff = (pooled - sa).abs()
    print(f"  max |diff| = {float(diff.max()):.6f}")
    print(f"  mean |diff| = {float(diff.mean()):.6f}")
    cos = torch.nn.functional.cosine_similarity(
        torch.nn.functional.normalize(pooled, dim=-1),
        torch.nn.functional.normalize(sa, dim=-1), dim=-1)
    print(f"  cosine min/mean/max = {float(cos.min()):.4f}/"
          f"{float(cos.mean()):.4f}/{float(cos.max()):.4f}")

    print("\n=== 2. probe recovers a PLANTED linear signal? ===")
    g = torch.Generator().manual_seed(0)
    n, d = 800, 512
    X = torch.randn(n, d, generator=g)
    w = torch.randn(d, 1, generator=g)
    y = X @ w
    y = (y - y.min()) / (y.max() - y.min())
    r2 = P.fit_probe(X, y, nonlinear=False, seed=0)
    print(f"  planted linear signal -> R^2 = {r2:.4f}  (expect ~1.0)")

    print("\n=== 3. constant target gives R^2 ~ 0? ===")
    yc = torch.full((n, 1), 0.5)
    r2c = P.fit_probe(X, yc, nonlinear=False, seed=0)
    print(f"  constant target       -> R^2 = {r2c:.4f}  (expect ~0)")

    print("\n=== 4. shuffled labels give R^2 <= 0? ===")
    ys = y[torch.randperm(n, generator=g)]
    r2s = P.fit_probe(X, ys, nonlinear=False, seed=0)
    print(f"  shuffled target       -> R^2 = {r2s:.4f}  (expect <=0)")


if __name__ == "__main__":
    main()
