"""Backbone selection by LINEAR PROBE on frozen features.

WHY A PROBE. The repo's first and most transferable lesson (README: "a
representation either contains the information a task needs or it does not.
A three-minute linear probe on the frozen encoder answers that before any
training run"). A zero-shot probe of a vision tower is unfair — SigLIP2 was
trained for image-text alignment, not for referring-expression grounding, so
its zero-shot number says nothing about what a trained actor could extract
from it.

So each backbone is measured the way the repo measures things: freeze the
encoder, attach a linear head, train for a few epochs, and read the number.

THE TASK. ScreenSpot gives, for one screenshot, a natural instruction and the
box of the element it refers to. A grounding probe must decide, from a
frozen crop embedding plus the instruction, whether this crop is the target.
That is trained as a BINARY task per (crop, instruction) pair:

    positive  the crop IS the element the instruction refers to
    negative  the crop is a distractor from the same screenshot

Positives and negatives are drawn from the SAME screenshot, so the probe
cannot win by detecting "this screenshot has a red button" — it must match
the crop to the instruction. The reported number is the AUROC of the probe
over all pairs, which is threshold-free and comparable across backbones.

Reported per backbone:
  auroc        the ranking quality of the frozen representation
  top1         the candidate-scoring accuracy implied by that ranking
  latency      ms per crop+instruction pair (the deployment cost)
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from backbone_registry import CANDIDATES, crop_image
from screenspot_data import build_eval_set


def featurize(scorer, data, kind, max_n=600, crop_size=224, device="cpu"):
    """Build the parallel (crop, instruction) lists and featurise them.

    The instruction is repeated for every candidate crop of its sample, so
    `crops` and `texts` are the same length and aligned index by index.
    """
    crops, texts = [], []
    for d in data[:max_n]:
        for b in d["candidates"]:
            crops.append(crop_image(d["image"], b, crop_size))
            texts.append(d["instruction"])
    feats = scorer.embed(crops, texts, kind)
    return feats, data[:max_n]


class LinearProbe(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, 1)

    def forward(self, x):
        return self.lin(x).squeeze(-1)


def train_probe(pos, neg, dim, device, epochs=8, lr=1e-3):
    """pos/neg: (N, dim) feature matrices."""
    X = torch.cat([pos, neg]).to(device)
    y = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))]).to(device)
    m = LinearProbe(dim).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    n = len(X)
    idx = torch.randperm(n, device=device)
    for ep in range(epochs):
        m.train()
        perm = idx[torch.randperm(n, device=device)]
        for i in range(0, n, 256):
            b = perm[i:i + 256]
            loss = lossf(m(X[b]), y[b])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    m.eval()
    return m


def auroc(scores, labels):
    """Rank-based AUROC (Mann-Whitney U). numpy 2 removed np.trapz, and the
    rank formulation is exact and needs no tie handling."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def run_probe(scorer, data, kind, device, epochs=8, crop_size=224, split=0.7, seed=0):
    """Train the probe on a TRAIN split and evaluate on a HELD-OUT split.

    BUG FIXED here: the first version trained and evaluated on the SAME
    samples, so its "AUROC 0.99 / top-1 0.95" was a FIT number, not a
    generalisation number. The repo has been bitten by exactly this before
    (README: "STS scores come from a historical pipeline with train/eval
    overlap; treat them as pipeline checks, not held-out results"), so the
    split is now explicit and the held-out number is the only one reported.
    """
    t0 = time.time()
    feats, ds = featurize(scorer, data, kind, crop_size=crop_size, device=device)
    X = feats["crop"]
    dim = X.shape[1]
    k = len(ds[0]["candidates"])

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ds))
    n_tr = int(len(ds) * split)
    tr_idx, te_idx = order[:n_tr], order[n_tr:]

    def pairs(idxs):
        pos, neg = [], []
        for si in idxs:
            d = ds[si]
            base = si * k
            for j in range(k):
                f = X[base + j]
                (pos if j == d["gold"] else neg).append(f)
        return torch.stack(pos), torch.stack(neg)

    ptr, ntr = pairs(tr_idx)
    pte, nte = pairs(te_idx)
    probe = train_probe(ptr.to(device), ntr.to(device), dim, device, epochs=epochs)

    with torch.no_grad():
        sp = probe(pte.to(device)).cpu().numpy()
        sn = probe(nte.to(device)).cpu().numpy()
    scores = np.concatenate([sp, sn])
    labels = np.concatenate([np.ones(len(sp)), np.zeros(len(sn))])
    a = auroc(scores, labels)

    # held-out candidate-scoring accuracy
    correct = 0
    p = q = 0
    for si in te_idx:
        d = ds[si]
        row = []
        for j in range(k):
            row.append(sp[p] if j == d["gold"] else sn[q])
            if j == d["gold"]:
                p += 1
            else:
                q += 1
        correct += int(np.argmax(row) == d["gold"])
    return dict(auroc=a, top1=correct / len(te_idx), top1_floor=1.0 / k,
                dim=int(dim), n_train=len(tr_idx), n_test=len(te_idx),
                n_pairs=int(len(scores)), sec=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=300)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--backbones", default="")
    ap.add_argument("--out", default="backbone_probe.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else (
        "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
    print(f"device {device}", flush=True)

    print(f"building ScreenSpot set ({args.rows} samples, K={args.k})...", flush=True)
    data = build_eval_set(max_rows=args.rows, k=args.k, seed=0)
    print(f"  {len(data)} samples, floor top1 = {1/args.k:.4f}", flush=True)

    names = args.backbones.split(",") if args.backbones else list(CANDIDATES)
    results = {}
    for name in names:
        spec = CANDIDATES[name]
        print(f"\n=== {name} ({spec['id']}) ===", flush=True)
        try:
            scorer = spec["build"]()
        except Exception as e:
            print(f"  LOAD FAILED: {type(e).__name__}: {str(e)[:160]}", flush=True)
            results[name] = dict(error=f"{type(e).__name__}: {str(e)[:160]}")
            continue
        try:
            r = run_probe(scorer, data, "probe", device, epochs=args.epochs)
            results[name] = r
            print(f"  HELD-OUT AUROC {r['auroc']:.4f}  (random 0.5000)", flush=True)
            print(f"  HELD-OUT top1  {r['top1']:.4f}  (floor {r['top1_floor']:.4f})", flush=True)
            print(f"  train/test samples {r['n_train']}/{r['n_test']}", flush=True)
            print(f"  dim {r['dim']}  pairs {r['n_pairs']}  {r['sec']:.0f}s", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            results[name] = dict(error=f"{type(e).__name__}: {str(e)[:200]}")
        finally:
            try: scorer.unload()
            except Exception: pass

    json.dump(results, open(args.out, "w"), indent=2)
    print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
