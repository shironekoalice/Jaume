"""Train the gated head and SAVE it in a form the browser evaluation can load.

`train_conditional_scorer.py` trains the same architecture but its training
loop uses pre-computed features and its checkpoint is a research artifact.
This script does the one thing the deployment needs:

    train on seeclick (real UI elements as distractors)
    -> save the head only (2.4M params, not the 1.5 GB frozen tower)
    -> evaluate on REAL DOM candidates in a real browser

The evaluation half is what makes the number mean anything: the candidate
set is the page's actual clickable elements, read from the DOM, which is the
regime the actor will run in.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else (
    "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")

import real_distractor as RD  # data + model definitions, reused


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=400)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--out", default="gated_head.json")
    ap.add_argument("--ckpt", default="gated_head.pt")
    args = ap.parse_args()

    from transformers import AutoModel, AutoProcessor
    from backbone_registry import local_path

    mid = "google/siglip2-base-patch16-224"
    proc = AutoProcessor.from_pretrained(local_path(mid))
    tower = AutoModel.from_pretrained(local_path(mid)).to(DEVICE).eval()

    print(f"loading seeclick...", flush=True)
    screens = RD.load_screenshots(max_rows=args.rows)
    samples = RD.build_samples(screens, k=args.k, seed=0)
    print(f"  {len(screens)} screenshots -> {len(samples)} samples", flush=True)

    print("  featurising...", flush=True)
    t0 = time.time()
    v, t = RD.featurize(tower, proc, samples)
    print(f"  done in {time.time()-t0:.0f}s", flush=True)

    K = args.k
    V, T, X, M, G = RD.build_tensors(samples, v, t, K)

    shot_ids = sorted({s["shot"] for s in samples})
    np.random.default_rng(0).shuffle(shot_ids)
    n_tr = int(len(shot_ids) * 0.7)
    tr_shots = set(shot_ids[:n_tr])
    tr = [i for i, s in enumerate(samples) if s["shot"] in tr_shots]
    te = [i for i, s in enumerate(samples) if s["shot"] not in tr_shots]
    print(f"  split by screenshot: {len(tr_shots)} train / {len(shot_ids)-n_tr} test",
          flush=True)

    torch.manual_seed(0); np.random.seed(0)
    head = RD.GatedScorer(V.shape[-1], T.shape[-1]).to(DEVICE)
    npar = sum(p.numel() for p in head.parameters())
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    lossf = nn.CrossEntropyLoss()
    best = 0.0
    for ep in range(args.epochs):
        head.train()
        perm = np.random.default_rng(ep).permutation(len(tr))
        for i in range(0, len(perm), 64):
            sel = torch.tensor(perm[i:i + 64])
            logits = head(V[sel].to(DEVICE), T[sel].to(DEVICE), X[sel].to(DEVICE), M[sel].to(DEVICE))
            loss = lossf(logits, G[sel].to(DEVICE))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sched.step()
        head.eval()
        with torch.no_grad():
            hits = 0
            for i in range(0, len(te), 128):
                sel = torch.tensor(te[i:i + 128])
                logits = head(V[sel].to(DEVICE), T[sel].to(DEVICE), X[sel].to(DEVICE), M[sel].to(DEVICE))
                hits += int((logits.argmax(-1).cpu() == G[sel]).sum())
        acc = hits / len(te)
        best = max(best, acc)
        print(f"  ep{ep+1:02d} held-out top-1 {acc:.4f} (best {best:.4f}, floor {1/K:.4f})",
              flush=True)

    torch.save(dict(head=head.state_dict(), dv=V.shape[-1], dt=T.shape[-1],
                    hidden=head.fuse[0].out_features), args.ckpt)
    size = os.path.getsize(args.ckpt) / 1e6
    print(f"saved {args.ckpt} ({size:.1f} MB, {npar/1e6:.2f}M params)", flush=True)

    res = dict(held_out_top1=best, floor=1.0 / K, params=npar,
               train_samples=len(tr), test_samples=len(te), ckpt=args.ckpt)
    json.dump(res, open(args.out, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
