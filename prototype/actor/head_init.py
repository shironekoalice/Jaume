"""Is the collapse caused by the HEAD's initialisation?

The decisive clue: `trunk_attribution.TrunkA` (one SpatialCNN, position CE
only) is STABLE on 3/3 seeds, while `detector_ablation.Det` (the same
SpatialCNN for positions, PLUS a second SpatialCNN for walls, PLUS a wall
head, with the SAME position-only loss) is BIMODAL on the same seeds:
0.0150 / 1.0000 / 0.0075.

The two model classes differ in the ORDER and NUMBER of randomly initialised
modules, so the head weights they receive are drawn from different points of
the RNG stream. A plausible mechanism: a 64-way head whose initial logits
are badly scaled makes the CE loss push the trunk toward a degenerate
"constant logit" solution instead of a spatially selective one, and once the
trunk is constant the head cannot recover.

This script isolates that: the TRUNK initialisation is held fixed (same seed,
same construction) and only the HEAD initialisation is varied, by scaling
its weights and/or zeroing its bias. If the collapse tracks the head's
initial logit scale, the fix is architectural (normalise the head, or use a
spatial-softmax head) rather than "train longer".

Reported: accuracy over several head scales x seeds.
"""
from __future__ import annotations

import argparse
import json
import random

import numpy as np
import torch
import torch.nn as nn

from envs import IMG, MAZE
from model import SpatialCNN
from perception_solver import build_dataset


def make_head(dim, scale, device, zero_bias=False, seed=0):
    torch.manual_seed(seed)
    h = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(inplace=True)).to(device)
    ha = nn.Linear(256, MAZE * MAZE).to(device)
    hg = nn.Linear(256, MAZE * MAZE).to(device)
    with torch.no_grad():
        for m in (h[0], ha, hg):
            m.weight.mul_(scale)
            if zero_bias:
                m.bias.zero_()
    return h, ha, hg


def run_one(trunk_seed, head_seed, head_scale, X, A, G, device, epochs, lr, bs,
            zero_bias=False):
    torch.manual_seed(trunk_seed); np.random.seed(trunk_seed); random.seed(trunk_seed)
    trunk = SpatialCNN(3, 32, 3, in_size=IMG, out_dim=256).to(device)
    h, ha, hg = make_head(256, head_scale, device, zero_bias, seed=head_seed)
    Xt = torch.from_numpy(X).float().div(255.)
    At = torch.from_numpy(A); Gt = torch.from_numpy(G)
    opt = torch.optim.AdamW(list(trunk.parameters()) + list(h.parameters()) +
                            list(ha.parameters()) + list(hg.parameters()), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lf = nn.CrossEntropyLoss()
    n = len(X); rng = np.random.default_rng(trunk_seed)
    for ep in range(epochs):
        trunk.train(); perm = rng.permutation(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            f = h(trunk(Xt[idx].to(device)))
            loss = 0.5 * (lf(ha(f), At[idx].to(device)) + lf(hg(f), Gt[idx].to(device)))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sched.step()
    trunk.eval()
    with torch.no_grad():
        f = h(trunk(torch.from_numpy(X).float().div(255.).to(device)))
        pa = ha(f).argmax(-1).cpu().numpy(); pg = hg(f).argmax(-1).cpu().numpy()
    return float((pa == A).mean()), float((pg == G).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--mazes", type=int, default=1200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--out", default="head_init_result.json")
    args = ap.parse_args()
    device = args.device

    X, A, G, W = build_dataset(args.mazes, seed=0)
    scales = [0.1, 1.0, 10.0]
    res = {}
    for sc in scales:
        accs = []
        for ts in range(3):
            a, g = run_one(ts, 100 + ts, sc, X, A, G, device, args.epochs, args.lr, args.bs)
            accs.append((a, g))
            print(f"  head_scale {sc:5.1f} trunk_seed {ts}: agent {a:.4f} goal {g:.4f}", flush=True)
        ag = [a for a, _ in accs]
        res[f"scale_{sc}"] = dict(agent_mean=float(np.mean(ag)), agent_min=float(np.min(ag)),
                                  per_seed=accs)
        print(f"  -> scale {sc}: agent mean {np.mean(ag):.4f} worst {np.min(ag):.4f}", flush=True)

    json.dump(res, open(args.out, "w"), indent=2)
    print(json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items() if kk != "per_seed"}
                      for k, v in res.items()}, indent=2))


if __name__ == "__main__":
    main()
