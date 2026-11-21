"""Where does the bimodal collapse come from? An implementation attribution.

OBSERVED (same task, same loss, 3 fixed seeds each):
  detector_ablation.py  -> SpatialCNN trunk  : agent 0.0150 / 1.0000 / 0.0075  (bimodal)
  spatial_softmax.py   -> SpatialTrunk trunk: agent 1.0000 / 1.0000 / 1.0000  (stable)

The two trunks differ in three ways, and this script separates them:

  D1  COORDINATE CHANNELS
      SpatialCNN concatenates two normalised (x, y) maps to the input.
      SpatialTrunk does not.

  D2  PROJECTION WIDTH
      SpatialCNN has Linear(C*H*W -> 256) with C=128, H=W=16  => 131k params.
      SpatialTrunk has no projection; the feature map goes straight to the head.

  D3  PARAMETER COUNT / OPTIMISATION GEOMETRY
      A consequence of D2: the big projection can memorise a shortcut that
      makes the CE loss drop without the trunk learning spatial selectivity.

Each variant is trained with an IDENTICAL recipe and evaluated identically.
The reported metric is the WORST seed, because the whole point is that a
design which only works on some initialisations is not a design.
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


class TrunkA(nn.Module):
    """SpatialCNN as used in detector_ablation (coord channels + 256 proj)."""

    def __init__(self):
        super().__init__()
        self.net = SpatialCNN(3, 32, 3, in_size=IMG, out_dim=256)
        self.dim = 256

    def forward(self, x):
        return self.net(x)


class TrunkB(nn.Module):
    """Pure conv stack, feature map flattened by the head (no coord channels)."""

    def __init__(self):
        super().__init__()
        layers, c = [], 3
        for i in range(3):
            co = 32 * (2 ** i)
            layers += [nn.Conv2d(c, co, 3, stride=2, padding=1, bias=False),
                       nn.BatchNorm2d(co), nn.ReLU(inplace=True)]
            c = co
        self.body = nn.Sequential(*layers)
        self.dim = c * (IMG // 8) * (IMG // 8)

    def forward(self, x):
        return self.body(x).flatten(1)


class TrunkC(nn.Module):
    """Pure conv stack BUT with an explicit 256-d projection (isolates D2)."""

    def __init__(self):
        super().__init__()
        layers, c = [], 3
        for i in range(3):
            co = 32 * (2 ** i)
            layers += [nn.Conv2d(c, co, 3, stride=2, padding=1, bias=False),
                       nn.BatchNorm2d(co), nn.ReLU(inplace=True)]
            c = co
        self.body = nn.Sequential(*layers)
        self.proj = nn.Sequential(nn.Linear(c * (IMG // 8) * (IMG // 8), 256), nn.ReLU(inplace=True))
        self.dim = 256

    def forward(self, x):
        return self.proj(self.body(x).flatten(1))


class TrunkD(nn.Module):
    """Coordinate channels WITHOUT the big projection (isolates D1)."""

    def __init__(self):
        super().__init__()
        layers, c = [], 5  # 3 image + 2 coord
        for i in range(3):
            co = 32 * (2 ** i)
            layers += [nn.Conv2d(c, co, 3, stride=2, padding=1, bias=False),
                       nn.BatchNorm2d(co), nn.ReLU(inplace=True)]
            c = co
        self.body = nn.Sequential(*layers)
        self.dim = c * (IMG // 8) * (IMG // 8)

    def forward(self, x):
        b, _, h, w = x.shape
        ys = torch.linspace(-1, 1, h, device=x.device).view(1, 1, h, 1).expand(b, 1, h, w)
        xs = torch.linspace(-1, 1, w, device=x.device).view(1, 1, 1, w).expand(b, 1, h, w)
        return self.body(torch.cat([x, ys, xs], 1)).flatten(1)


class TrunkE(nn.Module):
    """TrunkA + a SECOND unused SpatialCNN branch (the wall trunk from
    detector_ablation.Det) that is run in forward but contributes no loss.

    This is the minimal difference between trunk_attribution.TrunkA (stable
    on 3/3 seeds) and detector_ablation.Det (bimodal: 0.015 / 1.000 / 0.0075)
    under the "position only" loss. If E collapses, the presence of the
    second BatchNorm-ed branch is the cause; if E is stable, the cause is
    something else and the earlier bimodal measurement needs another
    explanation.
    """

    def __init__(self):
        super().__init__()
        self.net = SpatialCNN(3, 32, 3, in_size=IMG, out_dim=256)
        self.unused = SpatialCNN(3, 32, 3, in_size=IMG, out_dim=256)
        self.dim = 256

    def forward(self, x):
        p = self.net(x)
        _ = self.unused(x)   # runs, contributes nothing to the loss
        return p


def run(trunk_cls, seeds, X, A, G, device, epochs, lr, bs):
    accs = []
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        trunk = trunk_cls().to(device)
        head = nn.Sequential(nn.Linear(trunk.dim, 256), nn.ReLU(inplace=True)).to(device)
        ha = nn.Linear(256, MAZE * MAZE).to(device)
        hg = nn.Linear(256, MAZE * MAZE).to(device)
        Xt = torch.from_numpy(X).float().div(255.)
        At = torch.from_numpy(A); Gt = torch.from_numpy(G)
        opt = torch.optim.AdamW(list(trunk.parameters()) + list(head.parameters()) +
                                list(ha.parameters()) + list(hg.parameters()), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        lf = nn.CrossEntropyLoss()
        n = len(X); rng = np.random.default_rng(seed)
        for ep in range(epochs):
            trunk.train(); perm = rng.permutation(n)
            for i in range(0, n, bs):
                idx = perm[i:i+bs]
                f = head(trunk(Xt[idx].to(device)))
                loss = 0.5 * (lf(ha(f), At[idx].to(device)) + lf(hg(f), Gt[idx].to(device)))
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            sched.step()
        trunk.eval()
        with torch.no_grad():
            f = head(trunk(torch.from_numpy(X).float().div(255.).to(device)))
            pa = ha(f).argmax(-1).cpu().numpy(); pg = hg(f).argmax(-1).cpu().numpy()
        accs.append((float((pa == A).mean()), float((pg == G).mean())))
        print(f"    seed{seed}: agent {accs[-1][0]:.4f} goal {accs[-1][1]:.4f}", flush=True)
    ag = [a for a, _ in accs]
    return dict(per_seed=accs, agent_mean=float(np.mean(ag)), agent_min=float(np.min(ag)),
                goal_mean=float(np.mean([g for _, g in accs])),
                goal_min=float(np.min([g for _, g in accs])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--mazes", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="trunk_attribution.json")
    args = ap.parse_args()
    device = args.device

    X, A, G, W = build_dataset(args.mazes, seed=0)
    variants = {
        "A_spatialCNN_coord_proj": TrunkA,   # stable
        "B_conv_only": TrunkB,               # stable
        "C_conv_proj256": TrunkC,            # coord-free + big proj
        "D_coord_no_proj": TrunkD,           # coord + no big proj
        "E_plus_unused_branch": TrunkE,      # A + a second dead trunk
    }
    res = {}
    for name, cls in variants.items():
        print(f"[{name}]", flush=True)
        r = run(cls, range(args.seeds), X, A, G, device, args.epochs, args.lr, args.bs)
        res[name] = r
        print(f"  -> agent mean {r['agent_mean']:.4f} worst {r['agent_min']:.4f} | "
              f"goal mean {r['goal_mean']:.4f} worst {r['goal_min']:.4f}", flush=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items() if kk != "per_seed"}
                      for k, v in res.items()}, indent=2))


if __name__ == "__main__":
    main()
