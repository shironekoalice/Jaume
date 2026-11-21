"""Spatial-softmax localisation: a stabler way to find "where" than classification.

The problem being solved. Locating the agent and the goal by classifying a
64-way cell from a globally pooled feature is BIMODAL: out of 3 seeds, 2
collapse to chance (agent 0.0075) and 1 reaches 1.0000, with an identical
loss and no other task involved. The optimisation has to discover, from
scratch, that the trunk's output must encode position — and it usually
does not.

The standard fix in the vision literature is to make localisation
TRANSLATION-EQUIVARIANT instead of asking a global classifier to recover
position. A spatial softmax over a feature map gives a probability
distribution over spatial positions, computed as

    p_ij = softmax_{ij}(w . f_ij)

and the expectation (x, y) = sum_ij p_ij * (x_ij, y_ij) is the estimated
coordinate. The head then predicts a heatmap target and regresses the
expectation — a much easier, better-conditioned problem than a 64-way
choice, and the loss is defined over all spatial positions at once, so the
gradient signal is dense instead of one-hot.

This file compares, over several seeds:

  A  classification  — the current design (64-way CE on a pooled feature)
  B  spatial softmax — heatmap + soft-argmax on the same trunk
  C  spatial softmax + a wall head (the full detector)

and reports mean AND worst seed for each. A design that only works on
some seeds is not a design; the metric here is the worst seed.
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


class SpatialSoftmax(nn.Module):
    """Predicts a per-cell heatmap, then takes its soft-argmax.

    Output: (B, H, W) logits over spatial positions. The soft-argmax
    expectation over a coordinate grid gives the estimated (x, y).
    """

    def __init__(self, feat_dim, out_hw):
        super().__init__()
        self.proj = nn.Conv2d(feat_dim, 1, 1)
        self.out_hw = out_hw

    def forward(self, feat):
        # feat: (B, C, H, W) -> logits (B, H, W)
        return self.proj(feat).squeeze(1)

    def soft_argmax(self, logits):
        b, h, w = logits.shape
        p = torch.softmax(logits.reshape(b, -1), dim=-1).reshape(b, h, w)
        ys = torch.linspace(0, 1, h, device=logits.device).view(1, h, 1)
        xs = torch.linspace(0, 1, w, device=logits.device).view(1, 1, w)
        cx = (p * xs).sum(dim=(1, 2))
        cy = (p * ys).sum(dim=(1, 2))
        return torch.stack([cx, cy], dim=-1)

    def argmax_cell(self, logits, n_cells=MAZE):
        """Map the predicted heatmap back to an original grid cell.

        BUG FIXED here: the first version took a plain argmax over the 16x16
        OUTPUT grid and treated the result as an 8x8 cell index. With 16x16
        output for an 8x8 maze, each real cell covers 2x2 output pixels, so
        the argmax picks one of those four pixels and the mapping is wrong —
        the measured accuracy was 0.0000 on every seed, which looked like
        "spatial softmax does not work" but was a coordinate-convention bug.

        The correct procedure is the soft-argmax expectation in normalised
        coordinates, then rounding to the original cell grid. That is what
        the head is actually trained to represent (a distribution over
        positions), so it is also the natural readout.
        """
        b = logits.shape[0]
        h, w = logits.shape[1], logits.shape[2]
        p = torch.softmax(logits.reshape(b, -1), dim=-1).reshape(b, h, w)
        ys = torch.linspace(0, 1, h, device=logits.device).view(1, h, 1)
        xs = torch.linspace(0, 1, w, device=logits.device).view(1, 1, w)
        cx = (p * xs).sum(dim=(1, 2))
        cy = (p * ys).sum(dim=(1, 2))
        # normalised [0,1] -> cell index in [0, n_cells-1]
        col = torch.clamp((cx * n_cells).long(), 0, n_cells - 1)
        row = torch.clamp((cy * n_cells).long(), 0, n_cells - 1)
        return row * n_cells + col


class SpatialTrunk(nn.Module):
    """CNN that returns a feature MAP (not a pooled vector), so the spatial
    softmax has spatial structure to work with."""

    def __init__(self, width=32, depth=3, in_size=IMG):
        super().__init__()
        layers, c = [], 3
        for i in range(depth):
            co = width * (2 ** i)
            layers += [nn.Conv2d(c, co, 3, stride=2, padding=1, bias=False),
                       nn.BatchNorm2d(co), nn.ReLU(inplace=True)]
            c = co
        self.body = nn.Sequential(*layers)
        self.out_dim = c
        self.out_hw = in_size // (2 ** depth)

    def forward(self, x):
        return self.body(x)


def heatmap_target(cell, out_hw, device=None):
    """Gaussian heatmap at the cell's centre in the output grid.

    `cell` may be a scalar or an array of scalars (numpy ints inside a
    torch tensor are not convertible with int() in one go, so the batch case
    is handled by looping).
    """
    if np.ndim(cell) > 0:
        return torch.stack([heatmap_target(int(v), out_hw, device) for v in np.asarray(cell).ravel()])
    r, c = divmod(int(cell), MAZE)
    y = (r + 0.5) / MAZE * out_hw
    x = (c + 0.5) / MAZE * out_hw
    yy = torch.arange(out_hw).view(-1, 1).float()
    xx = torch.arange(out_hw).view(1, -1).float()
    d = ((yy - y) ** 2 + (xx - x) ** 2)
    h = torch.exp(-d / (2 * (out_hw / MAZE / 2.5) ** 2))
    return h.to(device) if device is not None else h


def train_cls(seed, X, A, G, device, epochs, lr, bs):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    trunk = SpatialTrunk().to(device)
    head = nn.Linear(trunk.out_dim * trunk.out_hw * trunk.out_hw, 256).to(device)
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
            f = trunk(Xt[idx].to(device)).flatten(1)
            la = ha(head(f)); lg = hg(head(f))
            loss = lf(la, At[idx].to(device)) + lf(lg, Gt[idx].to(device))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sched.step()
    trunk.eval()
    with torch.no_grad():
        f = trunk(torch.from_numpy(X).float().div(255.).to(device)).flatten(1)
        pa = ha(head(f)).argmax(-1).cpu().numpy()
        pg = hg(head(f)).argmax(-1).cpu().numpy()
    return float((pa == A).mean()), float((pg == G).mean())


def train_ssm(seed, X, A, G, device, epochs, lr, bs, with_wall=False, W=None):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    trunk = SpatialTrunk().to(device)
    ssm_a = SpatialSoftmax(trunk.out_dim, trunk.out_hw).to(device)
    ssm_g = SpatialSoftmax(trunk.out_dim, trunk.out_hw).to(device)
    params = list(trunk.parameters()) + list(ssm_a.parameters()) + list(ssm_g.parameters())
    ssm_w = None
    if with_wall:
        ssm_w = SpatialSoftmax(trunk.out_dim, trunk.out_hw).to(device)
        params += list(ssm_w.parameters())
    Xt = torch.from_numpy(X).float().div(255.)
    At = torch.from_numpy(A); Gt = torch.from_numpy(G)
    opt = torch.optim.AdamW(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lf = nn.MSELoss()
    Wt = torch.from_numpy(W).float() if W is not None else None
    n = len(X); rng = np.random.default_rng(seed)
    hw = trunk.out_hw
    for ep in range(epochs):
        trunk.train(); perm = rng.permutation(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            feat = trunk(Xt[idx].to(device))
            la = ssm_a(feat); lg = ssm_g(feat)
            tgt_a = heatmap_target(A[idx], hw).to(device)
            tgt_g = heatmap_target(G[idx], hw).to(device)
            loss = lf(la, tgt_a) + lf(lg, tgt_g)
            if with_wall:
                lw = ssm_w(feat)
                # wall heatmap: one target per wall cell (multi-label)
                tgt = torch.zeros(len(idx), hw, hw, device=device)
                for b in range(len(idx)):
                    w = W[idx[b]]
                    for r in range(MAZE):
                        for c in range(MAZE):
                            if w[r, c] > 0.5:
                                tgt[b] = torch.maximum(tgt[b], heatmap_target(r * MAZE + c, hw, device))
                loss = loss + lf(lw, tgt)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sched.step()
    trunk.eval()
    with torch.no_grad():
        feat = trunk(torch.from_numpy(X).float().div(255.).to(device))
        pa = ssm_a.argmax_cell(ssm_a(feat)).cpu().numpy()
        pg = ssm_g.argmax_cell(ssm_g(feat)).cpu().numpy()
    return float((pa == A).mean()), float((pg == G).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--mazes", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="spatial_softmax_result.json")
    args = ap.parse_args()
    device = args.device

    X, A, G, W = build_dataset(args.mazes, seed=0)
    Xte, Ate, Gte, Wte = build_dataset(400, seed=1)
    res = {}
    for name, fn in [
        ("A_classification", lambda s: train_cls(s, X, A, G, device, args.epochs, args.lr, args.bs)),
        ("B_spatial_softmax", lambda s: train_ssm(s, X, A, G, device, args.epochs, args.lr, args.bs)),
        ("C_ssm_plus_wall", lambda s: train_ssm(s, X, A, G, device, args.epochs, args.lr, args.bs, True, W)),
    ]:
        accs = [fn(s) for s in range(args.seeds)]
        ag = [a for a, _ in accs]; gg = [b for _, b in accs]
        res[name] = dict(agent_mean=float(np.mean(ag)), agent_min=float(np.min(ag)),
                         goal_mean=float(np.mean(gg)), goal_min=float(np.min(gg)),
                         per_seed=accs)
        print(f"[{name}] agent mean {np.mean(ag):.4f} worst {np.min(ag):.4f} | "
              f"goal mean {np.mean(gg):.4f} worst {np.min(gg):.4f}", flush=True)

    json.dump(res, open(args.out, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
