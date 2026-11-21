"""The final detector, with every shortcut-blocking fix applied, plus the
end-to-end closed-loop number under it.

Design (derived from the failure analysis, see docs/jaume-actor-prototype.md §3.6-3.8):

  * NO coordinate channels — they gave the model a pixel-independent
    shortcut to the answer (variant D collapsed to 0.0213 on the worst seed).
  * A Linear projection between the flattened feature map and the heads —
    without it, features reach the head unfiltered (variant D again).
  * Default head init (logit scale ~1) — scale 10 collapsed to 0.0192.
  * SEPARATE trunks for position and walls — the repo's own dual-tower
    finding (single tower, mixed tasks: 0.650 vs 0.887) and the measured
    dense-BCE-dominates-sparse-CE effect.
  * Planning is NOT learned: the decoded map goes to an exact BFS solver.

Reported over several seeds:
  agent / goal / wall accuracy, and closed-loop success of the full pipeline.
The headline number is the WORST seed.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from envs import IMG, MAZE, STEPS, make_maze, render_maze, step_maze
from perception_solver import build_dataset, bfs_plan


class ConvTrunk(nn.Module):
    """Plain conv stack, no coordinate channels, returns a feature map."""

    def __init__(self, width=32, depth=3, in_size=IMG):
        super().__init__()
        layers, c = [], 3
        for i in range(depth):
            co = width * (2 ** i)
            layers += [nn.Conv2d(c, co, 3, stride=2, padding=1, bias=False),
                       nn.BatchNorm2d(co), nn.ReLU(inplace=True)]
            c = co
        self.body = nn.Sequential(*layers)
        hw = in_size // (2 ** depth)
        self.proj = nn.Sequential(nn.Linear(c * hw * hw, 256), nn.ReLU(inplace=True))
        self.dim = 256

    def forward(self, x):
        return self.proj(self.body(x).flatten(1))


class FinalDetector(nn.Module):
    def __init__(self, width=32, depth=3):
        super().__init__()
        self.pos_trunk = ConvTrunk(width, depth)
        self.wall_trunk = ConvTrunk(width, depth)
        self.agent_head = nn.Linear(256, MAZE * MAZE)
        self.goal_head = nn.Linear(256, MAZE * MAZE)
        self.wall_head = nn.Linear(256, MAZE * MAZE)

    def forward(self, x):
        p = self.pos_trunk(x)
        w = self.wall_trunk(x)
        return self.agent_head(p), self.goal_head(p), self.wall_head(w)


def train_one(X, A, G, W, device, epochs, lr, bs, seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    det = FinalDetector().to(device)
    Xt = torch.from_numpy(X).float().div(255.)
    At = torch.from_numpy(A); Gt = torch.from_numpy(G)
    Wt = torch.from_numpy(W).float().reshape(len(W), -1)
    opt = torch.optim.AdamW([
        {"params": list(det.pos_trunk.parameters()) + list(det.agent_head.parameters()) + list(det.goal_head.parameters()), "lr": lr},
        {"params": list(det.wall_trunk.parameters()) + list(det.wall_head.parameters()), "lr": lr},
    ], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lf_a = nn.CrossEntropyLoss()
    lf_w = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.0))
    n = len(X); rng = np.random.default_rng(seed)
    for ep in range(epochs):
        det.train(); perm = rng.permutation(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            la, lg, lw = det(Xt[idx].to(device))
            loss = 0.5 * (lf_a(la, At[idx].to(device)) + lf_a(lg, Gt[idx].to(device))) \
                + lf_w(lw, Wt[idx].to(device))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sched.step()
    return det


@torch.no_grad()
def eval_detector(det, Xte, Ate, Gte, Wte, device):
    det.eval()
    Xt = torch.from_numpy(Xte).float().div(255.)
    la, lg, lw = det(Xt.to(device))
    return dict(
        agent=float((la.argmax(-1).cpu().numpy() == Ate).mean()),
        goal=float((lg.argmax(-1).cpu().numpy() == Gte).mean()),
        wall=float(((lw > 0).cpu().numpy() == Wte.reshape(len(Wte), -1)).mean()))


@torch.no_grad()
def eval_pipeline(det, device, n_maze=150, seed=999, budget=3 * MAZE):
    rng = random.Random(seed)
    mazes = [make_maze(rng) for _ in range(n_maze)]
    succ, ticks = 0, []
    for base in mazes:
        st = base
        t = 0
        while t < budget:
            frame = render_maze(st.agent, st.goal, st.walls)
            fr = np.clip(frame * 255, 0, 255).astype(np.uint8)
            x = torch.from_numpy(fr).unsqueeze(0).to(device).float().div(255.)
            la, lg, lw = det(x)
            w = (lw[0] > 0).reshape(MAZE, MAZE).cpu().numpy().astype(bool)
            a = bfs_plan(w, int(la.argmax(-1).item()), int(lg.argmax(-1).item()))
            st, done = step_maze(st, 0 if a is None else a)
            t += 1
            if done:
                break
        succ += int(st.agent == st.goal); ticks.append(t)
    return dict(success=succ / n_maze, ticks=float(np.mean(ticks)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--mazes", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="final_detector_result.json")
    args = ap.parse_args()
    device = args.device

    X, A, G, W = build_dataset(args.mazes, seed=0)
    Xte, Ate, Gte, Wte = build_dataset(400, seed=1)
    rows = []
    for s in range(args.seeds):
        det = train_one(X, A, G, W, device, args.epochs, args.lr, args.bs, s)
        m = eval_detector(det, Xte, Ate, Gte, Wte, device)
        p = eval_pipeline(det, device, n_maze=100)
        m.update(success=p["success"], ticks=p["ticks"], seed=s)
        rows.append(m)
        print(f"  seed{s}: agent {m['agent']:.4f} goal {m['goal']:.4f} wall {m['wall']:.4f} "
              f"| closed-loop {m['success']:.3f}", flush=True)
        if s == 0:
            torch.save(det.state_dict(), "final_detector.pt")

    def agg(k):
        return dict(mean=float(np.mean([r[k] for r in rows])), min=float(np.min([r[k] for r in rows])))

    out = dict(per_seed=rows,
               agent=agg("agent"), goal=agg("goal"), wall=agg("wall"),
               closed_loop=agg("success"))
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
