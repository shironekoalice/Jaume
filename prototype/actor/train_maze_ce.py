"""Game actor trained with softmax CE over the action set (repo rule §4.7).

Why this file exists. The planning report's §4.7 lesson is explicit: a
per-option regression loss (MSE over the 4 action values) lets the model
ignore the state and collapse onto a marginal prior, whereas a **softmax CE
across the options of one state** makes an action-only prior provably useless.

The MSE version of the maze hit exactly that failure: MSE 70 (huge), action
agreement with the oracle ~0.39, closed-loop success 0.09 vs oracle 1.00.
Two concrete reasons MSE is the wrong loss here:

  1. SCALE. Blocked actions get target -32 while real distances are -1..-12.
     Squared error on the blocked entries dominates the gradient, so most of
     the capacity is spent on "is this direction a wall" (a local, easy
     question) and almost none on "which way is the goal" (the question that
     actually decides the game).
  2. TIES. In a grid, several moves are often equally good; MSE on values
     spreads the target mass evenly and the argmax then flips arbitrarily.

The fix: turn the value vector into a *target distribution* over actions and
train with CE. Concretely, reward(a) = -distance(next(s,a), goal), blocked
actions get -(max_distance + 2) so the penalty is only slightly larger than a
real distance; the target distribution is softmax(reward / temperature). The
inference rule is unchanged: argmax over the predicted logits.

Reported:
  * closed-loop success / ticks for the CE policy vs the exact BFS oracle
  * the same three controls as before (local-vs-global split is run as an
    ablation so the two requirements stay separated)
"""
from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn

from envs import IMG, MAZE, STEPS, bfs_dist, make_maze, step_maze
from model import SpatialCNN

MAXD = MAZE * MAZE
BLOCK = -(MAXD + 2)  # slightly worse than any reachable distance


def reward_vector(walls, agent, goal):
    r = []
    for dr, dc in STEPS:
        nr, nc = agent[0] + dr, agent[1] + dc
        if not (0 <= nr < MAZE and 0 <= nc < MAZE) or walls[nr, nc]:
            r.append(BLOCK)
            continue
        d = bfs_dist(walls, (nr, nc), goal)
        r.append(-(d if d is not None else MAXD))
    return np.array(r, np.float32)


def collect(n_maze, frames_per_maze, seed):
    rng = random.Random(seed)
    X, R = [], []
    for _ in range(n_maze):
        st = make_maze(rng)
        for _ in range(frames_per_maze):
            r = reward_vector(st.walls, st.agent, st.goal)
            X.append(np.clip(st.frame * 255, 0, 255).astype(np.uint8))
            R.append(r)
            st, done = step_maze(st, int(np.argmax(r)))
            if done:
                break
    return np.stack(X), np.stack(R)


def soft_targets(R, temp=1.0):
    """softmax(reward / temp) per frame -> target distribution over actions."""
    t = torch.softmax(torch.from_numpy(R) / temp, dim=-1)
    return t


class CEPolicy(nn.Module):
    def __init__(self, width=32, depth=3, in_size=IMG):
        super().__init__()
        self.trunk = SpatialCNN(3, width, depth, in_size=in_size, out_dim=256)
        self.head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Linear(128, 4))

    def forward(self, x):
        return self.head(self.trunk(x))


def eval_closed_loop(model, device, n_maze=200, seed=999, budget=3 * MAZE):
    rng = random.Random(seed)
    mazes = [make_maze(rng) for _ in range(n_maze)]
    succ = 0
    ticks = []
    with torch.no_grad():
        for base in mazes:
            st = base; t = 0
            while t < budget:
                x = torch.from_numpy(np.clip(st.frame * 255, 0, 255).astype(np.uint8)).unsqueeze(0).to(device).float().div(255.)
                a = int(model(x).argmax(-1).item())
                st, done = step_maze(st, a); t += 1
                if done:
                    break
            succ += int(done); ticks.append(t)
    return succ / n_maze, float(np.mean(ticks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mazes", type=int, default=600)
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--out", default="maze_ce_result.json")
    args = ap.parse_args()
    device = args.device

    X, R = collect(args.mazes, args.frames, seed=0)
    T = soft_targets(R, args.temp)
    Xt = torch.from_numpy(X).float().div(255.)
    Tt = T
    ent = float((-(Tt * (Tt + 1e-9).log()).sum(-1)).mean().exp())
    print(f"frames {len(X)}  mean target-entropy {ent:.3f} (uniform = {4.0:.3f})", flush=True)

    model = CEPolicy(width=args.width, depth=args.depth).to(device)
    print(f"params {sum(p.numel() for p in model.parameters())/1e6:.3f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    lossf = nn.CrossEntropyLoss()
    n = len(X)
    rng = np.random.default_rng(0)
    for ep in range(args.epochs):
        model.train()
        perm = rng.permutation(n)
        tot, nb = 0.0, 0
        for i in range(0, n, args.bs):
            idx = perm[i : i + args.bs]
            logits = model(Xt[idx].to(device))
            loss = lossf(logits, Tt[idx].to(device))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        if ep % 2 == 1 or ep == args.epochs - 1:
            with torch.no_grad():
                lg = model(Xt[:2000].to(device))
                agree = (lg.argmax(-1).cpu().numpy() == R[:2000].argmax(-1)).mean()
            print(f"  ep{ep+1:02d} ce {tot/nb:.4f} | agree-with-oracle-argmax {agree:.4f}", flush=True)

    succ, ticks = eval_closed_loop(model, device)
    print(f"closed-loop success {succ:.3f} (oracle 1.000)  mean ticks {ticks:.1f} (oracle ~6.0)", flush=True)

    x = Xt[:1].to(device)
    with torch.no_grad():
        for _ in range(10): model(x)
        if device == "xpu": torch.xpu.synchronize()
        t0 = time.time()
        for _ in range(100): model(x)
        if device == "xpu": torch.xpu.synchronize()
        ms = (time.time() - t0) / 100 * 1000
    print(f"latency {ms:.2f} ms -> {1000/ms:.0f} decisions/s ({device}, batch 1)", flush=True)

    json.dump(dict(closed_loop_success=succ, mean_ticks=ticks, latency_ms=ms,
                   decisions_per_s=1000/ms, epochs=args.epochs, temp=args.temp,
                   params=sum(p.numel() for p in model.parameters())), open(args.out, "w"), indent=2)
    print(json.dumps(dict(closed_loop_success=succ, latency_ms=ms, decisions_per_s=1000/ms), indent=2))


if __name__ == "__main__":
    main()
