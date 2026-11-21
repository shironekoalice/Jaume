"""Decisive split: local goal-direction vs global shortest path.

State of the evidence so far:
  * perception is NOT the bottleneck — a linear probe on the spatial trunk
    localises agent 0.998 / goal 0.996 (pooled-CNN control: chance)
  * control still fails — closed-loop success 0.11 vs oracle 1.00

So the remaining failure is in the *policy*, and the policy has two very
different requirements mixed together:

  (A) LOCAL   : "which of the 4 moves reduces the distance to the goal?"
                needs only the agent's and the goal's positions.
  (B) GLOBAL  : "which move starts a GLOBAL shortest path?"
                needs to reason about the whole wall structure.

A CNN with a few conv layers can plausibly do (A) and struggle with (B).

This script trains both heads on the same trunk and reports each separately,
plus a third control (C): the *reachable* distance, which requires seeing
walls but not planning.
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

WALL_PENALTY = 32.0


def targets(walls, agent, goal):
    """(local_dir, global_path, reachable_dist) target vectors, 4 actions each."""
    loc, glo, rea = [], [], []
    for dr, dc in STEPS:
        nr, nc = agent[0] + dr, agent[1] + dc
        blocked = not (0 <= nr < MAZE and 0 <= nc < MAZE) or walls[nr, nc]
        if blocked:
            loc.append(-WALL_PENALTY); glo.append(-WALL_PENALTY); rea.append(-WALL_PENALTY)
            continue
        d = bfs_dist(walls, (nr, nc), goal)
        manhattan = abs(nr - goal[0]) + abs(nc - goal[1])
        loc.append(-manhattan)
        glo.append(-(d if d is not None else WALL_PENALTY))
        rea.append(-(d if d is not None else WALL_PENALTY))
    return (np.array(loc, np.float32), np.array(glo, np.float32), np.array(rea, np.float32))


def collect(n_maze, frames_per_maze, seed):
    rng = random.Random(seed)
    X, L, G, R = [], [], [], []
    for _ in range(n_maze):
        st = make_maze(rng)
        for _ in range(frames_per_maze):
            l, g, r = targets(st.walls, st.agent, st.goal)
            X.append(np.clip(st.frame * 255, 0, 255).astype(np.uint8))
            L.append(l); G.append(g); R.append(r)
            st, done = step_maze(st, int(np.argmax(g)))
            if done:
                break
    return np.stack(X), np.stack(L), np.stack(G), np.stack(R)


class MultiHeadActor(nn.Module):
    def __init__(self, width=32, depth=3):
        super().__init__()
        self.trunk = SpatialCNN(3, width, depth, in_size=IMG, out_dim=256)
        self.local = nn.Linear(256, 4)
        self.glob = nn.Linear(256, 4)

    def forward(self, x):
        f = self.trunk(x)
        return self.local(f), self.glob(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--mazes", type=int, default=400)
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--out", default="maze_split_result.json")
    args = ap.parse_args()
    device = args.device

    X, L, G, R = collect(args.mazes, args.frames, seed=0)
    Xt = torch.from_numpy(X).float().div(255.)
    Lt = torch.from_numpy(L); Gt = torch.from_numpy(G)
    print(f"frames {len(X)}", flush=True)

    model = MultiHeadActor().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    lossf = nn.MSELoss()
    n = len(X)
    rng = np.random.default_rng(0)
    for ep in range(args.epochs):
        model.train()
        perm = rng.permutation(n)
        tot, nb = 0.0, 0
        for i in range(0, n, 128):
            idx = perm[i : i + 128]
            xl, xg = model(Xt[idx].to(device))
            loss = 0.5 * lossf(xl, Lt[idx].to(device)) + 0.5 * lossf(xg, Gt[idx].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        if ep % 3 == 2 or ep == args.epochs - 1:
            with torch.no_grad():
                xl, xg = model(Xt[:2000].to(device))
                al = (xl.argmax(-1).cpu().numpy() == Lt[:2000].numpy().argmax(-1)).mean()
                ag = (xg.argmax(-1).cpu().numpy() == Gt[:2000].numpy().argmax(-1)).mean()
            print(f"  ep{ep+1:02d} mse {tot/nb:.2f} | LOCAL agree {al:.4f} | GLOBAL agree {ag:.4f}", flush=True)

    # closed-loop using the LOCAL head and the GLOBAL head, on identical mazes
    rng = random.Random(999)
    mazes = [make_maze(rng) for _ in range(150)]
    res = {"local": 0, "global": 0, "oracle": 0}
    tk = {"local": [], "global": [], "oracle": []}
    with torch.no_grad():
        for base in mazes:
            for mode in ("local", "global"):
                st = base; t = 0
                while t < 3 * MAZE:
                    x = torch.from_numpy(np.clip(st.frame * 255, 0, 255).astype(np.uint8)).unsqueeze(0).to(device).float().div(255.)
                    xl, xg = model(x)
                    a = int((xl if mode == "local" else xg).argmax(-1).item())
                    st, done = step_maze(st, a); t += 1
                    if done: break
                res[mode] += int(done); tk[mode].append(t)
            st = base; t = 0
            while t < 3 * MAZE:
                a = int(np.argmax(targets(st.walls, st.agent, st.goal)[1]))
                st, done = step_maze(st, a); t += 1
                if done: break
            res["oracle"] += int(done); tk["oracle"].append(t)
    out = {k: res[k] / len(mazes) for k in res}
    print("closed-loop success:", {k: round(v, 3) for k, v in out.items()}, flush=True)
    print("mean ticks:", {k: round(float(np.mean(v)), 1) for k, v in tk.items()}, flush=True)

    x = Xt[:1].to(device)
    with torch.no_grad():
        for _ in range(10): model(x)
        if device == "xpu": torch.xpu.synchronize()
        t0 = time.time()
        for _ in range(100): model(x)
        if device == "xpu": torch.xpu.synchronize()
        ms = (time.time() - t0) / 100 * 1000
    print(f"latency {ms:.2f} ms -> {1000/ms:.0f} decisions/s", flush=True)

    json.dump(dict(closed_loop=out, ticks={k: float(np.mean(v)) for k, v in tk.items()},
                   latency_ms=ms, decisions_per_s=1000/ms), open(args.out, "w"), indent=2)
    print(json.dumps(dict(closed_loop=out, latency_ms=ms), indent=2))


if __name__ == "__main__":
    main()
