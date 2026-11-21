"""Generalisation check: does the maze policy hold up on unseen maze families?

0.710 closed-loop success was measured on mazes from the SAME generator
(random walls at p=0.22, random endpoints). A policy can score well by
memorising that generator's statistics — the agent is always at one of the
few free cells, the goal is never on the border, walls never form a closed
pocket, and so on.

This file re-evaluates on **shifted distributions** the policy never saw:

  D1  denser walls (p=0.32)      — tighter corridors
  D2  sparser walls (p=0.12)      — open rooms
  D3  larger grid (12x12)         — longer horizons, more states
  D4  walls as thick blocks (2x2)  — different visual statistics
  D5  shifted colours             — same geometry, different palette

The honest reading is: if success stays high on D1/D2/D5 the policy learned
maze navigation; if it collapses on D3 it learned a fixed-horizon reaction,
not a plan.
"""
from __future__ import annotations

import argparse
import json
import random

import numpy as np
import torch
import torch.nn as nn

from envs import IMG, MAZE, STEPS, bfs_dist, make_maze, render_maze, step_maze
from model import SpatialCNN

MAXD = MAZE * MAZE
BLOCK = -(MAXD + 2)


def reward_vector(walls, agent, goal):
    r = []
    for dr, dc in STEPS:
        nr, nc = agent[0] + dr, agent[1] + dc
        if not (0 <= nr < MAZE and 0 <= nc < MAZE) or walls[nr, nc]:
            r.append(BLOCK); continue
        d = bfs_dist(walls, (nr, nc), goal)
        r.append(-(d if d is not None else MAXD))
    return np.array(r, np.float32)


def make_maze_custom(rng, size, p_wall, thick=False, palette=None):
    while True:
        walls = np.zeros((size, size), dtype=bool)
        if thick:
            for r in range(1, size - 1):
                for c in range(1, size - 1):
                    if rng.random() < p_wall:
                        for dr in range(2):
                            for dc in range(2):
                                if r + dr < size and c + dc < size:
                                    walls[r + dr, c + dc] = True
        else:
            for r in range(1, size - 1):
                for c in range(1, size - 1):
                    if rng.random() < p_wall:
                        walls[r, c] = True
        free = [(r, c) for r in range(size) for c in range(size) if not walls[r, c]]
        if len(free) < 10:
            continue
        start, goal = rng.sample(free, 2)
        walls[start] = False; walls[goal] = False
        if bfs_dist(walls, start, goal) is not None:
            frame = render_maze(start, goal, walls, size=size, palette=palette)
            return dict(agent=start, goal=goal, walls=walls, frame=frame)


class Policy(nn.Module):
    def __init__(self, width=48, depth=4, in_size=IMG):
        super().__init__()
        self.trunk = SpatialCNN(3, width, depth, in_size=in_size, out_dim=256)
        self.head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Linear(128, 4))

    def forward(self, x):
        return self.head(self.trunk(x))


def as_dict(m):
    """Normalise a maze to a dict so MazeState and the custom generator
    produce the same shape downstream."""
    if isinstance(m, dict):
        return m
    return dict(agent=m.agent, goal=m.goal, walls=m.walls, frame=m.frame)


def eval_domain(model, device, mazes, budget, size=MAZE):
    """Closed-loop evaluation.

    BUG FIXED here: the first version fed the model `m["frame"]` — the
    INITIAL frame — on every tick, so the model never saw the agent move.
    That produced 0.175 instead of the 0.885 the same policy scores under
    the correct evaluator, and it made every generalisation domain look
    broken for the wrong reason. The frame must be re-rendered after every
    move.
    """
    succ, ticks = 0, []
    with torch.no_grad():
        for m0 in mazes:
            m = as_dict(m0)
            agent, goal, walls = m["agent"], m["goal"], m["walls"]
            t = 0
            while t < budget:
                frame = render_maze(agent, goal, walls, size=size)
                fr = np.clip(frame * 255, 0, 255).astype(np.uint8)
                x = torch.from_numpy(fr).unsqueeze(0).to(device).float().div(255.)
                a = int(model(x).argmax(-1).item())
                dr, dc = STEPS[a]
                nr, nc = agent[0] + dr, agent[1] + dc
                if not (0 <= nr < size and 0 <= nc < size) or walls[nr, nc]:
                    nr, nc = agent
                agent = (nr, nc)
                t += 1
                if agent == goal:
                    break
            succ += int(agent == goal); ticks.append(t)
    return succ / len(mazes), float(np.mean(ticks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--mazes", type=int, default=1500)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--out", default="maze_generalisation.json")
    args = ap.parse_args()
    device = args.device

    # ---- train on the ORIGINAL distribution only
    rng = random.Random(0)
    X, R = [], []
    for _ in range(args.mazes):
        st = make_maze(rng)
        for _ in range(args.frames):
            r = reward_vector(st.walls, st.agent, st.goal)
            X.append(np.clip(st.frame * 255, 0, 255).astype(np.uint8)); R.append(r)
            st, done = step_maze(st, int(np.argmax(r)))
            if done: break
    X = np.stack(X); R = np.stack(R)
    T = torch.softmax(torch.from_numpy(R), dim=-1)
    Xt = torch.from_numpy(X).float().div(255.)
    model = Policy(width=48, depth=4).to(device)
    print(f"train frames {len(X)}  params {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    lossf = nn.CrossEntropyLoss()
    n = len(X); rngg = np.random.default_rng(0)
    for ep in range(args.epochs):
        model.train(); perm = rngg.permutation(n)
        tot, nb = 0.0, 0
        for i in range(0, n, 128):
            idx = perm[i:i+128]
            loss = lossf(model(Xt[idx].to(device)), T[idx].to(device))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        if ep % 5 == 4:
            with torch.no_grad():
                agree = (model(Xt[:2000].to(device)).argmax(-1).cpu().numpy() == R[:2000].argmax(-1)).mean()
            print(f"  ep{ep+1:02d} ce {tot/nb:.4f} agree {agree:.4f}", flush=True)
    model.eval()
    torch.save(model.state_dict(), "maze_gen_policy.pt")
    print("saved maze_gen_policy.pt", flush=True)

    # ---- evaluate on shifted domains
    domains = {}
    rng = random.Random(999)
    domains["D0_train_dist"] = ([make_maze(rng) for _ in range(200)], MAZE, 3 * MAZE)
    domains["D1_dense_p32"] = ([make_maze_custom(rng, MAZE, 0.32) for _ in range(200)], MAZE, 3 * MAZE)
    domains["D2_sparse_p12"] = ([make_maze_custom(rng, MAZE, 0.12) for _ in range(200)], MAZE, 3 * MAZE)
    domains["D3_grid12"] = ([make_maze_custom(rng, 12, 0.22) for _ in range(200)], 12, 4 * 12)
    domains["D4_thick_walls"] = ([make_maze_custom(rng, MAZE, 0.10, thick=True) for _ in range(200)], MAZE, 3 * MAZE)
    domains["D5_shift_palette"] = ([make_maze_custom(rng, MAZE, 0.22, palette="shift") for _ in range(200)], MAZE, 3 * MAZE)

    res = {}
    for name, (mazes, size, budget) in domains.items():
        s, tk = eval_domain(model, device, mazes, budget, size)
        res[name] = dict(success=s, ticks=tk)
        print(f"  {name:18s} success {s:.3f}  ticks {tk:.1f}", flush=True)

    json.dump(res, open(args.out, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
