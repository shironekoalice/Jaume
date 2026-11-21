"""Game-side actor as a VALUE model, not behaviour cloning.

BC on "a random shortest-path first move" failed (closed-loop success 0.09,
oracle 1.00). Diagnosis: the label carries almost no information — from a
given state only one or two moves are shortest-path-valid, and the sampled
one is ambiguous, so the model learns a marginal prior over directions
(right/down dominate the data) rather than a decision function.

The fix keeps the repo's core thesis (score the options, take an argmax, no
autoregressive decoding) but makes the supervision dense and unambiguous:

    for every state s and every action a:
        target(s, a) = -BFS_distance( next(s, a), goal )     (blocked -> large penalty)

Four informative labels per frame instead of one ambiguous label, and the
inference rule is exactly `argmax_a Q(s,a)` — the same shape as the UI actor's
candidate scoring. The oracle for this objective is computable exactly, so
the gap between model and oracle is meaningful.

Also reports an ablation: can the model recover the goal *direction* alone
(ignoring walls)? That separates "cannot see the goal" from "cannot see walls".
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

WALL_PENALTY = 32.0  # > any real distance in an 8x8 maze


def value_targets(walls, agent, goal):
    """-distance-to-goal for each action; blocked actions get -WALL_PENALTY."""
    out = []
    for dr, dc in STEPS:
        nr, nc = agent[0] + dr, agent[1] + dc
        if not (0 <= nr < MAZE and 0 <= nc < MAZE) or walls[nr, nc]:
            out.append(-WALL_PENALTY)
            continue
        d = bfs_dist(walls, (nr, nc), goal)
        out.append(-(d if d is not None else WALL_PENALTY))
    return np.array(out, dtype=np.float32)


def collect(n_maze, frames_per_maze, seed):
    rng = random.Random(seed)
    X, Y = [], []
    for _ in range(n_maze):
        st = make_maze(rng)
        for _ in range(frames_per_maze):
            X.append(np.clip(st.frame * 255, 0, 255).astype(np.uint8))
            Y.append(value_targets(st.walls, st.agent, st.goal))
            st, done = step_maze(st, int(np.argmax(Y[-1])))
            if done:
                break
    return np.stack(X), np.stack(Y)


class ValueActor(nn.Module):
    """Frame -> 4 action values. One forward, one argmax, zero decoding."""

    def __init__(self, width=32, depth=3):
        super().__init__()
        self.trunk = SpatialCNN(in_ch=3, width=width, depth=depth, in_size=IMG, out_dim=256)
        self.head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Linear(128, 4))

    def forward(self, x):
        return self.head(self.trunk(x))


def decide(model, frame, device):
    x = torch.from_numpy(frame).unsqueeze(0).to(device).float() / 255.0
    return int(model(x).argmax(-1).item())


def evaluate_closed_loop(model, device, n_maze=200, seed=999, budget=3 * MAZE):
    rng = random.Random(seed)
    mazes = [make_maze(rng) for _ in range(n_maze)]
    out = {"model": 0, "oracle": 0}
    mticks, oticks = [], []
    with torch.no_grad():
        for base in mazes:
            st = base
            t = 0
            while t < budget:
                fr = np.clip(st.frame * 255, 0, 255).astype(np.uint8)
                st, done = step_maze(st, decide(model, fr, device))
                t += 1
                if done:
                    break
            out["model"] += int(done)
            mticks.append(t)

            st = base
            t = 0
            while t < budget:
                tgt = value_targets(st.walls, st.agent, st.goal)
                st, done = step_maze(st, int(np.argmax(tgt)))
                t += 1
                if done:
                    break
            out["oracle"] += int(done)
            oticks.append(t)
    return {k: v / n_maze for k, v in out.items()}, float(np.mean(mticks)), float(np.mean(oticks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mazes", type=int, default=800)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--out", default="maze_value_result.json")
    args = ap.parse_args()

    device = args.device
    X, Y = collect(args.mazes, args.frames, seed=0)
    print(f"frames {len(X)}  value targets mean {Y.mean():.2f}", flush=True)
    Xt = torch.from_numpy(X).float() / 255.0
    Yt = torch.from_numpy(Y)

    model = ValueActor().to(device)
    print(f"params {sum(p.numel() for p in model.parameters())/1e6:.3f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    lossf = nn.MSELoss()
    n = len(X)
    rng = np.random.default_rng(0)
    for ep in range(args.epochs):
        model.train()
        perm = rng.permutation(n)
        tot, nb = 0.0, 0
        for i in range(0, n, args.bs):
            idx = perm[i : i + args.bs]
            loss = lossf(model(Xt[idx].to(device)), Yt[idx].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        if ep % 2 == 1 or ep == args.epochs - 1:
            with torch.no_grad():
                pv = model(Xt[:3000].to(device))
                agree = (pv.argmax(-1).cpu().numpy() == Yt[:3000].numpy().argmax(-1)).mean()
            print(f"  ep{ep+1:02d} mse {tot/nb:.3f}  action-agreement-with-oracle {agree:.4f}", flush=True)

    succ, mticks, oticks = evaluate_closed_loop(model, device)
    print(f"closed-loop success: model {succ['model']:.3f} | oracle {succ['oracle']:.3f}", flush=True)
    print(f"mean ticks: model {mticks:.1f} | oracle {oticks:.1f}", flush=True)

    x = Xt[:1].to(device)
    with torch.no_grad():
        for _ in range(10):
            model(x)
        if device == "xpu":
            torch.xpu.synchronize()
        t = time.time()
        for _ in range(100):
            model(x)
        if device == "xpu":
            torch.xpu.synchronize()
        ms = (time.time() - t) / 100 * 1000
    print(f"decision latency {ms:.2f} ms -> {1000/ms:.1f} decisions/s ({device}, batch 1)", flush=True)

    res = dict(closed_loop=succ, mean_ticks_model=mticks, mean_ticks_oracle=oticks,
               latency_ms=ms, decisions_per_s=1000 / ms, epochs=args.epochs,
               params=sum(p.numel() for p in model.parameters()))
    json.dump(res, open(args.out, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
