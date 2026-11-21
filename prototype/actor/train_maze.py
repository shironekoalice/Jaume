"""Closed-loop game training: behaviour cloning on a maze, then measure
closed-loop success and decision rate.

Why behaviour cloning and not RL: the repo's planning report showed that a
350M policy learning to *choose* among actions hit a representation wall
(top-1 0.475 vs greedy 1.000), and that greedy heuristics were already strong
on synthetic planning. BC from an *exact* BFS oracle avoids the whole
credit-assignment problem and isolates the question we actually care about:

    can a small vision model, one forward pass per tick, control a game
    from pixels at a usable rate?

Metrics reported:
  * closed-loop success = reaches goal within a tick budget
  * ticks-to-goal (model vs BFS oracle vs random)
  * decisions per second on the real device
"""
from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn

from envs import IMG, MAZE, STEPS, best_action, make_maze, step_maze
from model import MazeActor

TICK_BUDGET = 3 * MAZE  # generous but finite


def collect(n_maze: int, frames_per_maze: int, seed: int):
    """Roll out the BFS oracle and record (frame, action) pairs."""
    rng = random.Random(seed)
    X = np.zeros((n_maze * frames_per_maze, 3, IMG, IMG), dtype=np.uint8)
    Y = np.zeros((n_maze * frames_per_maze,), dtype=np.int64)
    i = 0
    for _ in range(n_maze):
        st = make_maze(rng)
        for _ in range(frames_per_maze):
            a = best_action(st.walls, st.agent, st.goal, rng=rng)
            X[i] = np.clip(st.frame * 255, 0, 255).astype(np.uint8)
            Y[i] = a
            i += 1
            st, done = step_maze(st, a)
            if done:
                break
    return X[:i], Y[:i]


@torch.no_grad()
def rollout(model, device, n_maze=100, seed=999, budget=TICK_BUDGET):
    rng = random.Random(seed)
    model.eval()
    succ, ticks, orc_ticks, rnd_ticks = 0, [], [], []
    for _ in range(n_maze):
        st = make_maze(rng)
        t = 0
        while t < budget:
            x = torch.from_numpy(np.clip(st.frame * 255, 0, 255).astype(np.uint8)).unsqueeze(0).to(device).float() / 255.0
            a = int(model(x).argmax(-1).item())
            st, done = step_maze(st, a)
            t += 1
            if done:
                break
        succ += int(done)
        ticks.append(t)
        # baselines on the same maze
        st2 = make_maze(random.Random(seed))  # placeholder, replaced below
    return succ / n_maze, float(np.mean(ticks))


def rollout_compare(model, device, n_maze=100, seed=999, budget=TICK_BUDGET):
    """Same mazes for model / oracle / random, so the numbers are comparable."""
    rng = random.Random(seed)
    mazes = [make_maze(rng) for _ in range(n_maze)]
    model.eval()
    res = {"model": 0, "oracle": 0, "random": 0}
    mticks, oticks = [], []
    with torch.no_grad():
        for base in mazes:
            st = base
            t = 0
            while t < budget:
                x = torch.from_numpy(np.clip(st.frame * 255, 0, 255).astype(np.uint8)).unsqueeze(0).to(device).float() / 255.0
                a = int(model(x).argmax(-1).item())
                st, done = step_maze(st, a)
                t += 1
                if done:
                    break
            res["model"] += int(done)
            mticks.append(t)

            st = base
            t = 0
            while t < budget:
                a = best_action(st.walls, st.agent, st.goal)
                st, done = step_maze(st, a)
                t += 1
                if done:
                    break
            res["oracle"] += int(done)
            oticks.append(t)

            st = base
            t = 0
            r2 = random.Random(1234 + t)
            while t < budget:
                a = r2.randrange(4)
                st, done = step_maze(st, a)
                t += 1
                if done:
                    break
            res["random"] += int(done)
    return {k: v / n_maze for k, v in res.items()}, float(np.mean(mticks)), float(np.mean(oticks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mazes", type=int, default=400)
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--out", default="maze_result.json")
    args = ap.parse_args()

    device = args.device
    X, Y = collect(args.mazes, args.frames, seed=0)
    print(f"oracle frames: {len(X)}  action histogram {np.bincount(Y, minlength=4)}", flush=True)
    Xt = torch.from_numpy(X).float() / 255.0
    Yt = torch.from_numpy(Y)

    model = MazeActor().to(device)
    npar = sum(p.numel() for p in model.parameters())
    print(f"model params {npar/1e6:.3f}M", flush=True)
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
            xb = Xt[idx].to(device)
            yb = Yt[idx].to(device)
            loss = lossf(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        if ep % 3 == 2 or ep == args.epochs - 1:
            acc = (model(Xt[:2000].to(device)).argmax(-1).cpu().numpy() == Yt[:2000].numpy()).mean()
            print(f"  ep{ep+1:02d} loss {tot/nb:.4f} frame-acc {acc:.4f}", flush=True)

    succ, mticks, oticks = rollout_compare(model, device)
    print(f"closed-loop success: model {succ['model']:.3f} | oracle {succ['oracle']:.3f} | random {succ['random']:.3f}", flush=True)
    print(f"mean ticks: model {mticks:.1f} | oracle {oticks:.1f}", flush=True)

    # decision rate
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

    res = dict(frame_acc=float((model(Xt[:2000].to(device)).argmax(-1).cpu().numpy() == Yt[:2000].numpy()).mean()),
               closed_loop=succ, mean_ticks_model=mticks, mean_ticks_oracle=oticks,
               latency_ms=ms, decisions_per_s=1000 / ms, params=npar, epochs=args.epochs)
    json.dump(res, open(args.out, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
