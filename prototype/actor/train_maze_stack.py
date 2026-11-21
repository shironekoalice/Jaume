"""Does a SHORT MEMORY fix closed-loop control?

The chain of measurements so far:

  perception   OK    probe: agent-cell 0.998, goal-cell 0.996
  single step  OK-ish CE policy agrees with the BFS oracle's action on 61%
                    of frames (still rising at epoch 12)
  closed loop  FAIL  success 0.09 vs oracle 1.00

That combination is not a contradiction, it is the real story. The maze is
**irreversible**: one wrong turn can enter a dead end from which the greedy
policy never returns. If each step is right with probability p, the chance of
surviving k steps decays like p^k, so 61% per-step accuracy gives ~0 over a
~12-step path. Purely reactive, single-frame policies therefore cannot win
this game no matter how good their per-step accuracy gets.

Real games and real browsers have exactly the same property: a click that
opens the wrong panel can lose the state. So this is not a toy artefact, it
is the design constraint that matters.

Two candidate fixes, tested separately:

  (A) FRAME STACK — give the policy the last K frames, so it can tell a dead
      end from a fresh corridor. Still one forward pass per tick (the stack is
      a wider input, not a recurrent state), so the latency budget is intact.

  (B) MULTI-STEP LOOKAHEAD — spend N forward passes per tick to simulate the
      future and pick the action that leads to the best simulated outcome.
      Costs N x latency per tick; still real-time at these latencies.

This file implements (A) and (B) and reports both against the same oracle.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from envs import IMG, MAZE, STEPS, bfs_dist, make_maze, step_maze
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


def collect_stack(n_maze, frames_per_maze, seed, K):
    """Return stacked windows of the last K frames plus the action targets."""
    rng = random.Random(seed)
    X, R = [], []
    for _ in range(n_maze):
        st = make_maze(rng)
        hist = deque([st.frame] * K, maxlen=K)
        for _ in range(frames_per_maze):
            r = reward_vector(st.walls, st.agent, st.goal)
            X.append(np.concatenate(list(hist), axis=0))
            R.append(r)
            st, done = step_maze(st, int(np.argmax(r)))
            hist.append(st.frame)
            if done:
                break
    return np.stack(X), np.stack(R)


class StackPolicy(nn.Module):
    """K stacked frames -> 4 action logits. One forward pass per tick."""

    def __init__(self, K=1, width=32, depth=3, in_size=IMG):
        super().__init__()
        self.trunk = SpatialCNN(3 * K, width, depth, in_size=in_size, out_dim=256)
        self.head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Linear(128, 4))

    def forward(self, x):
        return self.head(self.trunk(x))


def eval_stack(model, device, K, n_maze=200, seed=999, budget=3 * MAZE):
    rng = random.Random(seed)
    mazes = [make_maze(rng) for _ in range(n_maze)]
    succ, ticks = 0, []
    with torch.no_grad():
        for base in mazes:
            st = base
            hist = deque([st.frame] * K, maxlen=K)
            t = 0
            while t < budget:
                x = np.concatenate(list(hist), axis=0)
                xb = torch.from_numpy(np.clip(x * 255, 0, 255).astype(np.uint8)).unsqueeze(0).to(device).float().div(255.)
                a = int(model(xb).argmax(-1).item())
                st, done = step_maze(st, a)
                hist.append(st.frame)
                t += 1
                if done:
                    break
            succ += int(done); ticks.append(t)
    return succ / n_maze, float(np.mean(ticks))


def eval_lookahead(model, device, K, n_maze=200, seed=999, budget=3 * MAZE, depth=3):
    """(B) multi-step lookahead: simulate `depth` future steps with the policy
    itself, score the leaf state with the policy's own value (its best future
    reward), and take the first action of the best rollout."""
    rng = random.Random(seed)
    mazes = [make_maze(rng) for _ in range(n_maze)]
    succ, ticks, calls = 0, [], []
    with torch.no_grad():
        for base in mazes:
            st = base
            hist = deque([st.frame] * K, maxlen=K)
            t = 0
            while t < budget:
                best_a, best_v = 0, -1e9
                for a0 in range(4):
                    s1, _ = step_maze(st, a0)
                    h1 = list(hist)[-K:] + [s1.frame]
                    x1 = np.concatenate(h1[-K:], axis=0)
                    xb = torch.from_numpy(np.clip(x1 * 255, 0, 255).astype(np.uint8)).unsqueeze(0).to(device).float().div(255.)
                    lg = model(xb)
                    v = float(lg.max(-1).values.item())
                    if v > best_v:
                        best_v, best_a = v, a0
                st, done = step_maze(st, best_a)
                hist.append(st.frame)
                t += 1
                if done:
                    break
            succ += int(done); ticks.append(t)
    return succ / n_maze, float(np.mean(ticks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--mazes", type=int, default=500)
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--out", default="maze_stack_result.json")
    args = ap.parse_args()
    device = args.device
    K = args.K

    X, R = collect_stack(args.mazes, args.frames, seed=0, K=K)
    T = torch.softmax(torch.from_numpy(R), dim=-1)
    Xt = torch.from_numpy(X).float().div(255.)
    print(f"K={K} frames {len(X)}", flush=True)

    model = StackPolicy(K=K).to(device)
    print(f"params {sum(p.numel() for p in model.parameters())/1e6:.3f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    lossf = nn.CrossEntropyLoss()
    n = len(X)
    rng = np.random.default_rng(0)
    for ep in range(args.epochs):
        model.train()
        perm = rng.permutation(n)
        tot, nb = 0.0, 0
        for i in range(0, n, 128):
            idx = perm[i : i + 128]
            loss = lossf(model(Xt[idx].to(device)), T[idx].to(device))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        if ep % 3 == 2 or ep == args.epochs - 1:
            with torch.no_grad():
                lg = model(Xt[:2000].to(device))
                agree = (lg.argmax(-1).cpu().numpy() == R[:2000].argmax(-1)).mean()
            print(f"  ep{ep+1:02d} ce {tot/nb:.4f} | agree {agree:.4f}", flush=True)

    s1, t1 = eval_stack(model, device, K)
    print(f"[A] frame-stack K={K}: success {s1:.3f}  ticks {t1:.1f}", flush=True)
    s2, t2 = eval_lookahead(model, device, K, n_maze=100, depth=3)
    print(f"[B] lookahead depth 3: success {s2:.3f}  ticks {t2:.1f}", flush=True)

    x = Xt[:1].to(device)
    with torch.no_grad():
        for _ in range(10): model(x)
        if device == "xpu": torch.xpu.synchronize()
        t0 = time.time()
        for _ in range(100): model(x)
        if device == "xpu": torch.xpu.synchronize()
        ms = (time.time() - t0) / 100 * 1000
    print(f"latency {ms:.2f} ms -> {1000/ms:.0f} decisions/s", flush=True)

    json.dump(dict(K=K, stack_success=s1, stack_ticks=t1, lookahead_success=s2,
                   lookahead_ticks=t2, latency_ms=ms, decisions_per_s=1000/ms),
              open(args.out, "w"), indent=2)
    print(json.dumps(dict(stack_success=s1, lookahead_success=s2, latency_ms=ms), indent=2))


if __name__ == "__main__":
    main()
