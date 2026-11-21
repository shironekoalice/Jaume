"""Error taxonomy for the game policy.

The closed-loop success of the CE policy is 0.09 while its per-step agreement
with the BFS oracle is 0.61. Before scaling the model, find out *which* 39%
of the errors are being made. Three candidate error types, each with a
different fix:

  E1 "into a wall"      — the policy picked a blocked direction.
                           Fix: perception of walls, or an explicit
                           action-mask before the softmax.
  E2 "wrong free way"   — the policy picked a legal but non-optimal direction.
                           Fix: better state representation / more training.
  E3 "oscillation"     — the policy alternates between two directions.
                           Fix: needs memory, not perception.

E3 is the dangerous one for an irreversible environment: an oscillating policy
never reaches the goal even when its per-step "knowledge" is correct.

This script replays the trained policy on held-out mazes and labels each
step with the error type, plus measures how often the policy *decreases* the
true BFS distance (the property that actually guarantees success).
"""
from __future__ import annotations

import argparse
import json
import random

import numpy as np
import torch

from envs import IMG, MAZE, STEPS, bfs_dist, make_maze, step_maze
from train_maze_ce import CEPolicy, reward_vector, collect


def analyse(model, device, n_maze=200, seed=999, budget=3 * MAZE):
    rng = random.Random(seed)
    mazes = [make_maze(rng) for _ in range(n_maze)]
    cnt = {"E1_wall": 0, "E2_wrong_free": 0, "E3_oscillation": 0, "ok": 0}
    decreased = 0
    total = 0
    with torch.no_grad():
        for base in mazes:
            st = base
            last_a = None
            for _ in range(budget):
                x = np.clip(st.frame * 255, 0, 255).astype(np.uint8)
                a = int(model(torch.from_numpy(x).unsqueeze(0).to(device).float().div(255.)).argmax(-1).item())
                r = reward_vector(st.walls, st.agent, st.goal)
                best = int(np.argmax(r))
                total += 1
                if a == best:
                    cnt["ok"] += 1
                elif r[a] <= -1e5:  # blocked direction
                    cnt["E1_wall"] += 1
                elif last_a is not None and a != best and last_a != best and a != last_a:
                    cnt["E3_oscillation"] += 1
                else:
                    cnt["E2_wrong_free"] += 1
                # did the chosen move decrease the TRUE distance to the goal?
                dr, dc = STEPS[a]
                nr, nc = st.agent[0] + dr, st.agent[1] + dc
                if 0 <= nr < MAZE and 0 <= nc < MAZE and not st.walls[nr, nc]:
                    d_new = bfs_dist(st.walls, (nr, nc), st.goal)
                    d_old = bfs_dist(st.walls, st.agent, st.goal)
                    if d_new is not None and d_old is not None and d_new < d_old:
                        decreased += 1
                st, done = step_maze(st, a)
                last_a = a
                if done:
                    break
    return cnt, decreased / max(total, 1), total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--mazes", type=int, default=300)
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--out", default="maze_error_taxonomy.json")
    args = ap.parse_args()
    device = args.device

    X, R = collect(args.mazes, args.frames, seed=0)
    T = torch.softmax(torch.from_numpy(R), dim=-1)
    Xt = torch.from_numpy(X).float().div(255.)
    model = CEPolicy().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    lossf = torch.nn.CrossEntropyLoss()
    n = len(X); rng = np.random.default_rng(0)
    for ep in range(args.epochs):
        model.train(); perm = rng.permutation(n)
        for i in range(0, n, 128):
            idx = perm[i:i+128]
            loss = lossf(model(Xt[idx].to(device)), T[idx].to(device))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    model.eval()

    cnt, dec_frac, total = analyse(model, device)
    print(f"steps analysed {total}", flush=True)
    for k, v in cnt.items():
        print(f"  {k:16s} {v/max(total,1):.4f}", flush=True)
    print(f"moves that strictly decreased true BFS distance: {dec_frac:.4f}", flush=True)
    print("(success requires this fraction to be high; an irreversible maze with a "
          "~12-step path needs ~0.95+)", flush=True)
    json.dump(dict(counts=cnt, decreased_fraction=dec_frac, steps=total),
              open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
