"""Hybrid architecture: slow perception + fast planner.

The two architectures measured so far have opposite profiles:

  policy network   0.710 success   2.09 ms/tick   (478 Hz)
  detect + BFS     0.990 success   9.85 ms/tick   (101 Hz)

101 Hz is enough for a 15-60 Hz game loop but not for a 120-240 Hz one.
The hybrid: run the DETECTOR every N ticks, keep the decoded map, and let
the DETERMINISTIC PLANNER re-plan at full speed on the cached map while the
agent's position is tracked exactly (the agent's position is known to the
simulator; on a real game it would come from the detector, or from a cheap
motion tracker).

The point of the measurement is the cost/benefit curve:
  success(perception_rate) vs wall-clock cost

If success stays ~0.99 when the detector runs at 1/8 the tick rate, the
perception cost is amortised and the architecture is real-time at any rate.
"""
from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import torch

from envs import IMG, MAZE, STEPS, bfs_dist, make_maze, render_maze, step_maze
from perception_solver import Detector, bfs_plan


def eval_hybrid(det, device, n_maze=200, seed=999, budget=3 * MAZE, every=1):
    """Perception runs every `every` ticks; the planner uses the cached map."""
    rng = random.Random(seed)
    mazes = [make_maze(rng) for _ in range(n_maze)]
    succ = 0
    ticks = []
    det_calls = 0
    with torch.no_grad():
        for base in mazes:
            st = base
            wall_map = st.walls.copy()
            t = 0
            while t < budget:
                if t % every == 0:
                    frame = render_maze(st.agent, st.goal, st.walls)
                    fr = np.clip(frame * 255, 0, 255).astype(np.uint8)
                    x = torch.from_numpy(fr).unsqueeze(0).to(device).float().div(255.)
                    la, lg, lw = det(x)
                    det_calls += 1
                    wall_map = (lw[0] > 0).reshape(MAZE, MAZE).cpu().numpy().astype(bool)
                    a_cell = int(la.argmax(-1).item())
                    g_cell = int(lg.argmax(-1).item())
                else:
                    a_cell = st.agent[0] * MAZE + st.agent[1]
                    g_cell = st.goal[0] * MAZE + st.goal[1]
                a = bfs_plan(wall_map, a_cell, g_cell)
                if a is None:
                    a = 0
                st, done = step_maze(st, a)
                t += 1
                if done:
                    break
            succ += int(st.agent == st.goal)
            ticks.append(t)
    return dict(success=succ / n_maze, ticks=float(np.mean(ticks)),
                detect_calls_per_maze=det_calls / n_maze)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--mazes", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ckpt", default="perception_detector.pt")
    ap.add_argument("--out", default="hybrid_result.json")
    args = ap.parse_args()
    device = args.device

    from perception_solver import build_dataset

    X, A, G, W = build_dataset(args.mazes, seed=0)
    Xte, Ate, Gte, Wte = build_dataset(600, seed=1)
    Xt = torch.from_numpy(X).float().div(255.)
    Xte_t = torch.from_numpy(Xte).float().div(255.)

    det = Detector().to(device)
    opt = torch.optim.AdamW([
        {"params": list(det.pos_trunk.parameters()) + list(det.agent_head.parameters()) + list(det.goal_head.parameters()), "lr": args.lr},
        {"params": list(det.wall_trunk.parameters()) + list(det.wall_head.parameters()), "lr": args.lr},
    ], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    lf_a = torch.nn.CrossEntropyLoss()
    lf_w = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.0))
    At = torch.from_numpy(A); Gt = torch.from_numpy(G)
    Wt = torch.from_numpy(W).float().reshape(len(W), -1)
    n = len(X); rng = np.random.default_rng(0)
    for ep in range(args.epochs):
        det.train(); perm = rng.permutation(n)
        tot, nb = 0.0, 0
        for i in range(0, n, 64):
            idx = perm[i:i+64]
            la, lg, lw = det(Xt[idx].to(device))
            loss = lf_a(la, At[idx].to(device)) + lf_a(lg, Gt[idx].to(device)) + lf_w(lw, Wt[idx].to(device))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        if ep % 5 == 4:
            with torch.no_grad():
                la, lg, lw = det(Xte_t.to(device))
                aa = (la.argmax(-1).cpu().numpy() == Ate).mean()
                ww = ((lw > 0).cpu().numpy() == Wte.reshape(len(Wte), -1)).mean()
            print(f"  ep{ep+1:02d} loss {tot/nb:.4f} agent {aa:.4f} wall {ww:.4f}", flush=True)
    det.eval()
    torch.save(det.state_dict(), args.ckpt)
    print(f"saved {args.ckpt}", flush=True)

    # detector latency
    x = Xte_t[:1].to(device)
    with torch.no_grad():
        for _ in range(10): det(x)
        if device == "xpu": torch.xpu.synchronize()
        t0 = time.perf_counter()
        for _ in range(100): det(x)
        if device == "xpu": torch.xpu.synchronize()
        det_ms = (time.perf_counter() - t0) / 100 * 1000
    print(f"detector latency {det_ms:.2f} ms", flush=True)

    rows = []
    for every in (1, 2, 4, 8, 16):
        r = eval_hybrid(det, device, n_maze=150, every=every)
        # amortised cost: detection happens every `every` ticks
        amortised = det_ms / every
        r.update(every=every, amortised_ms_per_tick=amortised,
                 effective_hz=1000.0 / max(amortised, 1e-6))
        rows.append(r)
        print(f"  every={every:2d}  success {r['success']:.3f}  ticks {r['ticks']:.1f}  "
              f"amortised {amortised:.2f} ms/tick -> {r['effective_hz']:.0f} Hz", flush=True)

    json.dump(dict(detector_latency_ms=det_ms, rows=rows), open(args.out, "w"), indent=2)
    print(json.dumps(dict(detector_latency_ms=det_ms, rows=rows), indent=2))


if __name__ == "__main__":
    main()
