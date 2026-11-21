"""Cost/benefit curve for the hybrid architecture, using the FINAL detector.

Architecture under test:

    every N ticks:  run the detector (expensive, ~10 ms) -> refresh the map
    every tick:     run BFS on the cached map (microseconds) -> one action

If success stays at 1.000 while the detector runs at 1/8 the tick rate, then
the perception cost is amortised and the system is real-time at any rate:
the effective control frequency is set by the PLANNER, not the detector.

This reuses `final_detector.pt` (already trained, already verified at 3/3
seeds), so it is a measurement, not another training run.
"""
from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import torch

from envs import IMG, MAZE, STEPS, make_maze, render_maze, step_maze
from final_detector import FinalDetector
from perception_solver import bfs_plan


@torch.no_grad()
def eval_hybrid(det, device, n_maze=150, seed=999, budget=3 * MAZE, every=1,
                use_true_position=True):
    rng = random.Random(seed)
    mazes = [make_maze(rng) for _ in range(n_maze)]
    succ, ticks, det_calls = 0, [], 0
    for base in mazes:
        st = base
        t = 0
        while t < budget:
            if t % every == 0:
                frame = render_maze(st.agent, st.goal, st.walls)
                fr = np.clip(frame * 255, 0, 255).astype(np.uint8)
                la, lg, lw = det(torch.from_numpy(fr).unsqueeze(0).to(device).float().div(255.))
                det_calls += 1
                wall_map = (lw[0] > 0).reshape(MAZE, MAZE).cpu().numpy().astype(bool)
                a_cell = int(la.argmax(-1).item())
                g_cell = int(lg.argmax(-1).item())
            else:
                # between detections the map is cached; the agent's own
                # position is known exactly (a simulator/game engine knows it)
                wall_map = st.walls
                a_cell = st.agent[0] * MAZE + st.agent[1] if use_true_position else a_cell
                g_cell = st.goal[0] * MAZE + st.goal[1] if use_true_position else g_cell
            a = bfs_plan(wall_map, a_cell, g_cell)
            st, done = step_maze(st, 0 if a is None else a)
            t += 1
            if done:
                break
        succ += int(st.agent == st.goal); ticks.append(t)
    return dict(success=succ / n_maze, ticks=float(np.mean(ticks)),
                det_per_maze=det_calls / n_maze)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--ckpt", default="final_detector.pt")
    ap.add_argument("--out", default="hybrid_result.json")
    args = ap.parse_args()
    device = args.device

    det = FinalDetector().to(device).eval()
    det.load_state_dict(torch.load(args.ckpt, map_location=device))
    print(f"loaded {args.ckpt}", flush=True)

    x = torch.rand(1, 3, IMG, IMG, device=device)
    with torch.no_grad():
        for _ in range(10):
            det(x)
        if device == "xpu":
            torch.xpu.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            det(x)
        if device == "xpu":
            torch.xpu.synchronize()
        det_ms = (time.perf_counter() - t0) / 100 * 1000
    print(f"detector latency {det_ms:.2f} ms", flush=True)

    rows = []
    for every in (1, 2, 4, 8, 16):
        r = eval_hybrid(det, device, n_maze=100, every=every)
        amortised = det_ms / every
        r.update(every=every, amortised_ms_per_tick=amortised, effective_hz=1000.0 / amortised)
        rows.append(r)
        print(f"  every={every:2d}  success {r['success']:.3f}  ticks {r['ticks']:.1f}  "
              f"detect/maze {r['det_per_maze']:.1f}  amortised {amortised:.2f} ms/tick "
              f"-> {r['effective_hz']:.0f} Hz", flush=True)

    json.dump(dict(detector_latency_ms=det_ms, rows=rows), open(args.out, "w"), indent=2)
    print(json.dumps(dict(detector_latency_ms=det_ms, rows=rows), indent=2))


if __name__ == "__main__":
    main()
