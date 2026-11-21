"""Perception + deterministic solver: where is the real bottleneck?

Motivation. The pixel policy reached 0.710 closed-loop success with 7.2M
parameters, and its error analysis said: it NEVER hits a wall (E1 = 0.000),
it just picks the wrong legal direction (E2 = 0.586). That pattern is
consistent with two very different stories:

  (S1) PLANNING is the bottleneck — the policy sees everything but cannot
       choose the right direction, so a bigger policy network is the fix.
  (S2) PERCEPTION is the bottleneck — the policy's own view of the maze is
       noisy, so a better encoder is the fix.

These lead to opposite architectures, so they must be separated. This file
does that by building the *fastest possible* pipeline that still solves the
task:

    pixels ─> small detector (agent cell / goal cell / wall map)
           ─> deterministic BFS planner on the DECODED map
           ─> one move per tick

There is no learned policy here at all. If this pipeline reaches ~1.0, then
the right architecture for a game agent is **small perception + symbolic
planner**, and the policy network should be deleted, not enlarged. If it
fails, the detector is the problem and the fix is a better encoder.

Three numbers are reported:
  detector accuracy (agent cell / goal cell / wall cells)
  end-to-end closed-loop success of the full pipeline
  the decomposition: success when the oracle map is used (planning only)
  vs success when the detected map is used (perception + planning)
"""
from __future__ import annotations

import argparse
import json
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from envs import IMG, MAZE, STEPS, bfs_dist, make_maze, render_maze, step_maze
from model import SpatialCNN

CELL = IMG // MAZE


def decode_positions(frame):
    """Ground truth cell indices from the maze state (used only for labels)."""
    return None


def build_dataset(n_maze, seed):
    rng = random.Random(seed)
    X = np.zeros((n_maze, 3, IMG, IMG), dtype=np.uint8)
    agent = np.zeros((n_maze,), dtype=np.int64)
    goal = np.zeros((n_maze,), dtype=np.int64)
    walls = np.zeros((n_maze, MAZE, MAZE), dtype=np.uint8)
    for i in range(n_maze):
        st = make_maze(rng)
        X[i] = np.clip(st.frame * 255, 0, 255).astype(np.uint8)
        agent[i] = st.agent[0] * MAZE + st.agent[1]
        goal[i] = st.goal[0] * MAZE + st.goal[1]
        walls[i] = st.walls.astype(np.uint8)
    return X, agent, goal, walls


class Detector(nn.Module):
    """Frame -> agent cell, goal cell, wall map.

    TWO separate trunks, not one shared trunk:
      * `pos` trunk  -> agent + goal cell classification (sparse CE)
      * `wall` trunk -> per-cell wall classification (dense BCE)

    The first version used one shared trunk with a joint loss and it FAILED:
    the dense wall BCE term dominated the shared trunk's gradient and the
    sparse agent/goal signal never took hold (measured: agent 0.026 = chance,
    wall-cell 0.8744 = the majority-class constant). Splitting the trunks
    removes the conflict entirely — the repo's own dual-tower finding
    (README: "single tower, mixed tasks" collapses) reappears here in a new
    place.
    """

    def __init__(self, width=32, depth=3):
        super().__init__()
        self.pos_trunk = SpatialCNN(3, width, depth, in_size=IMG, out_dim=256)
        self.wall_trunk = SpatialCNN(3, width, depth, in_size=IMG, out_dim=256)
        self.agent_head = nn.Linear(256, MAZE * MAZE)
        self.goal_head = nn.Linear(256, MAZE * MAZE)
        self.wall_head = nn.Linear(256, MAZE * MAZE)

    def forward(self, x):
        p = self.pos_trunk(x)
        w = self.wall_trunk(x)
        return self.agent_head(p), self.goal_head(p), self.wall_head(w)


def bfs_plan(wall_map, agent_cell, goal_cell):
    """Deterministic shortest path on a decoded map. Returns an action index.

    BUG FIXED here: the first version reconstructed the path backwards but
    returned `prev[goal][1]` — the action that *arrives at* the goal, i.e. the
    LAST move of the path, not the first. With that bug the "oracle map"
    pipeline scored 0.395 instead of 1.000, which would have looked like a
    planning failure when it was a one-line reconstruction error. The path is
    now walked back collecting actions, and the FIRST move (the last one
    collected when walking backwards) is returned.
    """
    ar, ac = divmod(int(agent_cell), MAZE)
    gr, gc = divmod(int(goal_cell), MAZE)
    if (ar, ac) == (gr, gc):
        return None
    prev = {}
    q = deque([(ar, ac)])
    seen = {(ar, ac)}
    found = False
    while q:
        r, c = q.popleft()
        if (r, c) == (gr, gc):
            found = True
            break
        for a, (dr, dc) in enumerate(STEPS):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < MAZE and 0 <= nc < MAZE):
                continue
            if wall_map[nr, nc] or (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            prev[(nr, nc)] = ((r, c), a)
            q.append((nr, nc))
    if not found:
        return None
    cur = (gr, gc)
    acts = []
    while cur != (ar, ac):
        parent, a = prev[cur]
        acts.append(a)
        cur = parent
    return acts[-1]


def eval_pipeline(det, device, Xte, n=200, seed=999, use_oracle_map=False):
    """Run detect -> BFS -> move, on the SAME mazes for oracle and detected."""
    rng = random.Random(seed)
    mazes = [make_maze(rng) for _ in range(n)]
    idx = 0
    succ_det, succ_orc = 0, 0
    ticks_det, ticks_orc = [], []
    with torch.no_grad():
        for base in mazes:
            # --- planning only: the true map
            st = base
            t = 0
            while t < 3 * MAZE:
                a = bfs_plan(st.walls, st.agent[0] * MAZE + st.agent[1],
                             st.goal[0] * MAZE + st.goal[1])
                if a is None:
                    break
                st, done = step_maze(st, a); t += 1
                if done:
                    break
            succ_orc += int(st.agent == st.goal); ticks_orc.append(t)

            # --- perception + planning
            st = base
            t = 0
            while t < 3 * MAZE:
                fr = np.clip(st.frame * 255, 0, 255).astype(np.uint8)
                x = torch.from_numpy(fr).unsqueeze(0).to(device).float().div(255.)
                la, lg, lw = det(x)
                a_cell = int(la.argmax(-1).item()); g_cell = int(lg.argmax(-1).item())
                w = (lw[0] > 0).reshape(MAZE, MAZE).cpu().numpy().astype(bool)
                if use_oracle_map:
                    w = st.walls
                    a_cell = st.agent[0] * MAZE + st.agent[1]
                    g_cell = st.goal[0] * MAZE + st.goal[1]
                a = bfs_plan(w, a_cell, g_cell)
                if a is None:
                    a = 0
                st, done = step_maze(st, a); t += 1
                if done:
                    break
            succ_det += int(st.agent == st.goal); ticks_det.append(t)
    return dict(success_detected=succ_det / n, success_oracle_map=succ_orc / n,
                ticks_detected=float(np.mean(ticks_det)), ticks_oracle=float(np.mean(ticks_orc)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--mazes", type=int, default=4000)
    ap.add_argument("--out", default="perception_solver_result.json")
    args = ap.parse_args()
    device = args.device

    X, A, G, W = build_dataset(args.mazes, seed=0)
    Xte, Ate, Gte, Wte = build_dataset(800, seed=1)
    Xt = torch.from_numpy(X).float().div(255.)
    Xte_t = torch.from_numpy(Xte).float().div(255.)
    print(f"train {len(X)} test {len(Xte)}", flush=True)

    det = Detector().to(device)
    print(f"detector params {sum(p.numel() for p in det.parameters())/1e6:.3f}M", flush=True)
    opt = torch.optim.AdamW([
        {"params": list(det.pos_trunk.parameters()) + list(det.agent_head.parameters()) + list(det.goal_head.parameters()), "lr": args.lr},
        {"params": list(det.wall_trunk.parameters()) + list(det.wall_head.parameters()), "lr": args.lr},
    ], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    lf_a = nn.CrossEntropyLoss()
    lf_w = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.0))
    At = torch.from_numpy(A); Gt = torch.from_numpy(G)
    Wt = torch.from_numpy(W).float().reshape(len(W), -1)
    n = len(X); rng = np.random.default_rng(0)

    for ep in range(args.epochs):
        det.train()
        perm = rng.permutation(n)
        tot, nb = 0.0, 0
        for i in range(0, n, args.bs):
            idx = perm[i:i+args.bs]
            la, lg, lw = det(Xt[idx].to(device))
            loss = lf_a(la, At[idx].to(device)) + lf_a(lg, Gt[idx].to(device)) \
                + lf_w(lw, Wt[idx].to(device))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        if ep % 5 == 4 or ep == args.epochs - 1:
            with torch.no_grad():
                la, lg, lw = det(Xte_t.to(device))
                aa = (la.argmax(-1).cpu().numpy() == Ate).mean()
                gg = (lg.argmax(-1).cpu().numpy() == Gte).mean()
                ww = ((lw > 0).cpu().numpy() == Wte.reshape(len(Wte), -1)).mean()
            print(f"  ep{ep+1:02d} loss {tot/nb:.4f} | agent {aa:.4f} goal {gg:.4f} wall-cells {ww:.4f}", flush=True)
    det.eval()
    torch.save(det.state_dict(), "perception_detector.pt")
    print("saved perception_detector.pt", flush=True)

    with torch.no_grad():
        la, lg, lw = det(Xte_t.to(device))
        aa = float((la.argmax(-1).cpu().numpy() == Ate).mean())
        gg = float((lg.argmax(-1).cpu().numpy() == Gte).mean())
        ww = float(((lw > 0).cpu().numpy() == Wte.reshape(len(Wte), -1)).mean())

    res = eval_pipeline(det, device, Xte, n=200)
    print(f"\ndetector: agent {aa:.4f}  goal {gg:.4f}  wall-cell {ww:.4f}", flush=True)
    print(f"closed-loop success: detected-map {res['success_detected']:.3f} | "
          f"oracle-map {res['success_oracle_map']:.3f}", flush=True)
    print(f"mean ticks: detected {res['ticks_detected']:.1f} | oracle {res['ticks_oracle']:.1f}", flush=True)

    # latency of detect + plan
    x = Xte_t[:1].to(device)
    with torch.no_grad():
        for _ in range(10): det(x)
        if device == "xpu": torch.xpu.synchronize()
        t0 = time.perf_counter()
        for _ in range(100): det(x)
        if device == "xpu": torch.xpu.synchronize()
        ms = (time.perf_counter() - t0) / 100 * 1000
    print(f"detect latency {ms:.2f} ms (+~0.05 ms BFS) -> {1000/ms:.0f} decisions/s", flush=True)

    out = dict(detector=dict(agent_acc=aa, goal_acc=gg, wall_cell_acc=ww),
               pipeline=res, detect_latency_ms=ms, decisions_per_s=1000/ms,
               params=sum(p.numel() for p in det.parameters()))
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    import time
    main()
