"""Diagnostic probe: can the vision trunk localise the agent and the goal?

Before spending more on the control task, separate the two failure modes:

  (a) perception  — the trunk cannot tell where the agent / goal / walls are
  (b) decision    — the trunk sees them but cannot pick the right action

Probe: freeze a randomly-initialised SpatialCNN, train a linear probe on top
to predict the agent cell and the goal cell from a single frame (81 classes
each). If the probe cannot localise them, no policy head on top of this trunk
can ever control the maze, and the fix is capacity/resolution, not data.

Reference points for the same probe:
  * majority class = 1/64 = 0.016
  * a plain global-average-pooled CNN (no spatial layout) = chance by
    construction, and is reported as a control.
"""
from __future__ import annotations

import argparse
import json
import random

import numpy as np
import torch
import torch.nn as nn

from envs import IMG, MAZE, make_maze
from model import SpatialCNN, cnn


def collect(n, seed):
    rng = random.Random(seed)
    X = np.zeros((n, 3, IMG, IMG), dtype=np.uint8)
    A = np.zeros((n,), dtype=np.int64)
    G = np.zeros((n,), dtype=np.int64)
    for i in range(n):
        st = make_maze(rng)
        X[i] = np.clip(st.frame * 255, 0, 255).astype(np.uint8)
        A[i] = st.agent[0] * MAZE + st.agent[1]
        G[i] = st.goal[0] * MAZE + st.goal[1]
    return X, A, G


def run(trunk, name, Xtr, Atr, Gtr, Xte, Ate, Gte, device, epochs=30):
    feat_dim = trunk(Xtr[:2].float().div(255.0).to(device)).shape[1]
    # TWO separate heads. A single shared 64-way head cannot report both
    # accuracies: its argmax can only be one class, so agent-acc and goal-acc
    # come out complementary and the numbers look like a perception failure
    # that is actually an artefact of the probe.
    head_a = nn.Sequential(nn.Linear(feat_dim, 256), nn.ReLU(inplace=True),
                            nn.Linear(256, MAZE * MAZE)).to(device)
    head_g = nn.Sequential(nn.Linear(feat_dim, 256), nn.ReLU(inplace=True),
                            nn.Linear(256, MAZE * MAZE)).to(device)
    params = list(trunk.parameters()) + list(head_a.parameters()) + list(head_g.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    n = len(Xtr)
    rng = np.random.default_rng(0)
    for ep in range(epochs):
        perm = rng.permutation(n)
        tot, nb = 0.0, 0
        for i in range(0, n, 128):
            idx = perm[i : i + 128]
            xb = Xtr[idx].float().div(255.0).to(device)
            f = trunk(xb)
            y = torch.from_numpy(Atr[idx]).to(device)
            yg = torch.from_numpy(Gtr[idx]).to(device)
            loss = 0.5 * (lossf(head_a(f), y) + lossf(head_g(f), yg))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item(); nb += 1
    with torch.no_grad():
        f = trunk(torch.from_numpy(Xte).float().div(255.0).to(device))
        agent_acc = (head_a(f).argmax(-1).cpu().numpy() == Ate).mean()
        goal_acc = (head_g(f).argmax(-1).cpu().numpy() == Gte).mean()
    print(f"[{name}] probe: agent-cell {agent_acc:.4f} | goal-cell {goal_acc:.4f} "
          f"(chance {1/(MAZE*MAZE):.4f})", flush=True)
    return dict(agent_acc=float(agent_acc), goal_acc=float(goal_acc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    device = args.device

    Xtr, Atr, Gtr = collect(args.n, seed=0)
    Xte, Ate, Gte = collect(500, seed=1)
    Xtr_t = torch.from_numpy(Xtr)
    print(f"train frames {len(Xtr)}", flush=True)

    res = {}
    res["spatial_cnn"] = run(SpatialCNN(3, 32, 3, in_size=IMG).to(device), "spatial_cnn",
                              Xtr_t, Atr, Gtr, Xte, Ate, Gte, device, args.epochs)

    # control: global-average-pooled trunk (no position survives)
    class Pooled(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = cnn(3, 256, width=32, depth=3, in_size=IMG)

        def forward(self, x):
            return self.net(x)

    res["pooled_cnn"] = run(Pooled().to(device), "pooled_cnn",
                            Xtr_t, Atr, Gtr, Xte, Ate, Gte, device, args.epochs)
    json.dump(res, open("probe_localise.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
