"""Training entry for the Jaume-Actor (SigLIP2 + Jaume-350M) on NPU.

Two tasks, one model family:

  ui    : candidate grounding — pick the right element for an instruction
  maze  : closed-loop control from pixels

Both use the same skeleton: one forward pass per decision, softmax CE over
the candidate/action set, frozen vision tower, top-K Jaume layers unfrozen.

This file is written to run unchanged on Ascend 910B3 (torch_npu) and on the
local Intel XPU. It does not import torch_npu unless --device npu is asked
for, and it never assumes CUDA.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "code", "repo"))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from envs import CROP, IMG, MAZE, STEPS, bfs_dist, make_maze, step_maze  # noqa: E402
from envs_hard import make_ui_task_hard  # noqa: E402
from train_ui_hard import N_CAND, build as build_ui, batch as ui_batch, evaluate as ui_eval  # noqa: E402

MAXD = MAZE * MAZE
BLOCK = -(MAXD + 2)


def get_device(name: str):
    if name == "npu":
        import torch_npu  # noqa: F401

        return torch.device("npu:0")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda:0")
    if name == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu:0")
    return torch.device("cpu")


# ------------------------------------------------------------------ maze data

def reward_vector(walls, agent, goal):
    r = []
    for dr, dc in STEPS:
        nr, nc = agent[0] + dr, agent[1] + dc
        if not (0 <= nr < MAZE and 0 <= nc < MAZE) or walls[nr, nc]:
            r.append(BLOCK); continue
        d = bfs_dist(walls, (nr, nc), goal)
        r.append(-(d if d is not None else MAXD))
    return np.array(r, np.float32)


def collect_maze(n_maze, frames, seed):
    rng = random.Random(seed)
    X, R = [], []
    for _ in range(n_maze):
        st = make_maze(rng)
        for _ in range(frames):
            r = reward_vector(st.walls, st.agent, st.goal)
            X.append(np.clip(st.frame * 255, 0, 255).astype(np.uint8))
            R.append(r)
            st, done = step_maze(st, int(np.argmax(r)))
            if done:
                break
    return np.stack(X), np.stack(R)


class MazeJaumeActor(nn.Module):
    """Screen -> 4 action logits, using the SigLIP2 + Jaume trunk."""

    def __init__(self, siglip, jaume, fallback=False):
        super().__init__()
        from model_jaume import CandidateScorer

        self.fallback = fallback
        if fallback:
            from model import SpatialCNN

            self.cnn = SpatialCNN(3, 64, 4, in_size=IMG, out_dim=512)
            self.head = nn.Linear(512, 4)
            return
        self.vision = siglip
        self.trunk = jaume
        H = jaume.hidden
        self.screen_proj = nn.Linear(siglip.dim, H)
        self.head = nn.Sequential(nn.Linear(H, 256), nn.GELU(), nn.Linear(256, 4))

    def forward(self, frame):
        if self.fallback:
            return self.head(self.cnn(frame))
        x = self.vision(frame)
        if isinstance(x, (tuple, list)):
            x = x[0]
        if x.dim() == 3 and x.shape[1] != self.vision.dim:
            x = x.flatten(1, 2).transpose(1, 2)
        seq = self.screen_proj(x)
        h, _ = self.trunk.forward_embed(seq)
        return self.head(h.mean(dim=1))


# ----------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="npu")
    ap.add_argument("--task", choices=["ui", "maze"], default="ui")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--train", type=int, default=20000)
    ap.add_argument("--test", type=int, default=3000)
    ap.add_argument("--mazes", type=int, default=3000)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--siglip", default="google/siglip2-base-patch16-224")
    ap.add_argument("--jaume", default=os.environ.get("JAUME_CAUSAL_CKPT",
                                                    "350m-causal-mlm/final"))
    ap.add_argument("--unfreeze-top", type=int, default=4)
    ap.add_argument("--fallback", action="store_true", help="use the generic CNN trunk")
    ap.add_argument("--out", default="jaume_actor.json")
    ap.add_argument("--ckpt", default="jaume_actor.pt")
    args = ap.parse_args()

    device = get_device(args.device)
    print(f"device {device}", flush=True)

    from model_jaume import JaumeActor, JaumeTrunk, SigLIPBackbone

    if args.task == "ui":
        tr = build_ui(args.train, seed=0)
        te = build_ui(args.test, seed=1)
        print(f"ui train {len(tr['ans'])} test {len(te['ans'])}", flush=True)

        model = JaumeActor(args.siglip, args.jaume, args.unfreeze_top,
                           device=device, fallback=args.fallback).to(device)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
        lossf = nn.CrossEntropyLoss()
        order = np.arange(len(tr["ans"]))
        rng = np.random.default_rng(0)
        for ep in range(args.epochs):
            model.train()
            rng.shuffle(order)
            tot, nb = 0.0, 0
            for i in range(0, len(order), args.bs):
                idx = order[i : i + args.bs]
                logits = model(*ui_batch(tr, idx, device))
                loss = lossf(logits, torch.from_numpy(tr["ans"][idx]).to(device))
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                tot += loss.item(); nb += 1
            sched.step()
            if ep % 2 == 1 or ep == args.epochs - 1:
                acc = ui_eval(model, te, device)
                print(f"  ep{ep+1:02d} loss {tot/nb:.4f} top-1 {acc:.4f}", flush=True)
                torch.save(model.state_dict(), args.ckpt)
        acc = ui_eval(model, te, device)
        json.dump(dict(task="ui", top1=acc), open(args.out, "w"), indent=2)
        print(f"UI top-1 {acc:.4f}", flush=True)
        return

    # ---- maze
    X, R = collect_maze(args.mazes, args.frames, seed=0)
    T = torch.softmax(torch.from_numpy(R), dim=-1)
    Xt = torch.from_numpy(X).float().div(255.)
    print(f"maze frames {len(X)}", flush=True)

    if args.fallback:
        model = MazeJaumeActor(None, None, fallback=True).to(device)
    else:
        sig = SigLIPBackbone(args.siglip, device).to(device)
        trk = JaumeTrunk(args.jaume, args.unfreeze_top, device).to(device)
        model = MazeJaumeActor(sig, trk).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    lossf = nn.CrossEntropyLoss()
    n = len(X); rng = np.random.default_rng(0)
    for ep in range(args.epochs):
        model.train()
        perm = rng.permutation(n)
        tot, nb = 0.0, 0
        for i in range(0, n, args.bs):
            idx = perm[i : i + args.bs]
            loss = lossf(model(Xt[idx].to(device)), T[idx].to(device))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        if ep % 2 == 1 or ep == args.epochs - 1:
            with torch.no_grad():
                agree = (model(Xt[:2000].to(device)).argmax(-1).cpu().numpy()
                         == R[:2000].argmax(-1)).mean()
            print(f"  ep{ep+1:02d} ce {tot/nb:.4f} agree {agree:.4f}", flush=True)
            torch.save(model.state_dict(), args.ckpt)
    print("maze training done", flush=True)


if __name__ == "__main__":
    main()
