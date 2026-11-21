"""Train / evaluate the UI actor on the *hard* benchmark (twins + position).

Same protocol as train_ui.py, but with position disambiguation, so the model
must fuse a local crop feature with a global position feature.

Reported controls:
  random   = 1/N
  heuristic = color+shape match (cannot resolve twins -> floor)
  oracle    = color+shape+band+column (1.000 by construction)
"""
from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn

from envs import CROP, IMG
from envs_hard import heuristic_hard, make_ui_task_hard, oracle_hard
from model import UIActor

N_CAND = 6


def build(n: int, seed: int):
    rng = random.Random(seed)
    crops = np.zeros((n, N_CAND, 3, CROP, CROP), dtype=np.uint8)
    boxes = np.zeros((n, N_CAND, 4), dtype=np.float32)
    screen = np.zeros((n, 3, IMG, IMG), dtype=np.uint8)
    text = np.zeros((n, 6), dtype=np.int64)
    ans = np.zeros((n,), dtype=np.int64)
    heur = np.zeros((n,), dtype=np.int64)
    orac = np.zeros((n,), dtype=np.int64)
    for i in range(n):
        s, cands, instr, a = make_ui_task_hard(rng, N_CAND)
        for j, c in enumerate(cands):
            crops[i, j] = np.clip(c.crop * 255, 0, 255).astype(np.uint8)
            boxes[i, j] = c.box
        screen[i] = np.clip(s * 255, 0, 255).astype(np.uint8)
        text[i, : len(instr)] = instr
        ans[i] = a
        heur[i] = heuristic_hard(cands, instr)
        orac[i] = oracle_hard(cands, instr)
    return dict(crops=crops, boxes=boxes, screen=screen, text=text, ans=ans, heur=heur, orac=orac)


def batch(d, idx, device):
    f = lambda a: torch.from_numpy(a[idx]).to(device, non_blocking=True)
    return (
        f(d["crops"]).float() / 255.0,
        f(d["boxes"]),
        f(d["screen"]).float() / 255.0,
        f(d["text"]),
        torch.ones(len(idx), N_CAND, dtype=torch.bool, device=device),
    )


@torch.no_grad()
def evaluate(model, d, device, bs=100):
    model.eval()
    c = 0
    for i in range(0, len(d["ans"]), bs):
        idx = np.arange(i, min(i + bs, len(d["ans"])))
        logits = model(*batch(d, idx, device))
        c += (logits.argmax(-1).cpu().numpy() == d["ans"][idx]).sum()
    return c / len(d["ans"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--train", type=int, default=4000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--out", default="ui_hard_result.json")
    ap.add_argument("--ckpt", default="ui_hard_model.pt")
    args = ap.parse_args()

    device = args.device
    tr = build(args.train, seed=0)
    te = build(args.test, seed=1)
    print(f"controls  random {1/N_CAND:.4f} | heuristic(color+shape) {np.mean(te['heur']==te['ans']):.4f} "
          f"| oracle(color+shape+pos) {np.mean(te['orac']==te['ans']):.4f}", flush=True)

    results = {}
    variants = {
        "full": dict(use_screen=True, use_crop=True, use_box=True),
        "no_screen": dict(use_screen=False, use_crop=True, use_box=True),
        "no_crop": dict(use_screen=True, use_crop=False, use_box=True),
        "no_box": dict(use_screen=True, use_crop=True, use_box=False),
        "text_only": dict(use_screen=False, use_crop=False, use_box=False),
    }
    for name, cfg in variants.items():
        model = UIActor(**cfg).to(device)
        npar = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
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
                logits = model(*batch(tr, idx, device))
                loss = lossf(logits, torch.from_numpy(tr["ans"][idx]).to(device))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += loss.item(); nb += 1
            sched.step()
            if ep % 5 == 4 or ep == args.epochs - 1:
                print(f"  [{name}] ep{ep+1:02d} loss {tot/nb:.4f} top-1 {evaluate(model, te, device):.4f}", flush=True)
        acc = evaluate(model, te, device)
        results[name] = dict(top1=acc, params=npar)
        if name == "full":
            torch.save(model.state_dict(), args.ckpt)
            print(f"saved full-model weights to {args.ckpt}", flush=True)
        print(f"[{name}] params {npar/1e6:.3f}M top-1 {acc:.4f}", flush=True)

    json.dump(results, open(args.out, "w"), indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
