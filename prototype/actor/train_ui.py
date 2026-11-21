"""Train the UI actor and report the only numbers that matter.

Protocol (mirrors the repo's habit of declaring the protocol before the table):
  * 3000 train screens / 800 held-out screens, 6 candidates each, seed 0/1
  * loss = softmax CE across the 6 candidates *inside one screen* (§4.7 fix)
  * metric = top-1 accuracy on held-out screens
  * controls = random (1/6) and the hand-written color+shape heuristic
  * ablations = drop the screen frame, drop the crop, drop the box

Run:  python train_ui.py --device xpu --epochs 12
"""
from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn

from envs import IMG, CROP, N_WORDS, UITask, heuristic_ui, make_ui_task
from model import UIActor

N_CAND = 6


def build(n: int, seed: int, with_screen: bool = True):
    rng = random.Random(seed)
    crops = np.zeros((n, N_CAND, 3, CROP, CROP), dtype=np.uint8)
    boxes = np.zeros((n, N_CAND, 4), dtype=np.float32)
    screen = np.zeros((n, 3, IMG, IMG), dtype=np.uint8)
    text = np.zeros((n, 6), dtype=np.int64)
    ans = np.zeros((n,), dtype=np.int64)
    heur = np.zeros((n,), dtype=np.int64)
    for i in range(n):
        t: UITask = make_ui_task(rng, N_CAND, with_screen=with_screen)
        for j, c in enumerate(t.candidates):
            crops[i, j] = np.clip(c.crop * 255, 0, 255).astype(np.uint8)
            boxes[i, j] = c.box
        screen[i] = np.clip(t.screen * 255, 0, 255).astype(np.uint8)
        text[i, : len(t.instruction)] = t.instruction
        ans[i] = t.answer
        heur[i] = heuristic_ui(t)
    return dict(crops=crops, boxes=boxes, screen=screen, text=text, ans=ans, heur=heur)


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
    correct = 0
    for i in range(0, len(d["ans"]), bs):
        idx = np.arange(i, min(i + bs, len(d["ans"])))
        logits = model(*batch(d, idx, device))
        correct += (logits.argmax(-1).cpu().numpy() == d["ans"][idx]).sum()
    return correct / len(d["ans"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--train", type=int, default=3000)
    ap.add_argument("--test", type=int, default=800)
    ap.add_argument("--out", default="ui_result.json")
    args = ap.parse_args()

    device = args.device
    t0 = time.time()
    tr = build(args.train, seed=0)
    te = build(args.test, seed=1)
    print(f"data built in {time.time()-t0:.1f}s  (train {len(tr['ans'])} / test {len(te['ans'])})", flush=True)
    print(f"controls: random {1/N_CAND:.4f}   heuristic {np.mean(te['heur']==te['ans']):.4f}", flush=True)

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
        n_par = sum(p.numel() for p in model.parameters())
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
                tot += loss.item()
                nb += 1
            sched.step()
            if ep == args.epochs - 1 or ep % 4 == 3:
                acc = evaluate(model, te, device)
                print(f"  [{name}] ep{ep+1:02d} loss {tot/nb:.4f}  held-out top-1 {acc:.4f}", flush=True)
        acc = evaluate(model, te, device)
        results[name] = dict(top1=acc, params=n_par)
        print(f"[{name}] params {n_par/1e6:.3f}M  top-1 {acc:.4f}", flush=True)

    # latency of one decision on the full model
    model = UIActor().to(device).eval()
    idx = np.arange(1)
    *inputs, = batch(tr, idx, device)
    with torch.no_grad():
        for _ in range(10):
            model(*inputs)
        if device == "xpu":
            torch.xpu.synchronize()
        t = time.time()
        for _ in range(200):
            model(*inputs)
        if device == "xpu":
            torch.xpu.synchronize()
        ms = (time.time() - t) / 200 * 1000
    results["latency_ms_per_decision_bs1"] = ms
    print(f"latency: {ms:.2f} ms per screen decision (batch 1, {device})", flush=True)

    json.dump(results, open(args.out, "w"), indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
