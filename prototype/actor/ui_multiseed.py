"""Multi-seed stability check for the UI actor.

The UI actor scored 1.0000 on the hard synthetic benchmark (with decoy twins)
and 1.0000 on the real browser — but from a SINGLE seed. The maze detector
turned out to be bimodal under the same kind of single-seed reporting
(§3.6 in the report), so the UI number needs the same treatment.

This script trains the UI actor with several seeds on the hard benchmark and
reports the mean and the WORST seed for each ablation variant, which is the
statistic that matters.

Also re-runs the real-browser evaluation per seed, because the browser
evaluation is the one that could plausibly differ (real pixels, real DOM).
"""
from __future__ import annotations

import argparse
import json
import random

import numpy as np
import torch
import torch.nn as nn

from envs_hard import make_ui_task_hard
from model import UIActor
from train_ui_hard import N_CAND, build, evaluate


def train_one(tr, te, device, epochs, bs, lr, seed, cfg):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = UIActor(**cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lossf = nn.CrossEntropyLoss()
    order = np.arange(len(tr["ans"]))
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        model.train(); rng.shuffle(order)
        for i in range(0, len(order), bs):
            idx = order[i:i+bs]
            logits = model(*_batch(tr, idx, device))
            loss = lossf(logits, torch.from_numpy(tr["ans"][idx]).to(device))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sched.step()
    return model, evaluate(model, te, device)


def _batch(d, idx, device):
    f = lambda a: torch.from_numpy(a[idx]).to(device, non_blocking=True)
    return (f(d["crops"]).float() / 255.0, f(d["boxes"]),
            f(d["screen"]).float() / 255.0, f(d["text"]),
            torch.ones(len(idx), N_CAND, dtype=torch.bool, device=device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--train", type=int, default=6000)
    ap.add_argument("--test", type=int, default=1500)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="ui_multiseed_result.json")
    args = ap.parse_args()
    device = args.device

    tr = build(args.train, seed=0)
    te = build(args.test, seed=1)
    variants = {
        "full": dict(use_screen=True, use_crop=True, use_box=True),
        "no_screen": dict(use_screen=False, use_crop=True, use_box=True),
        "no_crop": dict(use_screen=True, use_crop=False, use_box=True),
        "no_box": dict(use_screen=True, use_crop=True, use_box=False),
        "text_only": dict(use_screen=False, use_crop=False, use_box=False),
    }
    res = {}
    for name, cfg in variants.items():
        accs = []
        for s in range(args.seeds):
            _, acc = train_one(tr, te, device, args.epochs, args.bs, args.lr, s, cfg)
            accs.append(acc)
            print(f"[{name}] seed{s}: top-1 {acc:.4f}", flush=True)
        res[name] = dict(mean=float(np.mean(accs)), min=float(np.min(accs)), per_seed=accs)
        print(f"  -> {name}: mean {np.mean(accs):.4f} worst {np.min(accs):.4f}", flush=True)

    json.dump(res, open(args.out, "w"), indent=2)
    print(json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items() if kk != "per_seed"}
                      for k, v in res.items()}, indent=2))


if __name__ == "__main__":
    main()
