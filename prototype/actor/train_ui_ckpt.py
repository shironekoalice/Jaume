"""Train only the 'full' UI variant and save its weights for the real-browser
evaluation. The full multi-variant run is wasteful here: the ablations are
for the report, the checkpoint is for the browser test.
"""
from __future__ import annotations

import argparse
import json
import random

import numpy as np
import torch
import torch.nn as nn

from envs import CROP, IMG
from envs_hard import make_ui_task_hard
from model import UIActor
from train_ui_hard import N_CAND, batch, build, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--train", type=int, default=12000)
    ap.add_argument("--test", type=int, default=2000)
    ap.add_argument("--ckpt", default="ui_hard_model.pt")
    ap.add_argument("--out", default="ui_ckpt_result.json")
    args = ap.parse_args()
    device = args.device

    tr = build(args.train, seed=0)
    te = build(args.test, seed=1)
    print(f"train {len(tr['ans'])} test {len(te['ans'])}", flush=True)

    model = UIActor().to(device)
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
            loss = lossf(model(*batch(tr, idx, device)), torch.from_numpy(tr["ans"][idx]).to(device))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        if ep % 5 == 4 or ep == args.epochs - 1:
            print(f"  ep{ep+1:02d} loss {tot/nb:.5f} top-1 {evaluate(model, te, device):.4f}", flush=True)

    torch.save(model.state_dict(), args.ckpt)
    acc = evaluate(model, te, device)
    print(f"saved {args.ckpt}  held-out top-1 {acc:.4f}", flush=True)
    json.dump(dict(top1=acc, ckpt=args.ckpt, epochs=args.epochs), open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
