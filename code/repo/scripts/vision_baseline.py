"""Pure-CNN baseline: can the task be learned at all without Jaume?

If this learns and the Jaume bridge does not, the bottleneck is the bridge.
If this also fails, the task or the optimisation is broken -- and we must not
blame the encoder.
"""
import argparse
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
REPO = os.path.join(PROJECT_ROOT, "code", "repo")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jaume.device import resolve_device, seed_all, describe
from vision_suite import build

FAMILIES = ["attr", "count", "spatial", "compare", "scene", "multi_attr"]


def to_tensor(img):
    import torchvision.transforms.functional as TF
    return TF.to_tensor(img)


class Net(nn.Module):
    def __init__(self, n_cls):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, n_cls)

    def forward(self, x):
        return self.fc(self.body(x).flatten(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=9000)
    ap.add_argument("--n-val", type=int, default=1200)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)
    print("[base] device=%s (%s)" % (device, describe(device)), flush=True)
    seed_all(device, 0)

    t0 = time.time()
    tr = build(args.n_train, seed=11)
    va = build(args.n_val, seed=777)
    print("[base] data %d/%d %.0fs" % (len(tr), len(va), time.time() - t0),
          flush=True)

    # One class per (family, which, answer) so the classifier is not forced to
    # share a head across incompatible answer spaces. We still score per family.
    keys = sorted({(b[1], b[2], b[3]) for b in tr + va})
    k2i = {k: i for i, k in enumerate(keys)}
    print("[base] classes=%d" % len(keys), flush=True)

    net = Net(len(keys)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    def batches(items, bs, shuffle, epoch=0):
        idx = list(range(len(items)))
        if shuffle:
            random.Random(epoch).shuffle(idx)
        for i in range(0, len(items) - bs + 1, bs):
            sel = idx[i:i + bs]
            x = torch.stack([to_tensor(items[j][0]) for j in sel]).to(device)
            y = torch.tensor([k2i[(items[j][1], items[j][2], items[j][3])]
                              for j in sel], device=device)
            yield x, y, [items[j] for j in sel]

    @torch.no_grad()
    def evaluate():
        net.eval()
        per = {f: [0, 0] for f in FAMILIES}
        for x, y, meta in batches(va, 64, False):
            p = net(x).argmax(-1)
            for i, m in enumerate(meta):
                fam = m[1]
                per[fam][1] += 1
                per[fam][0] += int(p[i].item() == y[i].item())
        return per

    for ep in range(args.epochs):
        net.train()
        tot = nb = 0
        t0 = time.time()
        for x, y, _ in batches(tr, args.batch_size, True, ep):
            loss = F.cross_entropy(net(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss); nb += 1
        sched.step()
        per = evaluate()
        parts = ["%s %.3f" % (f, per[f][0] / max(per[f][1], 1)) for f in FAMILIES]
        mean = sum(per[f][0] / max(per[f][1], 1) for f in FAMILIES) / len(FAMILIES)
        print("[base] ep %d/%d loss %.3f | %s | mean %.3f (%.0fs)" % (
            ep + 1, args.epochs, tot / max(nb, 1), "  ".join(parts), mean,
            time.time() - t0), flush=True)

    per = evaluate()
    print("=== FINAL CNN PROFILE ===", flush=True)
    for f in FAMILIES:
        c, n = per[f]
        print("  %-11s %.4f  (%d/%d)" % (f, c / max(n, 1), c, n), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
