"""Freeze the backbone, train only a small adapter on top of the frozen features.

Why this is the right next step
-------------------------------
Measured facts:
  * the pretrained encoder ALREADY separates gold blocks (cos +0.021, rank 3.27)
  * fine-tuning the whole 350M model DESTROYS that separation at every learning
    rate tried (1e-5 / 3e-5 / 1e-4 all end at sep <= +0.004)
  * training loss stays at 10-23 while a perfect solution would be ~0.001, so
    the full-model update is not converging; with only 514 training items this
    is classic over-parameterised fine-tuning, not a missing capability
  * gradients are healthy (total norm 14.3), so it is not a vanishing/exploding
    gradient problem

Therefore: keep the encoder frozen and learn only a small scoring head that
operates on (question, block) features. This tests whether the existing
representation is USABLE; if it is, the problem was fine-tuning, not the memory
mechanism. If even a frozen-feature head cannot beat BM25, the task is beyond
this encoder and the task must change.

The head sees concatenated [q, b, q*b, |q-b|] features per (question, block)
pair -- a standard pair encoder -- with very few parameters.

Usage:
    python scripts/run_frozen_head.py --split 0 --seed 0
"""
import os as _os
PROJECT_ROOT = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import argparse
import json
import os
import statistics
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = fr"{PROJECT_ROOT}\code\repo"
sys.path.insert(0, REPO)
from jaume.device import resolve_device, has_amp, make_scaler, empty_cache, max_memory_allocated, reset_peak_memory_stats, total_memory, seed_all
sys.path.insert(0, os.path.join(REPO, "scripts"))

from jaume.inference.encode import JaumeEmbedding  # noqa: E402
import qasper_batch as QB  # noqa: E402

MAX_BLOCKS = 12
MAX_LENGTH = 128
BASE = fr"{PROJECT_ROOT}\350m-mlm"


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def make_frozen(dev):
    rm = JaumeEmbedding.from_pretrained(BASE)
    base = rm.model
    from jaume.modeling.memory import build_memory_pooling
    base.config.pooling = "recurrent_memory_gate"
    base.config.memory_output = "new_slots"
    base.config.num_slots = 8
    base.config.slot_heads = 8
    base.pooling = build_memory_pooling(base.config)
    base.to(dev)
    sd = torch.load(os.path.join(BASE, "pytorch_model.bin"),
                    map_location="cpu", weights_only=True)
    ms = base.state_dict()
    for k, v in ms.items():
        if k in sd and tuple(sd[k].shape) == tuple(v.shape):
            v.copy_(sd[k])
    for p in base.parameters():
        p.requires_grad_(False)
    base.eval()
    return rm


class PairHead(nn.Module):
    """score(q, b) from frozen features; deliberately small."""

    def __init__(self, dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 4, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, 1))

    def forward(self, q, b):          # q: [D], b: [N, D]
        qe = q.expand_as(b)
        feat = torch.cat([qe, b, qe * b, (qe - b).abs()], dim=-1)
        return self.net(feat).squeeze(-1)


def set_f1(pred, gold):
    p, g = set(pred), set(gold)
    i = len(p & g)
    return 2 * i / (len(p) + len(g)) if p and g else 0.0


def featurize(rm, rows):
    """Frozen features for every (question, block) pair, batched."""
    out = []
    with torch.no_grad():
        for s in range(0, len(rows), 8):
            batch = rows[s : s + 8]
            reps = QB.run_batch_states(rm, batch, True, MAX_LENGTH, MAX_BLOCKS)
            qs = QB.question_embs(rm, [r["question"] for r in batch])
            for i, r in enumerate(batch):
                out.append((F.normalize(qs[i], dim=-1).cpu(),
                            F.normalize(reps[i], dim=-1).cpu(),
                            r))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=fr"{PROJECT_ROOT}\data\qasper_comp_clean")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--outroot", default=fr"{PROJECT_ROOT}\output\comp_head")
    args = ap.parse_args()

    dev = resolve_device()
    torch.manual_seed(args.seed)
    d = os.path.join(args.data, f"split{args.split}")
    tr = load(os.path.join(d, "train.jsonl"))[:520]
    te = load(os.path.join(d, "test.jsonl"))
    print(f"[head] split={args.split} seed={args.seed} "
          f"train={len(tr)} test={len(te)}", flush=True)

    rm = make_frozen(dev)
    print("[head] featurizing (frozen encoder)...", flush=True)
    t0 = time.time()
    tr_f = featurize(rm, tr)
    te_f = featurize(rm, te)
    dim = tr_f[0][1].size(-1)
    print(f"[head] done in {time.time()-t0:.0f}s, dim={dim}", flush=True)

    head = PairHead(dim).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-2)
    nparam = sum(p.numel() for p in head.parameters())
    print(f"[head] head params={nparam/1e3:.1f}K", flush=True)

    dl = torch.utils.data.DataLoader(
        list(range(len(tr_f))), batch_size=16, shuffle=True)
    for ep in range(args.epochs):
        head.train()
        tot, nb = 0.0, 0
        for idx in dl:
            loss = 0.0
            opt.zero_grad()
            for i in idx.tolist():
                q, b, r = tr_f[i]
                q, b = q.to(dev), b.to(dev)
                logits = head(q, b) * 8.0
                gold = set(r["gold_blocks"])
                y = torch.tensor([1.0 if j in gold else 0.0
                                  for j in range(len(logits))], device=dev)
                w = torch.where(y > 0, torch.full_like(y, 4.0),
                                torch.ones_like(y))
                loss = loss + F.binary_cross_entropy_with_logits(
                    logits, y, weight=w)
            (loss / len(idx)).backward()
            opt.step()
            tot += float(loss) / len(idx)
            nb += 1
        if ep % 5 == 0 or ep == args.epochs - 1:
            head.eval()
            with torch.no_grad():
                f1s = []
                for q, b, r in te_f:
                    sc = head(q.to(dev), b.to(dev))
                    k = len(r["gold_blocks"])
                    pred = torch.topk(sc, k).indices.tolist()
                    f1s.append(set_f1(pred, r["gold_blocks"]))
            print(f"  ep{ep:3d} loss {tot/max(1,nb):.4f} "
                  f"test_f1 {statistics.mean(f1s):.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    head.eval()
    with torch.no_grad():
        f1s = []
        for q, b, r in te_f:
            sc = head(q.to(dev), b.to(dev))
            k = len(r["gold_blocks"])
            pred = torch.topk(sc, k).indices.tolist()
            f1s.append(set_f1(pred, r["gold_blocks"]))
    final = statistics.mean(f1s)
    print(f"[head] split={args.split} seed={args.seed} test_f1={final:.4f}")
    print(f"[head] reference: BM25 0.322, full-finetune 0.14-0.21")

    os.makedirs(args.outroot, exist_ok=True)
    out = os.path.join(args.outroot, f"split{args.split}_seed{args.seed}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"split": args.split, "seed": args.seed,
                   "test_f1": final, "head_params": nparam,
                   "epochs": args.epochs}, f, indent=2)
    print(f"[head] wrote {out}")


if __name__ == "__main__":
    main()
