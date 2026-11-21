"""Does the state add INDEPENDENT value when it is offered alongside the block?

Previous finding (memory-breakthrough-diagnosis.md):
    static   0.29  (block's own vector, no memory)
    carried  0.21  (state replaces the block vector)
    shuffled 0.17
So the state is informative (carried > shuffled, +0.032) but is a BAD REPLACEMENT
for the block vector (carried < static, -0.079).

The open question: is the state useless, or merely mis-wired? If the state holds
information the block vector lacks, then offering BOTH should beat either alone.
A model given both channels can learn to use whichever helps.

Conditions (frozen backbone, identical head architecture and budget):
    static    [b]                block only
    carried   [s]                state only
    both      [b ; s]            concatenated channels
    dual      score = f(b) + f(s)  two independent scorers, summed

`both` and `dual` are the tests. If they beat `static`, the state carries
complementary information and is worth keeping. If they do not, the state adds
nothing beyond what the block vector already provides, and the negative result
stands.

Usage:
    python scripts/run_concat_state.py --split 0 --seed 0
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
    def __init__(self, dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 4, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, 1))

    def forward(self, q, b):
        qe = q.expand_as(b)
        return self.net(torch.cat([qe, b, qe * b, (qe - b).abs()],
                                  dim=-1)).squeeze(-1)


class DualHead(nn.Module):
    """Two independent pair scorers (one per channel), summed."""

    def __init__(self, dim, hidden=256):
        super().__init__()
        self.block_head = PairHead(dim, hidden)
        self.state_head = PairHead(dim, hidden)

    def forward(self, q, b, s):
        return self.block_head(q, b) + self.state_head(q, s)


def set_f1(pred, gold):
    p, g = set(pred), set(gold)
    i = len(p & g)
    return 2 * i / (len(p) + len(g)) if p and g else 0.0


def featurize(rm, rows):
    """Return [(q, block_repr, state_repr, row)] with frozen features."""
    out = []
    B = 8
    with torch.no_grad():
        for s in range(0, len(rows), B):
            batch = rows[s : s + B]
            blk = QB.run_batch_states(rm, batch, False, MAX_LENGTH, MAX_BLOCKS)
            st = QB.run_batch_states(rm, batch, True, MAX_LENGTH, MAX_BLOCKS)
            qs = QB.question_embs(rm, [r["question"] for r in batch])
            for i, r in enumerate(batch):
                out.append((F.normalize(qs[i], dim=-1).cpu(),
                            F.normalize(blk[i], dim=-1).cpu(),
                            F.normalize(st[i], dim=-1).cpu(), r))
    return out


def train_single(feats, dim, epochs, lr, dev, seed, which="block"):
    torch.manual_seed(seed)
    head = PairHead(dim).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-2)
    n = len(feats)
    for ep in range(epochs):
        head.train()
        order = torch.randperm(n).tolist()
        for s in range(0, n, 16):
            idx = order[s : s + 16]
            opt.zero_grad()
            loss = 0.0
            for i in idx:
                q, b, st, r = feats[i]
                rep = (b if which == "block" else st).to(dev)
                q = q.to(dev)
                logits = head(q, rep) * 8.0
                gold = set(r["gold_blocks"])
                y = torch.tensor([1.0 if j in gold else 0.0
                                  for j in range(len(logits))], device=dev)
                w = torch.where(y > 0, torch.full_like(y, 4.0),
                                torch.ones_like(y))
                loss = loss + F.binary_cross_entropy_with_logits(
                    logits, y, weight=w)
            (loss / len(idx)).backward()
            opt.step()
    return head


def score_single(head, feats, dev, which="block"):
    head.eval()
    per = {}
    with torch.no_grad():
        for q, b, st, r in feats:
            rep = (b if which == "block" else st).to(dev)
            sc = head(q.to(dev), rep)
            k = len(r["gold_blocks"])
            per[r["id"]] = set_f1(torch.topk(sc, k).indices.tolist(),
                                  r["gold_blocks"])
    return per


def train_dual(feats, dim, epochs, lr, dev, seed):
    torch.manual_seed(seed)
    head = DualHead(dim).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-2)
    n = len(feats)
    for ep in range(epochs):
        head.train()
        order = torch.randperm(n).tolist()
        for s in range(0, n, 16):
            idx = order[s : s + 16]
            opt.zero_grad()
            loss = 0.0
            for i in idx:
                q, b, st, r = feats[i]
                logits = head(q.to(dev), b.to(dev), st.to(dev)) * 8.0
                gold = set(r["gold_blocks"])
                y = torch.tensor([1.0 if j in gold else 0.0
                                  for j in range(len(logits))], device=dev)
                w = torch.where(y > 0, torch.full_like(y, 4.0),
                                torch.ones_like(y))
                loss = loss + F.binary_cross_entropy_with_logits(
                    logits, y, weight=w)
            (loss / len(idx)).backward()
            opt.step()
    return head


def score_dual(head, feats, dev):
    head.eval()
    per = {}
    with torch.no_grad():
        for q, b, st, r in feats:
            sc = head(q.to(dev), b.to(dev), st.to(dev))
            k = len(r["gold_blocks"])
            per[r["id"]] = set_f1(torch.topk(sc, k).indices.tolist(),
                                  r["gold_blocks"])
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=fr"{PROJECT_ROOT}\data\qasper_comp_clean")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--outroot", default=fr"{PROJECT_ROOT}\output\comp_concat")
    args = ap.parse_args()

    dev = resolve_device()
    d = os.path.join(args.data, f"split{args.split}")
    tr = load(os.path.join(d, "train.jsonl"))[:520]
    te = load(os.path.join(d, "test.jsonl"))
    print(f"[cat] split={args.split} seed={args.seed} "
          f"train={len(tr)} test={len(te)}", flush=True)

    rm = make_frozen(dev)
    t0 = time.time()
    trf = featurize(rm, tr)
    tef = featurize(rm, te)
    dim = trf[0][1].size(-1)
    print(f"[cat] featurized in {time.time()-t0:.0f}s", flush=True)

    res = {}
    h = train_single(trf, dim, args.epochs, args.lr, dev, args.seed, "block")
    res["static"] = score_single(h, tef, dev, "block")
    print(f"  static  {statistics.mean(res['static'].values()):.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)

    h = train_single(trf, dim, args.epochs, args.lr, dev, args.seed, "state")
    res["carried"] = score_single(h, tef, dev, "state")
    print(f"  carried {statistics.mean(res['carried'].values()):.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)

    h = train_dual(trf, dim, args.epochs, args.lr, dev, args.seed)
    res["dual"] = score_dual(h, tef, dev)
    print(f"  dual    {statistics.mean(res['dual'].values()):.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)

    summary = {k: statistics.mean(v.values()) for k, v in res.items()}
    print(f"\n[cat] split={args.split} seed={args.seed} " +
          " ".join(f"{k}={v:.4f}" for k, v in summary.items()))
    print(f"      static->dual delta = {summary['dual']-summary['static']:+.4f}")
    print(f"      reference: BM25 0.322")

    os.makedirs(args.outroot, exist_ok=True)
    out = os.path.join(args.outroot, f"split{args.split}_seed{args.seed}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"split": args.split, "seed": args.seed,
                   "summary": summary, "per_item": res,
                   "epochs": args.epochs}, f)
    print(f"[cat] wrote {out}")


if __name__ == "__main__":
    main()
