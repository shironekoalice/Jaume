"""The decisive test on frozen features: does the carried STATE add anything?

Everything that blocked us before is now removed:
  * backbone is frozen -> no destructive fine-tuning (full FT broke the signal)
  * head trains in ~3 minutes -> we can afford many conditions and seeds
  * the only thing that varies is HOW the block representation is produced

Conditions (same features, same head architecture, same budget, same seeds):
    carried   block repr = state after reading blocks 0..i  (memory path)
    static    block repr = the block's own pooled vector    (no memory)
    shuffled  carried states permuted across items          (wrong memory)

If `carried` beats `static` and `shuffled` here, the memory mechanism adds
information that the encoder's per-block representation lacks. If it does not,
then on this task the state carries nothing useful -- independent of any
fine-tuning issue.

Also reports the BM25 reference (0.32) so the head is never mistaken for SOTA.

Usage:
    python scripts/run_state_on_frozen.py --split 0 --seed 0
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


def set_f1(pred, gold):
    p, g = set(pred), set(gold)
    i = len(p & g)
    return 2 * i / (len(p) + len(g)) if p and g else 0.0


def featurize_condition(rm, rows, condition):
    """Frozen features for one condition. Returns [(q, blockreprs, row)]."""
    out = []
    B = 8
    with torch.no_grad():
        for s in range(0, len(rows), B):
            batch = rows[s : s + B]
            if condition == "static":
                reps = QB.run_batch_states(rm, batch, False, MAX_LENGTH,
                                           MAX_BLOCKS)
            else:
                reps = QB.run_batch_states(rm, batch, True, MAX_LENGTH,
                                           MAX_BLOCKS)
            if condition == "shuffled":
                g = torch.Generator().manual_seed(0)
                perm = torch.randperm(len(batch), generator=g).tolist()
                reps = [reps[j] for j in perm]
            qs = QB.question_embs(rm, [r["question"] for r in batch])
            for i, r in enumerate(batch):
                out.append((F.normalize(qs[i], dim=-1).cpu(),
                            F.normalize(reps[i], dim=-1).cpu(), r))
    return out


def train_head(feats, dim, epochs, lr, dev, seed):
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
                q, b, r = feats[i]
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
    return head


def score_head(head, feats, dev):
    head.eval()
    per_item = {}
    with torch.no_grad():
        for q, b, r in feats:
            sc = head(q.to(dev), b.to(dev))
            k = len(r["gold_blocks"])
            pred = torch.topk(sc, k).indices.tolist()
            per_item[r["id"]] = set_f1(pred, r["gold_blocks"])
    return per_item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=fr"{PROJECT_ROOT}\data\qasper_comp_clean")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--outroot", default=fr"{PROJECT_ROOT}\output\comp_state_frozen")
    args = ap.parse_args()

    dev = resolve_device()
    torch.manual_seed(args.seed)
    d = os.path.join(args.data, f"split{args.split}")
    tr = load(os.path.join(d, "train.jsonl"))[:520]
    te = load(os.path.join(d, "test.jsonl"))
    print(f"[sfz] split={args.split} seed={args.seed} "
          f"train={len(tr)} test={len(te)}", flush=True)

    rm = make_frozen(dev)
    t0 = time.time()
    results = {}
    for cond in ("carried", "static", "shuffled"):
        trf = featurize_condition(rm, tr, cond)
        tef = featurize_condition(rm, te, cond)
        dim = trf[0][1].size(-1)
        head = train_head(trf, dim, args.epochs, args.lr, dev, args.seed)
        per_item = score_head(head, tef, dev)
        results[cond] = per_item
        print(f"  {cond:9s} test_f1 = "
              f"{statistics.mean(per_item.values()):.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        del trf, tef, head
        empty_cache(dev)

    summary = {c: statistics.mean(v.values()) for c, v in results.items()}
    print(f"\n[sfz] split={args.split} seed={args.seed} "
          f"carried={summary['carried']:.4f} static={summary['static']:.4f} "
          f"shuffled={summary['shuffled']:.4f}")
    print(f"       c-static={summary['carried']-summary['static']:+.4f}  "
          f"c-shuffled={summary['carried']-summary['shuffled']:+.4f}")
    print(f"       reference: BM25 0.322 (zero training)")

    os.makedirs(args.outroot, exist_ok=True)
    out = os.path.join(args.outroot, f"split{args.split}_seed{args.seed}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"split": args.split, "seed": args.seed,
                   "summary": summary, "per_item": results,
                   "epochs": args.epochs, "lr": args.lr}, f)
    print(f"[sfz] wrote {out}")


if __name__ == "__main__":
    main()
