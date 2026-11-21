"""Does ANY model learn the v2 cross-block task?

Motivation: the 350M pretrained encoder cannot reduce the loss on the clean v2
task (15 epochs, 4.87 -> 4.60). Two explanations remain:

  a) the task is simply not learnable by an encoder of this kind, or
  b) the pretrained NLI representation is the obstacle.

Training a tiny model from scratch separates them, and costs a couple of
minutes. If the small model learns it, the task is fine and the blocker is the
pretrained backbone / adaptation. If the small model also fails, the task design
is at fault and must change before any memory conclusion.

The tiny model is deliberately simple: an embedding table + mean pooling over
tokens + a linear projection to a small dimension, trained with the same
MultipleNegativesRankingLoss. Chunks A and B are concatenated for the passage
side so the model sees the whole context; the query side is the chunk-B text.
"""
import os as _os
PROJECT_ROOT = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn

REPO = fr"{PROJECT_ROOT}\code\repo"
sys.path.insert(0, REPO)
from jaume.device import resolve_device, has_amp, make_scaler, empty_cache, max_memory_allocated, reset_peak_memory_stats, total_memory, seed_all
from jaume.losses.contrastive import MultipleNegativesRankingLoss  # noqa: E402


class TinyEncoder(nn.Module):
    def __init__(self, vocab, dim=128, max_len=192):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim, padding_idx=0)
        self.proj = nn.Linear(dim, dim)
        self.register_buffer("pos", torch.zeros(max_len, dim), persistent=False)

    def forward(self, ids):
        mask = (ids != 0).unsqueeze(-1).float()
        h = self.emb(ids) * mask
        h = h.sum(1) / mask.sum(1).clamp(min=1.0)  # mean pool
        return torch.nn.functional.normalize(self.proj(h), dim=-1)


def build_vocab(rows):
    words = set()
    for q, p in rows:
        words.update(q.lower().split())
        words.update(p.lower().split())
    words = sorted(words)
    return {w: i + 1 for i, w in enumerate(words)}


def encode(texts, vocab, max_len):
    out = torch.zeros(len(texts), max_len, dtype=torch.long)
    for i, t in enumerate(texts):
        for j, w in enumerate(t.lower().split()[:max_len]):
            out[i, j] = vocab.get(w, 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=fr"{PROJECT_ROOT}\data\synth_crossblock\train_pairs_v2.jsonl")
    ap.add_argument("--test", default=fr"{PROJECT_ROOT}\data\synth_crossblock\test.jsonl")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=192)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = resolve_device()

    # training pairs: query (chunk B + question) -> positive (entity + colour)
    rows = [json.loads(l) for l in open(args.train, encoding="utf-8") if l.strip()]
    pairs = [(r["query"], r["positive"]) for r in rows]

    # the passage side must carry the ANSWER too, otherwise the model cannot
    # possibly predict it: concatenate the matching chunk A onto the query side
    # is NOT possible from this file alone, so we rebuild full items instead
    src_train = os.path.join(os.path.dirname(args.train), "train.jsonl")
    items = [json.loads(l) for l in open(src_train, encoding="utf-8") if l.strip()]
    trip = []
    for it in items:
        chunks = it["chunks"]
        a_idx = 0 if len(chunks[0]) >= len(chunks[1]) else 1
        b_idx = 1 - a_idx
        # context = both chunks (this is the "concat" strategy, no memory needed)
        context = f"{chunks[a_idx]} {chunks[b_idx]}"
        answer = f"{it['target_entity'].lower()} {it['answer']}"
        trip.append((it["query"], context, answer))

    vocab = build_vocab([(q, c) for q, c, _ in trip])
    print(f"[tiny] vocab={len(vocab)} train={len(trip)} dim={args.dim} dev={dev}")

    model = TinyEncoder(len(vocab) + 1, args.dim, args.max_len).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lossf = MultipleNegativesRankingLoss()

    q_ids = encode([q for q, _, _ in trip], vocab, args.max_len).to(dev)
    c_ids = encode([c for _, c, _ in trip], vocab, args.max_len).to(dev)
    a_ids = encode([a for _, _, a in trip], vocab, args.max_len).to(dev)

    n = len(trip)
    t0 = time.time()
    for ep in range(args.epochs):
        perm = torch.randperm(n)
        tot, nb = 0.0, 0
        for s in range(0, n, args.batch):
            idx = perm[s : s + args.batch]
            if len(idx) < 2:
                continue
            qe = model(q_ids[idx])
            se = model(c_ids[idx])
            ae = model(a_ids[idx])
            # the query must match the (entity, colour) answer, and the answer
            # must be reachable from the context: two symmetric losses
            loss = lossf(qe, ae) + lossf(se, ae)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            tot += float(loss)
            nb += 1
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"[tiny] epoch {ep:3d} loss {tot/max(1,nb):.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # ---- evaluate on the synthetic test set with an oracle-free readout ----
    test_items = [json.loads(l) for l in open(args.test, encoding="utf-8") if l.strip()]
    model.eval()
    chance = sum(1.0 / len(it["candidates"]) for it in test_items) / len(test_items)
    with torch.no_grad():
        context = []
        for it in test_items:
            chunks = it["chunks"]
            a_idx = 0 if len(chunks[0]) >= len(chunks[1]) else 1
            b_idx = 1 - a_idx
            context.append(f"{chunks[a_idx]} {chunks[b_idx]}")
        ce = model(encode(context, vocab, args.max_len).to(dev))
        # candidates are colour words; score each against the context
        all_c = sorted({c for it in test_items for c in it["candidates"]})
        cv_map = {c: i for i, c in enumerate(all_c)}
        cve = model(encode(all_c, vocab, args.max_len).to(dev))
        correct = 0
        for i, it in enumerate(test_items):
            scores = ce[i] @ cve.T
            pick = [cv_map[c] for c in it["candidates"]]
            best = pick[int(scores[pick].argmax())]
            if best == it["gold"]:
                correct += 1
    acc = correct / len(test_items)
    print(f"\n[tiny] test n={len(test_items)} chance={chance:.3f} acc={acc:.3f}")
    if acc > chance + 0.10:
        print("[tiny] PASS: the task IS learnable -> the blocker is the pretrained")
        print("       backbone / adaptation, not the task design.")
    else:
        print("[tiny] FAIL: even a from-scratch model does not learn it ->")
        print("       the task design is at fault; change it before judging memory.")


if __name__ == "__main__":
    main()
