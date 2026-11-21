"""G2 gate: can a model learn the v3 task WITHOUT any memory mechanism?

This is the gate that decides whether P1.5 continues at all.

Pass condition (fixed before running, per review):
    accuracy(concat) >= chance + 0.10   on the INDEPENDENT TEST split,
    reproduced on >= 2 of 3 seeds.

"concat" here means the memoryless condition: both chunks are joined into a
single input string and encoded in one pass. No state is carried. If even this
cannot learn the task, then a memory mechanism has nothing to improve on and
comparing `carried` vs `reset` would be meaningless -- we must stop and fix the
task instead of training more.

Two encoders are tried:
  1. a bag-of-words position-aware tiny model (fast, from scratch) -- tells us
     whether the task is learnable in principle
  2. optionally the pretrained 350M encoder (--use-350m)

Usage:
    python scripts/g2_concat_gate.py --data D:/Jaume/data/synth_v3 --seeds 0 1 2
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


def load_items(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def concat_text(it):
    """The memoryless condition: both chunks in ONE input string."""
    return " ".join(it["chunks"])


def answer_text(it):
    return it["candidates"][it["gold"]]


class BowEncoder(nn.Module):
    """Bag-of-words with a projection. Position is deliberately NOT modelled,
    so succeeding here genuinely requires combining facts, not reading order."""

    def __init__(self, vocab, dim=256):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim, padding_idx=0)
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.GELU(),
                                 nn.Linear(dim, dim))

    def forward(self, ids):
        mask = (ids != 0).unsqueeze(-1).float()
        h = self.emb(ids)
        h = (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return torch.nn.functional.normalize(self.net(h), dim=-1)


def build_vocab(texts):
    words = set()
    for t in texts:
        words.update(t.lower().split())
    return {w: i + 1 for i, w in enumerate(sorted(words))}


def encode(texts, vocab, max_len=96):
    out = torch.zeros(len(texts), max_len, dtype=torch.long)
    for i, t in enumerate(texts):
        for j, w in enumerate(t.lower().split()[:max_len]):
            out[i, j] = vocab.get(w, 0)
    return out


def run_bow(train, test, seed, epochs, dim, bs, lr, dev):
    torch.manual_seed(seed)
    # The vocabulary must cover the TEST vocabulary too. Pseudo-word names are
    # generated per split and never overlap, so a train-only vocabulary mapped
    # every test candidate name to the padding id, making all candidates
    # identical and the accuracy meaningless.
    vocab = build_vocab([concat_text(it) for it in train]
                        + [answer_text(it) for it in train]
                        + [concat_text(it) for it in test]
                        + [c for it in test for c in it["candidates"]])
    model = BowEncoder(len(vocab) + 1, dim).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    lossf = MultipleNegativesRankingLoss()

    c_ids = encode([concat_text(it) for it in train], vocab).to(dev)
    a_ids = encode([answer_text(it) for it in train], vocab).to(dev)
    n = len(train)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for s in range(0, n, bs):
            idx = perm[s : s + bs]
            if len(idx) < 2:
                continue
            ce = model(c_ids[idx])
            ae = model(a_ids[idx])
            loss = lossf(ce, ae)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

    model.eval()
    with torch.no_grad():
        # Every candidate must be scored as a TRAINING-STYLE answer string, so
        # the readout lives on the same distribution as the loss. Candidate
        # names are unique pseudo-words, so "the name" IS the answer text and
        # the model was trained to map context -> name. We therefore encode the
        # names with the same encoder call used at training time.
        flat_names, owner = [], []
        for i, it in enumerate(test):
            for c in it["candidates"]:
                flat_names.append(c)
                owner.append(i)
        nm = model(encode(flat_names, vocab).to(dev))
        te_c = model(encode([concat_text(it) for it in test], vocab).to(dev))
        correct = 0
        start = 0
        for i, it in enumerate(test):
            k = len(it["candidates"])
            block = nm[start : start + k]
            start += k
            sc = te_c[i] @ block.T
            if int(sc.argmax()) == it["gold"]:
                correct += 1
    return correct / len(test)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=fr"{PROJECT_ROOT}\data\synth_v3")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    dev = resolve_device()
    train = load_items(os.path.join(args.data, "train.jsonl"))
    test = load_items(os.path.join(args.data, "test.jsonl"))
    chance = sum(1.0 / len(it["candidates"]) for it in test) / len(test)
    gate = chance + 0.10

    print(f"[G2] train={len(train)} test={len(test)} chance={chance:.3f} "
          f"gate(concat >= chance+0.10)={gate:.3f}")

    accs = []
    for sd in args.seeds:
        t0 = time.time()
        acc = run_bow(train, test, sd, args.epochs, args.dim, args.bs, args.lr, dev)
        accs.append(acc)
        print(f"[G2] seed {sd}: concat acc={acc:.3f} "
              f"({'PASS' if acc >= gate else 'fail'}) ({time.time()-t0:.0f}s)",
              flush=True)

    passed = sum(a >= gate for a in accs)
    print(f"\n[G2] seeds over gate: {passed}/{len(accs)}")
    print(f"[G2] mean={sum(accs)/len(accs):.3f} chance={chance:.3f} gate={gate:.3f}")
    ok = passed >= 2
    print(f"\n[G2] GATE: {'PASS' if ok else 'FAIL'}")
    if ok:
        print("     -> the memoryless baseline CAN learn the task; comparing")
        print("        carried/reset/shuffled is now meaningful.")
    else:
        print("     -> STOP. A memory mechanism has nothing to improve on.")
        print("        Fix the task or the training setup; do not train JM-R.")

    out = os.path.join(args.data, "g2_concat_gate.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"chance": chance, "gate": gate, "accs": accs,
                   "seeds": args.seeds, "passed": passed, "gate_ok": ok}, f, indent=2)
    print(f"\n[G2] wrote {out}")


if __name__ == "__main__":
    main()
