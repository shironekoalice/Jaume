"""SciFact multi-stage training: does the JM-R state transfer to real text?

Same mechanism as the synthetic Plan-1 experiment, but on real scientific text:

    stage 1: stage1 text (first half of the cited abstracts) -> s_1
    stage 2: stage2 text (second half)                       -> s_2
    stage 3: the CLAIM                                      -> 3-way label

The claim never appears in stages 1-2 (verified by check_scifact_data.py L1-L2),
so a correct label requires information carried through s_2.

Architecture is FROZEN and identical to the plan-1 model: 8-slot recurrent
pooler, memory_output=new_slots, no GRU, no aux loss.

Training uses a classification head on the step-3 representation. The head is a
small linear layer on the pooled embedding -- it is the prediction head every
arm shares, not part of the memory mechanism.

Usage:
    python scripts/train_scifact.py --seed 0 --epochs 12 --out ...
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
import torch.nn.functional as F

REPO = fr"{PROJECT_ROOT}\code\repo"
sys.path.insert(0, REPO)
from jaume.device import resolve_device, has_amp, make_scaler, empty_cache, max_memory_allocated, reset_peak_memory_stats, total_memory, seed_all

from jaume.inference.encode import JaumeEmbedding  # noqa: E402


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def macro_f1(y_true, y_pred, n_cls=3):
    f1s = []
    for c in range(n_cls):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / n_cls, f1s


class Head(nn.Module):
    """Shared prediction head: pooled embedding -> 3 classes."""

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 256), nn.GELU(),
                                 nn.Dropout(0.1), nn.Linear(256, 3))

    def forward(self, x):
        return self.net(x)


def run_stage(model, texts, state, max_length=256, bs=32):
    """Encode a stage and return (pooled_after_projection, new_state)."""
    outs, states = [], []
    for i in range(0, len(texts), bs):
        part = texts[i : i + bs]
        e = model.tokenizer(part, padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt").to(model.device)
        st = None if state is None else state[i : i + bs].to(model.device)
        pooled, _, new_state = model.model.encode(e["input_ids"],
                                                  e["attention_mask"], st)
        outs.append(model.model.projection(pooled))
        if new_state is not None:
            states.append(new_state.detach().float().cpu())
    emb = torch.cat(outs, dim=0)
    st_all = torch.cat(states, dim=0) if states else None
    return emb, st_all


def evaluate(model, head, rows, dev, max_length=256):
    model.model.eval()
    head.eval()
    s1 = [r["stage1"] for r in rows]
    s2 = [r["stage2"] for r in rows]
    cl = [r["claim"] for r in rows]
    y = [r["label"] for r in rows]
    preds = {}
    with torch.no_grad():
        _, st1 = run_stage(model, s1, None, max_length)
        _, st2 = run_stage(model, s2, st1, max_length)
        for cond in ("carried", "reset", "shuffled"):
            st = None
            if cond == "carried":
                st = st2
            elif cond == "shuffled":
                g = torch.Generator().manual_seed(0)
                st = st2[torch.randperm(st2.size(0), generator=g)]
            emb, _ = run_stage(model, cl, st, 256)
            logits = head(emb.to(dev))
            preds[cond] = logits.argmax(-1).cpu().tolist()
        # concat: both halves available in ONE pass, no state
        both = [f"{r['stage1']} {r['stage2']}" for r in rows]
        _, stc = run_stage(model, both, None, 512)
        emb, _ = run_stage(model, cl, None, 256)
        # give concat the same second forward: encode claim with a state built
        # from the FULL context, so it has no unfair disadvantage
        emb2, _ = run_stage(model, cl, stc, 256)
        preds["concat"] = head(emb2.to(dev)).argmax(-1).cpu().tolist()
    out = {}
    for cond, p in preds.items():
        mf1, per = macro_f1(y, p)
        out[cond] = {"macro_f1": mf1, "per_class_f1": per,
                     "acc": sum(1 for a, b in zip(y, p) if a == b) / len(y)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=fr"{PROJECT_ROOT}\data\scifact_blocks")
    ap.add_argument("--init-from", default=fr"{PROJECT_ROOT}\350m-mlm")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-length", type=int, default=256)
    args = ap.parse_args()

    dev = resolve_device()
    torch.manual_seed(args.seed)

    tr = load_jsonl(os.path.join(args.data, "train.jsonl"))
    va = load_jsonl(os.path.join(args.data, "val.jsonl"))
    te = load_jsonl(os.path.join(args.data, "test.jsonl"))
    print(f"[sf] train={len(tr)} val={len(va)} test={len(te)}")

    model = JaumeEmbedding.from_pretrained(args.init_from)
    base = model.model
    from jaume.modeling.memory import build_memory_pooling
    base.config.pooling = "recurrent_memory_gate"
    base.config.memory_output = "new_slots"
    base.config.num_slots = 8
    base.config.slot_heads = 8
    base.pooling = build_memory_pooling(base.config)
    base.to(dev)

    src = os.path.join(args.init_from, "pytorch_model.bin")
    if os.path.exists(src):
        sd = torch.load(src, map_location="cpu", weights_only=True)
        ms = base.state_dict()
        n = sum(1 for k, v in ms.items()
                if k in sd and tuple(sd[k].shape) == tuple(v.shape))
        for k, v in ms.items():
            if k in sd and tuple(sd[k].shape) == tuple(v.shape):
                v.copy_(sd[k])
        print(f"[sf] reused {n}/{len(ms)} tensors")

    head = Head(base.config.embedding_dim).to(dev)
    params = list(base.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    # class weights: NEI/SUPPORT/CONTRADICT are imbalanced
    cnt = [sum(1 for r in tr if r["label"] == c) for c in range(3)]
    w = torch.tensor([len(tr) / (3 * max(1, c)) for c in cnt], dtype=torch.float).to(dev)
    print(f"[sf] class counts={cnt} weights={[round(float(x),2) for x in w]}")

    base.train()
    head.train()
    best = {"macro_f1": -1}
    t0 = time.time()
    for ep in range(args.epochs):
        perm = torch.randperm(len(tr))
        tot, nb = 0.0, 0
        for s in range(0, len(tr), args.bs):
            idx = perm[s : s + args.bs].tolist()
            if len(idx) < 2:
                continue
            sub = [tr[i] for i in idx]
            s1 = [r["stage1"] for r in sub]
            s2 = [r["stage2"] for r in sub]
            cl = [r["claim"] for r in sub]
            y = torch.tensor([r["label"] for r in sub], device=dev)

            with torch.autocast(dev, dtype=torch.bfloat16, enabled=has_amp(dev)):
                e1 = model.tokenizer(s1, padding=True, truncation=True,
                                     max_length=args.max_length, return_tensors="pt").to(dev)
                _, _, st1 = base.encode(e1["input_ids"], e1["attention_mask"], None)
                e2 = model.tokenizer(s2, padding=True, truncation=True,
                                     max_length=args.max_length, return_tensors="pt").to(dev)
                _, _, st2 = base.encode(e2["input_ids"], e2["attention_mask"], st1)
                ec = model.tokenizer(cl, padding=True, truncation=True,
                                     max_length=128, return_tensors="pt").to(dev)
                pooled, _, _ = base.encode(ec["input_ids"], ec["attention_mask"], st2)
                emb = base.projection(pooled)
                logits = head(emb.float())
                loss = F.cross_entropy(logits, y, weight=w)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            tot += float(loss)
            nb += 1

        val = evaluate(model, head, va, dev, args.max_length)
        mf1 = val["carried"]["macro_f1"]
        print(f"  [sf-s{args.seed}] ep{ep} loss {tot/max(1,nb):.4f} "
              f"val carried macroF1 {mf1:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if mf1 > best["macro_f1"]:
            best = {"macro_f1": mf1, "epoch": ep}
            os.makedirs(args.out, exist_ok=True)
            torch.save(base.state_dict(), os.path.join(args.out, "pytorch_model.bin"))
            torch.save(head.state_dict(), os.path.join(args.out, "head.bin"))
            model.tokenizer.save_pretrained(args.out)
            with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as f:
                json.dump(base.config.to_dict(), f, indent=2)

    print(f"[sf] best val carried macroF1 {best['macro_f1']:.4f} @ep{best['epoch']}")
    with open(os.path.join(args.out, "train_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "best_val": best,
                   "seconds": round(time.time() - t0, 1)}, f, indent=2)


if __name__ == "__main__":
    main()
