"""G3 step 0: can the 350M encoder learn the v3 task under `concat`?

The G2 gate was passed by a from-scratch bag-of-words model. Before comparing
memory conditions we must confirm the SAME task is learnable by the actual
Jaume-350M encoder under the memoryless `concat` strategy -- otherwise a memory
comparison on this encoder would again be confounded by "the model cannot learn
the task".

Inputs are built from the v3 synthetic items:
    query  = chunk B (the comparison instruction) + the question
    target = the answer protocol name
Chunks are NOT given to the query side; the 350M encoder sees only chunk B plus
the question, which is exactly the memoryless condition when chunk A is absent.

For `concat` (the gate) the model sees BOTH chunks in one string.

Usage:
    python scripts/g3_350m_concat.py --data D:/Jaume/data/synth_v3 --epochs 2
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

REPO = fr"{PROJECT_ROOT}\code\repo"
sys.path.insert(0, REPO)
from jaume.device import resolve_device, has_amp, make_scaler, empty_cache, max_memory_allocated, reset_peak_memory_stats, total_memory, seed_all
sys.path.insert(0, os.path.join(REPO, "scripts"))

from jaume.losses.contrastive import MultipleNegativesRankingLoss  # noqa: E402
from jaume.inference.encode import JaumeEmbedding  # noqa: E402


def load_items(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def enc_batch(model, texts, max_length=256, bs=64):
    outs = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            part = texts[i : i + bs]
            e = model.tokenizer(part, padding=True, truncation=True,
                                max_length=max_length,
                                return_tensors="pt").to(model.device)
            emb, _ = model.model(e["input_ids"], e["attention_mask"])
            outs.append(emb.float().cpu())
    return torch.cat(outs, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=fr"{PROJECT_ROOT}\data\synth_v3")
    ap.add_argument("--model", default=fr"{PROJECT_ROOT}\350m-mlm")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = resolve_device()
    train = load_items(os.path.join(args.data, "train.jsonl"))
    test = load_items(os.path.join(args.data, "test.jsonl"))
    chance = sum(1.0 / len(it["candidates"]) for it in test) / len(test)
    gate = chance + 0.10
    print(f"[G3-0] train={len(train)} test={len(test)} chance={chance:.3f} "
          f"gate={gate:.3f}")

    model = JaumeEmbedding.from_pretrained(args.model)
    base = model.model
    base.train()
    opt = torch.optim.AdamW(base.parameters(), lr=args.lr, weight_decay=0.01)
    lossf = MultipleNegativesRankingLoss()

    # concat condition: both chunks in one string -> answer name
    def ctx(it):
        return " ".join(it["chunks"]) + " " + it["query"]

    texts_c = [ctx(it) for it in train]
    texts_a = [it["candidates"][it["gold"]] for it in train]
    n = len(train)
    t0 = time.time()
    for ep in range(args.epochs):
        perm = torch.randperm(n)
        for s in range(0, n, args.bs):
            idx = perm[s : s + args.bs].tolist()
            if len(idx) < 2:
                continue
            ce = model.tokenizer([texts_c[i] for i in idx], padding=True,
                                 truncation=True, max_length=256,
                                 return_tensors="pt").to(dev)
            ae = model.tokenizer([texts_a[i] for i in idx], padding=True,
                                 truncation=True, max_length=256,
                                 return_tensors="pt").to(dev)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=has_amp(dev)):
                cemb = base(ce["input_ids"], ce["attention_mask"])[0]
                aemb = base(ae["input_ids"], ae["attention_mask"])[0]
                loss = lossf(cemb, aemb)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        print(f"[G3-0] epoch {ep} loss {float(loss):.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    base.eval()
    ctx_e = enc_batch(model, [ctx(it) for it in test])
    flat, owner = [], []
    for i, it in enumerate(test):
        for c in it["candidates"]:
            flat.append(c)
            owner.append(i)
    nm = enc_batch(model, flat)
    correct = 0
    ptr = 0
    ctx_e = torch.nn.functional.normalize(ctx_e, dim=-1)
    nm = torch.nn.functional.normalize(nm, dim=-1)
    for i, it in enumerate(test):
        k = len(it["candidates"])
        block = nm[ptr : ptr + k]
        ptr += k
        sc = ctx_e[i] @ block.T
        if int(sc.argmax()) == it["gold"]:
            correct += 1
    acc = correct / len(test)
    print(f"\n[G3-0] 350M concat test acc={acc:.3f} chance={chance:.3f} gate={gate:.3f}")
    print(f"[G3-0] {'PASS' if acc >= gate else 'FAIL'}")

    out = os.path.join(args.data, "g3_350m_concat.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"acc": acc, "chance": chance, "gate": gate,
                   "passed": acc >= gate, "epochs": args.epochs}, f, indent=2)
    print(f"[G3-0] wrote {out}")


if __name__ == "__main__":
    main()
