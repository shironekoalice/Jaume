"""G3: the memory comparison, with the shuffled-state control.

Preconditions already met:
  G1  task has no shortcut (lexical leakage 0.000, oracle 1.000, answers unique)
  G2  memoryless learning is possible (bow 0.44-0.47, 350M concat 0.471 vs
      chance 0.167)

Conditions compared, all trained from the SAME base with the SAME budget:

  concat    both chunks in one input, no state        (upper reference)
  reset     chunks encoded in sequence, state NOT passed
  carried   chunks encoded in sequence, state passed
  shuffled  chunks encoded in sequence, state passed but RANDOMLY SHUFFLED
            across items in the batch

Why `shuffled` matters (raised in review): if `carried` beats `reset` merely
because carrying a state changes batch statistics or adds capacity, then feeding
a *wrong* state from another item should help just as much. Only a gain that
survives `carried > reset` AND collapses under `shuffled` can be attributed to
memory that is actually about the current input.

The task: chunk A states two protocols with numeric values, chunk B says which
EXTREME to take ("smaller"/"larger"), and the answer is the protocol name.
Only chunk B is given to the query side; chunk A must come from memory.

Usage:
    python scripts/g3_memory.py --data D:/Jaume/data/synth_v3 --epochs 2
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
from jaume.losses.contrastive import MultipleNegativesRankingLoss  # noqa: E402
from jaume.inference.encode import JaumeEmbedding  # noqa: E402

RECURRENT = ("recurrent_memory", "recurrent_memory_gate")


def load_items(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def split_chunks(it):
    """Return (chunk_A, chunk_B) regardless of the shuffled order in the file.

    Chunk A is the one carrying the two numeric facts; chunk B is the
    instruction. This is determined by the presence of digits, not by position.
    """
    def has_num(s):
        return any(ch.isdigit() for ch in s)
    a, b = it["chunks"][0], it["chunks"][1]
    if has_num(a) and not has_num(b):
        return a, b
    if has_num(b) and not has_num(a):
        return b, a
    # both/neither: fall back to the longer one being A
    return (a, b) if len(a) >= len(b) else (b, a)


def query_text(it):
    """The query side: chunk B (instruction) + question. NEVER chunk A."""
    _, b = split_chunks(it)
    return f"{b} {it['query']}"


def target_text(it):
    return it["candidates"][it["gold"]]


def build_repr(model, items, condition, max_length=256):
    """Document/passage representation per condition, on CPU tensors."""
    device = model.device
    out = []
    if condition == "concat":
        texts = [" ".join(split_chunks(it)) for it in items]
        with torch.no_grad():
            for i in range(0, len(texts), 64):
                e = model.tokenizer(texts[i : i + 64], padding=True,
                                    truncation=True, max_length=max_length,
                                    return_tensors="pt").to(device)
                emb, _ = model.model(e["input_ids"], e["attention_mask"])
                out.append(emb.float().cpu())
        return torch.cat(out)

    # sequential conditions
    for it in items:
        a, _ = split_chunks(it)
        state = None
        pooled = None
        with torch.no_grad():
            for ch in (a,):
                e = model.tokenizer([ch], padding=True, truncation=True,
                                    max_length=max_length,
                                    return_tensors="pt").to(device)
                if condition in ("carried", "shuffled"):
                    pooled, _, state = model.model.encode(
                        e["input_ids"], e["attention_mask"], state)
                else:  # reset: state never passed
                    pooled, _, _ = model.model.encode(
                        e["input_ids"], e["attention_mask"], None)
        # `carried`/`shuffled` read the STATE (dim 1 = slots), while `reset`
        # has no state to read, so it reads the pooled vector instead. Both are
        # reduced to one [dim] vector per item.
        vec = state if condition in ("carried", "shuffled") else pooled
        if vec.dim() == 3:          # [1, slots, dim]
            vec = vec.mean(dim=1)
        elif vec.dim() == 1:        # already [dim]
            vec = vec.unsqueeze(0)
        out.append(vec.float().cpu())
    return torch.cat(out)


def eval_condition(model, items, condition, shuffle=False, max_length=256):
    device = model.device
    rep = build_repr(model, items, condition, max_length)
    if shuffle:
        # feed a state from a DIFFERENT item: same tensor distribution, wrong
        # content. If the gain is real it must disappear here.
        g = torch.Generator().manual_seed(0)
        rep = rep[torch.randperm(rep.size(0), generator=g)]
    rep = torch.nn.functional.normalize(rep, dim=-1)

    q = query_text_batch(model, items, max_length)
    names, owner = [], []
    for i, it in enumerate(items):
        for c in it["candidates"]:
            names.append(c)
            owner.append(i)
    nm = torch.nn.functional.normalize(encode_texts(model, names, max_length),
                                       dim=-1)
    correct = ptr = 0
    for i, it in enumerate(items):
        k = len(it["candidates"])
        block = nm[ptr : ptr + k]
        ptr += k
        sc = rep[i] @ block.T + q[i] @ block.T
        if int(sc.argmax()) == it["gold"]:
            correct += 1
    return correct / len(items)


def encode_texts(model, texts, max_length=256):
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            e = model.tokenizer(texts[i : i + 64], padding=True,
                                truncation=True, max_length=max_length,
                                return_tensors="pt").to(model.device)
            emb, _ = model.model(e["input_ids"], e["attention_mask"])
            out.append(emb.float().cpu())
    return torch.cat(out)


def query_text_batch(model, items, max_length=256):
    q = encode_texts(model, [query_text(it) for it in items], max_length)
    return torch.nn.functional.normalize(q, dim=-1)


def train(arm, data, epochs, lr, bs, seed, out):
    dev = resolve_device()
    torch.manual_seed(seed)
    train_items = load_items(os.path.join(data, "train.jsonl"))
    model = JaumeEmbedding.from_pretrained(arm)
    base = model.model
    base.train()
    opt = torch.optim.AdamW(base.parameters(), lr=lr, weight_decay=0.01)
    lossf = MultipleNegativesRankingLoss()
    n = len(train_items)
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(n)
        for s in range(0, n, bs):
            idx = perm[s : s + bs].tolist()
            if len(idx) < 2:
                continue
            sub = [train_items[i] for i in idx]
            # the training signal: query side (chunk B + question) must match the
            # answer name; the passage side is chunk A, so the model must learn
            # to combine "which extreme" with "which number".
            qs = [query_text(it) for it in sub]
            ps = [split_chunks(it)[0] for it in sub]
            ts = [target_text(it) for it in sub]
            e_q = model.tokenizer(qs, padding=True, truncation=True,
                                  max_length=256, return_tensors="pt").to(dev)
            e_p = model.tokenizer(ps, padding=True, truncation=True,
                                  max_length=256, return_tensors="pt").to(dev)
            e_t = model.tokenizer(ts, padding=True, truncation=True,
                                  max_length=256, return_tensors="pt").to(dev)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=has_amp(dev)):
                q_emb = base(e_q["input_ids"], e_q["attention_mask"])[0]
                # passage encoded as the FIRST chunk with no prior state
                p_emb = base(e_p["input_ids"], e_p["attention_mask"])[0]
                t_emb = base(e_t["input_ids"], e_t["attention_mask"])[0]
                loss = lossf(p_emb, t_emb) + lossf(q_emb, t_emb)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        print(f"  [{os.path.basename(out)}] ep{ep} loss {float(loss):.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    base.eval()
    test_items = load_items(os.path.join(data, "test.jsonl"))
    res = {}
    for cond in ("concat", "reset", "carried"):
        res[cond] = eval_condition(model, test_items, cond)
        print(f"  [{os.path.basename(out)}] {cond:8s} {res[cond]:.3f}", flush=True)
    res["shuffled"] = eval_condition(model, test_items, "carried", shuffle=True)
    print(f"  [{os.path.basename(out)}] shuffled {res['shuffled']:.3f}", flush=True)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "g3_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=fr"{PROJECT_ROOT}\data\synth_v3")
    ap.add_argument("--arms", default="B5=/D:/Jaume/output/mem/B5-recurrent-s1/final")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--outroot", default=fr"{PROJECT_ROOT}\output\mem")
    args = ap.parse_args()

    test_items = load_items(os.path.join(args.data, "test.jsonl"))
    chance = sum(1.0 / len(it["candidates"]) for it in test_items) / len(test_items)
    print(f"[G3] chance={chance:.3f}")

    allres = {}
    for spec in args.arms.split(","):
        name, arm = spec.split("=", 1)
        arm = arm.lstrip("/")
        for sd in args.seeds:
            out = os.path.join(args.outroot, f"G3-{name}-s{sd}")
            print(f"\n[G3] {name} seed {sd} arm={arm}")
            allres[f"{name}-s{sd}"] = train(arm, args.data, args.epochs,
                                            args.lr, args.bs, sd, out)

    print("\n" + "=" * 78)
    print(f"{'run':22s} {'concat':>7s} {'reset':>7s} {'carried':>8s} {'shuffled':>9s}")
    for k, r in allres.items():
        print(f"{k:22s} {r['concat']:7.3f} {r['reset']:7.3f} "
              f"{r['carried']:8.3f} {r['shuffled']:9.3f}")
    print("=" * 78)
    print("VERDICT RULE (fixed in advance):")
    print("  memory helps  <=> carried > reset AND carried > shuffled")
    print("  if carried ~= reset or carried ~= shuffled, no memory benefit")


if __name__ == "__main__":
    main()
