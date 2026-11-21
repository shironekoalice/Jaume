"""Fast evaluation of experiment A, numerically equivalent to g3_memory.

g3_memory's `build_repr` encodes chunk A one item at a time (one tokenizer call
and one forward per item), which on 1000 test items x 4 conditions dominates the
runtime. Here chunk A is pre-encoded in batches and only the stateful walk stays
serial -- the state walk is the thing being measured, so it must stay serial to
be honest, but the stateless part does not.

Mapping to the conditions:
  concat   = encode "chunkA chunkB" in one pass
  reset    = per-chunk walk with state=None every step
  carried  = per-chunk walk passing the state
  shuffled = same as carried, states permuted across items before scoring
"""
import os as _os
PROJECT_ROOT = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import argparse
import json
import os
import statistics
import sys

import torch

REPO = fr"{PROJECT_ROOT}\code\repo"
sys.path.insert(0, REPO)
from jaume.device import resolve_device, has_amp, make_scaler, empty_cache, max_memory_allocated, reset_peak_memory_stats, total_memory, seed_all
sys.path.insert(0, os.path.join(REPO, "scripts"))

from jaume.inference.encode import JaumeEmbedding  # noqa: E402
import g3_memory as G  # noqa: E402


def enc(model, texts, max_length=256, bs=64):
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


def walk(model, item_texts, pass_state, max_length=256):
    """Serial per-chunk walk. `pass_state` toggles the condition being measured."""
    out = []
    states = [None] * len(item_texts)
    for i, text in enumerate(item_texts):
        with torch.no_grad():
            e = model.tokenizer([text], padding=True, truncation=True,
                                max_length=max_length,
                                return_tensors="pt").to(model.device)
            pooled, _, new_state = model.model.encode(
                e["input_ids"], e["attention_mask"],
                states[i] if pass_state else None)
        states[i] = new_state
        vec = new_state.mean(dim=1) if pass_state else pooled
        out.append(vec.float().cpu())
    return torch.cat(out, dim=0)


def score_repr(rep, items, model, chance_check=True):
    """Score a [N, dim] representation against each item's candidates."""
    rep = torch.nn.functional.normalize(rep, dim=-1)
    qt = [G.query_text(it) for it in items]
    q = torch.nn.functional.normalize(enc(model, qt), dim=-1)
    flat, _ = [], []
    for it in items:
        for c in it["candidates"]:
            flat.append(c)
    nm = torch.nn.functional.normalize(enc(model, flat), dim=-1)
    correct = ptr = 0
    for i, it in enumerate(items):
        k = len(it["candidates"])
        block = nm[ptr : ptr + k]
        ptr += k
        sc = rep[i] @ block.T + q[i] @ block.T
        if int(sc.argmax()) == it["gold"]:
            correct += 1
    return correct / len(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=fr"{PROJECT_ROOT}\data\synth_v3")
    ap.add_argument("--root", default=fr"{PROJECT_ROOT}\output\mem")
    ap.add_argument("--tags", nargs="+", default=["A-B6-carry", "A-B6-base"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--limit", type=int, default=0, help="0 = all test items")
    args = ap.parse_args()

    items = G.load_items(os.path.join(args.data, "test.jsonl"))
    if args.limit:
        items = items[: args.limit]
    chance = sum(1.0 / len(it["candidates"]) for it in items) / len(items)
    print(f"[fastA] items={len(items)} chance={chance:.3f}", flush=True)

    chunk_a = [G.split_chunks(it)[0] for it in items]
    concat_texts = [" ".join(G.split_chunks(it)) for it in items]

    results = {}
    for tag in args.tags:
        rows = {}
        for sd in args.seeds:
            arm = os.path.join(args.root, f"{tag}-s{sd}", "final")
            if not os.path.exists(arm):
                print(f"  MISSING {arm}")
                continue
            model = JaumeEmbedding.from_pretrained(arm)
            model.model.eval()

            concat_rep = enc(model, concat_texts)
            reset_rep = walk(model, chunk_a, pass_state=False)
            carried_rep = walk(model, chunk_a, pass_state=True)
            g = torch.Generator().manual_seed(0)
            shuffled_rep = carried_rep[torch.randperm(carried_rep.size(0),
                                                      generator=g)]

            r = {
                "concat": score_repr(concat_rep, items, model),
                "reset": score_repr(reset_rep, items, model),
                "carried": score_repr(carried_rep, items, model),
                "shuffled": score_repr(shuffled_rep, items, model),
            }
            rows[sd] = r
            print(f"  {tag} s{sd}: concat={r['concat']:.3f} reset={r['reset']:.3f} "
                  f"carried={r['carried']:.3f} shuffled={r['shuffled']:.3f}",
                  flush=True)
            del model
            empty_cache()
        results[tag] = rows

    print("\n" + "=" * 88)
    print(f"{'tag':13s} {'cond':9s} {'mean':>7s} {'sd':>6s} {'min':>6s} {'max':>6s}  per-seed")
    print("=" * 88)
    for tag, rows in results.items():
        for cond in ("concat", "reset", "carried", "shuffled"):
            vals = [rows[s][cond] for s in sorted(rows)]
            if not vals:
                continue
            sdv = statistics.stdev(vals) if len(vals) > 1 else 0.0
            print(f"{tag:13s} {cond:9s} {statistics.mean(vals):7.3f} {sdv:6.3f} "
                  f"{min(vals):6.3f} {max(vals):6.3f}  {[round(v,3) for v in vals]}")
        print("-" * 88)

    print("\nVERDICT (means across seeds):")
    for tag, rows in results.items():
        if not rows:
            continue
        c = statistics.mean([rows[s]["carried"] for s in rows])
        r = statistics.mean([rows[s]["reset"] for s in rows])
        sh = statistics.mean([rows[s]["shuffled"] for s in rows])
        co = statistics.mean([rows[s]["concat"] for s in rows])
        helps = (c > r) and (c > sh)
        print(f"  {tag:13s} carried={c:.3f} reset={r:.3f} shuffled={sh:.3f} "
              f"concat={co:.3f} -> {'MEMORY HELPS' if helps else 'no benefit'}")
        print(f"       carried-reset={c-r:+.3f} carried-shuffled={c-sh:+.3f}")

    out = os.path.join(args.data, "experiment_a_eval.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"chance": chance, "n": len(items),
                   "results": {k: {str(s): v for s, v in r.items()}
                               for k, r in results.items()}}, f, indent=2)
    print(f"\n[fastA] wrote {out}")


if __name__ == "__main__":
    main()
