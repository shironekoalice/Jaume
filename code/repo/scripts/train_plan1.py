"""Plan 1 training: make the model USE the state, not just have one.

The defect in every previous training run was a mismatch of information flow:

    train : chunk B -> state -> chunk A -> answer     (A's facts are the CURRENT
                                                       input, so the state never
                                                       has to be read)
    test  : chunk A -> state -> answer                (A's facts must come FROM
                                                       the state, but training
                                                       never required that)

Plan 1 fixes this by making the information flow identical in training and
evaluation, per review condition A:

    s_A = f(chunk A, s_0)          facts are written into the state
    y   = g(chunk B, s_A)          ONLY chunk B's text + the state

Step 2's input contains no digits and no protocol names (verified by
scripts/check_plan1_data.py: C1 0/5000, C2 0/5000), so the answer cannot be
recovered without the state.

Architecture is FROZEN: same 8-slot recurrent pooler, no GRU additions, no slot
count changes, no read-out layer, no auxiliary loss (review scope).

Usage:
    python scripts/train_plan1.py --seed 0 --epochs 3 --out ...
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
import g3_memory as G  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data", default=fr"{PROJECT_ROOT}\data\synth_v3")
    ap.add_argument("--init-from", default=fr"{PROJECT_ROOT}\350m-mlm")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--memory-output", default="new_slots",
                    choices=["new_slots", "carry"],
                    help="experiment A: which slots feed the output. "
                        "new_slots = baseline (original JM-R); carry = the "
                        "UPDATED state reaches the prediction.")
    args = ap.parse_args()

    dev = resolve_device()
    torch.manual_seed(args.seed)

    train = G.load_items(os.path.join(args.data, "train.jsonl"))
    test = G.load_items(os.path.join(args.data, "test.jsonl"))
    chance = sum(1.0 / len(it["candidates"]) for it in test) / len(test)
    print(f"[P1] seed={args.seed} train={len(train)} test={len(test)} "
          f"chance={chance:.3f}")

    model = JaumeEmbedding.from_pretrained(args.init_from)
    base = model.model
    from jaume.modeling.memory import build_memory_pooling
    base.config.pooling = "recurrent_memory_gate"
    base.config.memory_output = args.memory_output
    base.config.num_slots = 8
    base.config.slot_heads = 8
    base.pooling = build_memory_pooling(base.config)
    base.to(dev)

    src = os.path.join(args.init_from, "pytorch_model.bin")
    if os.path.exists(src):
        sd = torch.load(src, map_location="cpu", weights_only=True)
        ms = base.state_dict()
        reused = sum(1 for k, v in ms.items()
                     if k in sd and tuple(sd[k].shape) == tuple(v.shape))
        for k, v in ms.items():
            if k in sd and tuple(sd[k].shape) == tuple(v.shape):
                v.copy_(sd[k])
        print(f"[P1] tensors reused={reused} fresh={len(ms)-reused}")

    base.train()
    opt = torch.optim.AdamW(base.parameters(), lr=args.lr, weight_decay=0.01)
    lossf = MultipleNegativesRankingLoss()
    n = len(train)
    t0 = time.time()

    for ep in range(args.epochs):
        perm = torch.randperm(n)
        run = 0.0
        nb = 0
        for s in range(0, n, args.bs):
            idx = perm[s : s + args.bs].tolist()
            if len(idx) < 2:
                continue
            sub = [train[i] for i in idx]
            a_txt = [G.split_chunks(it)[0] for it in sub]   # facts
            b_txt = [G.split_chunks(it)[1] for it in sub]   # instruction only
            tgt = [it["candidates"][it["gold"]] for it in sub]
            # The candidate LIST (1 gold + distractors) is part of the real
            # task, but in-batch negatives only ever show one answer per item,
            # so the model learned "is this my positive" rather than "which of
            # these candidates is right". Training measured 92% retrieval yet
            # scored 15.6% over candidates. Flatten gold + distractors so the
            # contrastive pool contains the actual decision the task requires.
            cand_pool = [it["candidates"] for it in sub]

            ea = model.tokenizer(a_txt, padding=True, truncation=True,
                                 max_length=args.max_length,
                                 return_tensors="pt").to(dev)
            eb = model.tokenizer(b_txt, padding=True, truncation=True,
                                 max_length=64, return_tensors="pt").to(dev)
            et = model.tokenizer(tgt, padding=True, truncation=True,
                                 max_length=64, return_tensors="pt").to(dev)
            k = len(cand_pool[0])
            flat_c = [c for cs in cand_pool for c in cs]
            ec = model.tokenizer(flat_c, padding=True, truncation=True,
                                 max_length=64, return_tensors="pt").to(dev)

            with torch.autocast(dev, dtype=torch.bfloat16, enabled=has_amp(dev)):
                # step 1: chunk A writes the facts into the state
                _, _, s_a = base.encode(ea["input_ids"], ea["attention_mask"], None)
                # step 2: chunk B text + the carried state -> prediction.
                # This is the SAME flow the evaluator uses under `carried`.
                out, _, _ = base.encode(eb["input_ids"], eb["attention_mask"], s_a)
                out = torch.nn.functional.normalize(base.projection(out), dim=-1)
                # contrastive pool = every candidate of every item in the batch,
                # so the objective matches the candidate-picking decision the
                # task actually scores
                c_emb = base(ec["input_ids"], ec["attention_mask"])[0]
                logits = out @ c_emb.T
                labels = torch.arange(len(sub), device=dev) * k + \
                    torch.tensor([it["gold"] for it in sub], device=dev)
                loss = torch.nn.functional.cross_entropy(
                    logits / 0.05, labels)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            run += float(loss)
            nb += 1
        print(f"  [P1-s{args.seed}] ep{ep} loss {run/max(1,nb):.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    base.eval()
    os.makedirs(args.out, exist_ok=True)
    torch.save(base.state_dict(), os.path.join(args.out, "pytorch_model.bin"))
    model.tokenizer.save_pretrained(args.out)
    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as f:
        cfg = base.config.to_dict() if hasattr(base.config, "to_dict") else {}
        json.dump(cfg, f, indent=2)

    with open(os.path.join(args.out, "train_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump({"seed": args.seed, "epochs": args.epochs,
                   "steps": (n // args.bs) * args.epochs,
                   "seconds": round(time.time() - t0, 1),
                   "final_loss": run / max(1, nb), "chance": chance}, f, indent=2)
    print(f"[P1] saved -> {args.out}")


if __name__ == "__main__":
    main()
