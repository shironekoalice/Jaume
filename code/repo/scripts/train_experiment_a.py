"""Experiment A training driver.

Why this exists instead of reusing train_memory.py directly: experiment A changes
the OUTPUT PATH (`mean(carry)` instead of `mean(new_slots)`), so A must be
RETRAINED -- evaluating the old checkpoint once would not be a fair comparison
(review point 2). `train_memory.py` already supports `--memory-output`, but it
feeds the pair dataset, whose query side cannot express "chunk B then chunk A
through the recurrent pooler".

This script therefore builds the two-chunk sequence explicitly:

    step 1: encode chunk B with state=None      -> s_B
    step 2: encode chunk A with state=s_B       -> output, s_AB
    loss  : output must match the answer name

so the model is trained END-TO-END through the recurrent path it will be
evaluated on, and experiment A's changed output path is actually in the gradient.

  baseline  : output = pool(new_slots)  (original JM-R)
  exp A     : output = pool(carry)      (UPDATED state reaches the prediction)

Everything else identical: data, init, schedule, seed, evaluation.

Usage:
    python scripts/train_experiment_a.py --mode carry --seed 0 --epochs 2
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
    ap.add_argument("--mode", choices=["carry", "new_slots"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data", default=fr"{PROJECT_ROOT}\data\synth_v3")
    ap.add_argument("--init-from", default=fr"{PROJECT_ROOT}\350m-mlm")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-length", type=int, default=256)
    args = ap.parse_args()

    dev = resolve_device()
    torch.manual_seed(args.seed)

    items = G.load_items(os.path.join(args.data, "train.jsonl"))
    test_items = G.load_items(os.path.join(args.data, "test.jsonl"))
    chance = sum(1.0 / len(it["candidates"]) for it in test_items) / len(test_items)
    print(f"[A] mode={args.mode} seed={args.seed} train={len(items)} "
          f"test={len(test_items)} chance={chance:.3f}")

    model = JaumeEmbedding.from_pretrained(args.init_from)
    base = model.model
    # install the recurrent pooler with the chosen output path
    from jaume.modeling.memory import build_memory_pooling
    base.config.pooling = "recurrent_memory_gate"
    base.config.memory_output = args.mode
    base.config.num_slots = 8
    base.config.slot_heads = 8
    base.pooling = build_memory_pooling(base.config)
    base.to(dev)

    # re-load the trained gate/slot params if the init checkpoint has them
    src = os.path.join(args.init_from, "pytorch_model.bin")
    if os.path.exists(src):
        sd = torch.load(src, map_location="cpu", weights_only=True)
        ms = base.state_dict()
        reused = fresh = 0
        for k, v in ms.items():
            s2 = sd.get(k)
            if s2 is not None and tuple(s2.shape) == tuple(v.shape):
                v.copy_(s2)
                reused += 1
            else:
                fresh += 1
        print(f"[A] tensors reused={reused} fresh={fresh}")

    base.train()
    opt = torch.optim.AdamW(base.parameters(), lr=args.lr, weight_decay=0.01)
    lossf = MultipleNegativesRankingLoss()
    n = len(items)
    t0 = time.time()
    for ep in range(args.epochs):
        perm = torch.randperm(n)
        for s in range(0, n, args.bs):
            idx = perm[s : s + args.bs].tolist()
            if len(idx) < 2:
                continue
            sub = [items[i] for i in idx]
            bs = [G.split_chunks(it)[1] for it in sub]   # chunk B = instruction
            as_ = [G.split_chunks(it)[0] for it in sub]  # chunk A = facts
            tgt = [it["candidates"][it["gold"]] for it in sub]

            eb = model.tokenizer(bs, padding=True, truncation=True,
                                 max_length=args.max_length,
                                 return_tensors="pt").to(dev)
            ea = model.tokenizer(as_, padding=True, truncation=True,
                                 max_length=args.max_length,
                                 return_tensors="pt").to(dev)
            et = model.tokenizer(tgt, padding=True, truncation=True,
                                 max_length=64, return_tensors="pt").to(dev)

            with torch.autocast(dev, dtype=torch.bfloat16, enabled=has_amp(dev)):
                # step 1: the instruction, fresh state
                _, _, s_b = base.encode(eb["input_ids"], eb["attention_mask"], None)
                # step 2: the facts, carrying the instruction state ->
                # this is the output the task is judged on.
                # NOTE: encode() returns the PRE-projection pooled vector, and
                # the loss must be taken after `projection` + L2 norm, exactly
                # like the inference path. Using the raw pooled vector made the
                # loss sit at ln(batch)=3.466 and never move (the model cannot
                # learn a normalised contrastive objective from an unnormalised
                # 512-d vector).
                out, _, _ = base.encode(ea["input_ids"], ea["attention_mask"], s_b)
                out = torch.nn.functional.normalize(base.projection(out), dim=-1)
                t_emb = base(et["input_ids"], et["attention_mask"])[0]
                loss = lossf(out, t_emb)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        print(f"  [A-{args.mode}-s{args.seed}] ep{ep} loss {float(loss):.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    base.eval()
    os.makedirs(args.out, exist_ok=True)
    base.config.memory_output = args.mode
    base.save_pretrained(args.out) if hasattr(base, "save_pretrained") else None
    torch.save(base.state_dict(), os.path.join(args.out, "pytorch_model.bin"))
    model.tokenizer.save_pretrained(args.out)

    # quick readout on the test split, using the SAME scorer as the main eval
    import eval_experiment_a_fast as F
    model.model.eval()
    chunk_a = [G.split_chunks(it)[0] for it in test_items]
    reps = F.walk(model, chunk_a, pass_state=True)
    acc = F.score_repr(reps, test_items, model)
    print(f"[A-{args.mode}-s{args.seed}] carried-path test acc={acc:.3f} "
          f"(chance {chance:.3f})")
    with open(os.path.join(args.out, "train_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump({"mode": args.mode, "seed": args.seed,
                   "epochs": args.epochs, "steps": (n // args.bs) * args.epochs,
                   "seconds": round(time.time() - t0, 1),
                   "final_loss": float(loss), "carried_acc": acc,
                   "chance": chance}, f, indent=2)


if __name__ == "__main__":
    main()
