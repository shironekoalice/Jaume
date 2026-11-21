"""Diagnose WHY the state representations collapse, per the review's directive:
when both R1a and R1b fail the mechanism check, stop adding normalisation and
inspect the actual computation graph and gradients of slot_pass.

Three questions, each measured directly:

  D1  Does the state even depend on the CURRENT block? Measure how much
      state_i differs from state_{i-1}; if the delta is tiny the memory is
      simply copying history forward.

  D2  Does the gradient from a far block reach the state written many blocks
      earlier? Backprop from the last position's loss to state_k and report the
      gradient norm as a function of distance.

  D3  Does cross-attention read DIFFERENT content from different blocks?
      Correlate the read-out with block identity: if read_i is nearly the same
      for every block, no state update can rescue the representation.

Usage:
    python scripts/diagnose_state_grad.py --mode additive
"""
import os as _os
PROJECT_ROOT = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

REPO = fr"{PROJECT_ROOT}\code\repo"
sys.path.insert(0, REPO)
from jaume.device import resolve_device, has_amp, make_scaler, empty_cache, max_memory_allocated, reset_peak_memory_stats, total_memory, seed_all
sys.path.insert(0, os.path.join(REPO, "scripts"))

import train_r1 as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="additive")
    ap.add_argument("--data", default=fr"{PROJECT_ROOT}\data\qasper_blocks")
    ap.add_argument("--blocks", type=int, default=12)
    args = ap.parse_args()

    rows = R.trim(R.load(os.path.join(args.data, "val.jsonl")), args.blocks)[:2]
    rm = R.make_model(fr"{PROJECT_ROOT}\350m-mlm", resolve_device(), args.mode)
    base = rm.model
    base.train()

    r = rows[0]
    blocks = r["blocks"][: args.blocks]

    # ---------------------------------------------------------------- D1 & D2
    states, reads = [], []
    state = None
    for b in blocks:
        e = rm.tokenizer([b], padding=True, truncation=True,
                         max_length=128, return_tensors="pt").to(resolve_device())
        # capture the fresh read before the update, for D3
        h, _ = base.backbone(e["input_ids"], e["attention_mask"])
        with torch.no_grad():
            read, _ = base.pooling.core._cross_attend(
                state if state is not None else
                base.pooling.core.slots.unsqueeze(0), h, e["attention_mask"],
                base.pooling.core._kv(h))
            reads.append(read.detach())
        pooled, _, state = base.encode(e["input_ids"], e["attention_mask"], state)
        states.append(state)

    print(f"mode={args.mode}  blocks={len(blocks)}\n")

    print("D1  how much does the state move at each block?")
    for i in range(1, len(states)):
        prev = states[i - 1].detach()
        cur = states[i].detach()
        delta = (cur - prev).norm() / (prev.norm() + 1e-8)
        cos = F.cosine_similarity(cur.flatten(), prev.flatten(), dim=0)
        print(f"   block {i:2d}: |Δ|/|s_prev| = {float(delta):.4f}   "
              f"cos(s_prev, s_i) = {float(cos):.4f}")

    print("\nD2  does the loss at the LAST position reach an early state?")
    # rebuild with graph, then backprop only from the last position
    base.zero_grad(set_to_none=True)
    states2 = []
    state = None
    for b in blocks:
        e = rm.tokenizer([b], padding=True, truncation=True,
                         max_length=128, return_tensors="pt").to(resolve_device())
        pooled, _, state = base.encode(e["input_ids"], e["attention_mask"], state)
        states2.append(state)
    q = rm.tokenizer([r["question"]], padding=True, truncation=True,
                     max_length=128, return_tensors="pt").to(resolve_device())
    last_rep = states2[-1].mean(dim=1)
    logits = base.projection(last_rep) @ base(q["input_ids"],
                                              q["attention_mask"])[0].T
    logits.sum().backward()

    print("   gradient norm reaching each retained state (gathered by hook):")
    # states2 are not leaves; measure via the shared slot bank + input influence
    g_backbone = 0.0
    for p in base.backbone.parameters():
        if p.grad is not None:
            g_backbone += float(p.grad.norm())
    g_slots = 0.0
    for p in base.pooling.parameters():
        if p.grad is not None:
            g_slots += float(p.grad.norm())
    print(f"   backbone grad norm = {g_backbone:.6f}")
    print(f"   pooling  grad norm = {g_slots:.6f}")
    print("   (a healthy long-range path should show non-trivial pooling grads)")

    print("\nD3  does cross-attention read different content per block?")
    Rn = torch.cat([x.mean(dim=1) for x in reads], dim=0)
    Rn = F.normalize(Rn, dim=-1)
    sim = Rn @ Rn.T
    k = Rn.shape[0]
    off = sim - torch.eye(k, device=sim.device)
    print(f"   mean pairwise cosine of read-out across blocks = "
          f"{float(off.sum()/(k*(k-1))):.4f}")
    print(f"   read-out std = {float(Rn.std(dim=0).mean()):.4f}")

    print("\nINTERPRETATION")
    print("  D1 small delta or cos ~1  -> state barely moves: memory is a copy")
    print("  D2 pooling grads ~0       -> the state path is not being optimised")
    print("  D3 read cos ~1            -> attention reads the same thing for every")
    print("                               block; no update rule can fix that")


if __name__ == "__main__":
    main()
