"""Diagnostics for the JM-S negative result.

Measures whether the slot bank actually differentiates:
  1. slot diversity   mean pairwise cosine between slots (0 = collapsed)
  2. slot usage       entropy of the cross-attention mass per slot
                      (uniform = all slots used; peaked = one slot dominates)
  3. attention spread entropy of the token distribution each slot attends to

Usage:
    python scripts/diagnose_slots.py --model out/B3-latent-s3/final
"""
import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jaume.inference.encode import JaumeEmbedding

TEXTS = [
    "A man is playing a guitar on stage in front of a large audience.",
    "The company reported a 12 percent increase in quarterly revenue.",
    "Researchers trained a transformer model on a multilingual corpus.",
    "The cat sat on the windowsill and watched the rain for hours.",
    "A new vaccine was approved for use in several European countries.",
    "The engine overheated after the cooling system failed during the race.",
    "Students protested the decision to close the university library.",
    "The recipe requires flour, sugar, eggs and about twenty minutes.",
]


def entropy(p):
    p = p[p > 1e-12]
    return float(-(p * torch.log(p)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model = JaumeEmbedding.from_pretrained(args.model)
    pooler = model.model.pooling
    arm = model.model.config.pooling
    print(f"[diag] arm={arm} device={model.device}")

    if not hasattr(pooler, "slots"):
        print("[diag] no slot bank in this arm; nothing to diagnose")
        return

    slots = pooler.slots.detach().float()
    n = slots.shape[0]
    cos = torch.nn.functional.cosine_similarity(slots.unsqueeze(1), slots.unsqueeze(0), dim=-1)
    iu = torch.triu_indices(n, n, offset=1)
    pair = cos[iu[0], iu[1]]
    print(f"[diag] learned slots: n={n}  norm={slots.norm(dim=-1).mean():.4f}")
    print(f"[diag] slot pairwise cosine: mean {pair.mean():.4f}  min {pair.min():.4f}  max {pair.max():.4f}")
    print(f"[diag]   (mean ~0 => differentiated; ~1 => collapsed to one direction)")

    # cross-attention behaviour over real text
    enc = model.tokenizer(TEXTS, padding=True, truncation=True, max_length=128,
                          return_tensors="pt").to(model.device)
    with torch.no_grad():
        hidden, _ = model.model.backbone(enc["input_ids"], enc["attention_mask"])
        B, N, _ = (hidden.size(0), pooler.slots.size(0), hidden.size(-1))
        H = pooler.num_heads
        hd = pooler.hidden_size // H
        q = pooler.q_proj(pooler.norm_slots(pooler.slots.unsqueeze(0).expand(B, -1, -1)))
        q = q.view(B, N, H, hd).transpose(1, 2)
        hv = pooler.norm_hidden(hidden)
        T = hv.size(1)
        k = pooler.k_proj(hv).view(B, T, H, hd).transpose(1, 2)
        mask = (1.0 - enc["attention_mask"][:, None, None, :].to(q.dtype)) * torch.finfo(q.dtype).min
        attn = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(hd) + mask, dim=-1)
        attn = attn.mean(dim=1)  # average over heads -> [B, N, T]

    # NOTE: attention mass per slot is always 1 by softmax normalisation, so it
    # cannot measure usage. What matters is whether slots attend to DIFFERENT
    # tokens and produce DIFFERENT states; both are measured below.
    js = []
    for b in range(B):
        a = attn[b]                                   # [N, T]
        m = 0.5 * (a.unsqueeze(1) + a.unsqueeze(0))    # [N, N, T]
        def kl(p, q):
            return (p * (torch.log(p + 1e-12) - torch.log(q + 1e-12))).sum(-1)
        d = 0.5 * kl(a.unsqueeze(1), m) + 0.5 * kl(a.unsqueeze(0), m)
        iu = torch.triu_indices(n, n, offset=1)
        js.append(d[iu[0], iu[1]].mean())
    js_val = float(torch.tensor(js).mean())
    print(f"[diag] slot attention-distribution divergence (JS, 0=identical): {js_val:.4f}")
    print(f"[diag]   (JS ~0 => every slot reads the same tokens => no specialisation)")

    # do the resulting slot STATES differ from each other?
    with torch.no_grad():
        states = pooler.slot_pass(hidden, enc["attention_mask"])   # [B, N, D]
    scos = torch.nn.functional.cosine_similarity(
        states.unsqueeze(2), states.unsqueeze(1), dim=-1
    )  # [B, N, N]
    iu = torch.triu_indices(n, n, offset=1)
    spair = scos[:, iu[0], iu[1]]
    print(f"[diag] slot-state pairwise cosine: mean {spair.mean():.4f} "
          f"min {spair.min():.4f} max {spair.max():.4f}")

    # how much of the pooled vector does the mean over slots destroy?
    pooled = pooler.pool_slots(states)
    per_slot_norm = states.norm(dim=-1).mean()
    print(f"[diag] slot state norm {per_slot_norm:.4f}  pooled norm {pooled.norm(dim=-1).mean():.4f} "
          f"(ratio {pooled.norm(dim=-1).mean()/per_slot_norm:.3f})")
    print(f"[diag]   (ratio <<1 => slots cancel each other under mean pooling)")

    spread = torch.tensor([entropy(attn[b, s]) for b in range(B) for s in range(n)])
    low_spread = float(spread.reshape(B, n).min(dim=1).values.mean())
    print(f"[diag] token-spread entropy per slot: mean {spread.mean():.3f} "
          f"min {spread.min():.3f} max {spread.max():.3f}")
    print(f"[diag]   (low => slot locks onto a single token; high => diffuse over the sentence)")
    print(f"[diag] per-example min spread (most peaked slot): {low_spread:.3f}")

    out = {
        "model": args.model, "arm": arm, "num_slots": n,
        "slot_pairwise_cosine_mean": round(float(pair.mean()), 4),
        "token_spread_entropy_mean": round(float(spread.mean()), 4),
        "slot_attention_js_divergence": round(js_val, 4),
        "slot_state_pairwise_cosine_mean": round(float(spair.mean()), 4),
        "pooled_over_slotstate_norm_ratio": round(
            float(pooled.norm(dim=-1).mean() / per_slot_norm), 4
        ),
    }
    out_path = args.out or os.path.join(args.model, "slot_diagnostics.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(out, open(out_path, "w", encoding="utf-8"), indent=2)
    print(f"[diag] wrote {out_path}")


if __name__ == "__main__":
    main()
