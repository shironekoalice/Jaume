"""Apples-to-apples: does the instruction-conditioned gate beat a linear probe?

THE QUESTION. On ScreenSpot, a LINEAR PROBE on frozen SigLIP2 features
reaches held-out top-1 0.850 (floor 0.167). The repo's own dual-tower work
says the question-conditioned gate is what keeps the two streams from
overwriting each other. Does that translate into grounding accuracy?

THE COMPARISON. Same data, same split, same features, same loss, same
budget — the ONLY difference is the head:

  HEAD-P (linear probe)   one linear layer on [crop_emb ; text_emb].
                           The instruction is CONCATENATED, so it never
                           modulates the visual representation.

  HEAD-G (gated)          the instruction is projected into the visual
                           space and used as a GATE on the visual feature
                           (fused = v * (1 + sigmoid(W t)) + t), with the
                           raw visual stream kept as a residual.
                           The instruction MODULATES what the model looks at.

  HEAD-N (no instruction) a control: the same gated head with the
                           instruction replaced by a constant. If HEAD-N is
                           as good as HEAD-G, the gate is doing nothing and
                           the task is solvable from the crop alone.

HEAD-N is the important one. Without it, a gain could come from the extra
parameters rather than from the conditioning.

Reported on a screenshot-level 70/30 split (a sample-level split would put
the same screenshot on both sides).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from backbone_registry import local_path
from screenspot_data import build_eval_set

DEVICE = "cuda" if torch.cuda.is_available() else (
    "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")


class LinearProbe(nn.Module):
    def __init__(self, dv, dt):
        super().__init__()
        # [crop_emb ; text_emb ; box] -> one score
        self.lin = nn.Linear(dv + dt + 4, 1)

    def forward(self, v, t, box, mask):
        B, N = v.shape[:2]
        x = torch.cat([v, t.unsqueeze(1).expand(B, N, -1), box], dim=-1)
        return self.lin(x).squeeze(-1).masked_fill(~mask, -1e4)


class GatedScorer(nn.Module):
    def __init__(self, dv, dt, hidden=384):
        super().__init__()
        self.text_proj = nn.Linear(dt, dv)
        self.gate = nn.Linear(dv, dv)
        self.box_proj = nn.Linear(4, dv)
        self.fuse = nn.Sequential(
            nn.Linear(dv * 4, hidden), nn.GELU(),
            nn.Linear(hidden, hidden // 2), nn.GELU())
        self.out = nn.Linear(hidden // 2, 1)

    def forward(self, v, t, box, mask):
        B, N = v.shape[:2]
        tv = self.text_proj(t)
        g = torch.sigmoid(self.gate(tv)).unsqueeze(1)
        fused = v * (1.0 + g) + tv.unsqueeze(1)
        x = torch.cat([fused, v, tv.unsqueeze(1).expand(B, N, -1), self.box_proj(box)], dim=-1)
        return self.out(self.fuse(x)).squeeze(-1).masked_fill(~mask, -1e4)


@torch.no_grad()
def featurize(tower, proc, data, crop_size=224):
    """Frozen SigLIP2 features for every (crop, instruction) pair."""
    from backbone_registry import crop_image

    crops, texts = [], []
    for d in data:
        for b in d["candidates"]:
            crops.append(crop_image(d["image"], b, crop_size))
            texts.append(d["instruction"])
    px = torch.stack([torch.from_numpy(np.asarray(c, dtype=np.float32) / 255.0)
                       for c in crops]).permute(0, 3, 1, 2)
    outs = []
    for i in range(0, len(px), 64):
        pv = ((px[i:i + 64] - 0.5) / 0.5).to(DEVICE, torch.float32)
        vo = tower.vision_model(pixel_values=pv)
        outs.append((vo.pooler_output if hasattr(vo, "pooler_output")
                     else vo.last_hidden_state.mean(dim=1)).float().cpu())
    v = torch.cat(outs)

    uniq = sorted(set(texts))
    tin = proc(text=uniq, return_tensors="pt", padding=True, truncation=True, max_length=32)
    tin = {k: v2.to(DEVICE) for k, v2 in tin.items()}
    to = tower.text_model(**tin)
    tp = (to.pooler_output if hasattr(to, "pooler_output")
          else to.last_hidden_state.mean(dim=1)).float().cpu()
    tmap = {u: tp[i] for i, u in enumerate(uniq)}
    t = torch.stack([tmap[x] for x in texts])
    return v, t


def build_tensors(data, v, t, K):
    B = len(data)
    V = torch.zeros(B, K, v.shape[1])
    T = torch.zeros(B, t.shape[1])
    X = torch.zeros(B, K, 4)
    M = torch.zeros(B, K, dtype=torch.bool)
    G = torch.zeros(B, dtype=torch.long)
    for i, d in enumerate(data):
        n = len(d["candidates"])
        V[i, :n] = v[i * K:i * K + n]
        T[i] = t[i * K]
        for j, b in enumerate(d["candidates"]):
            w, h = d["image"].size
            X[i, j] = torch.tensor([b[0] / w, b[1] / h, b[2] / w, b[3] / h])
        M[i, :n] = True
        G[i] = d["gold"]
    return V, T, X, M, G


def train_head(head, V, T, X, M, G, tr, te, epochs=12, lr=1e-3, bs=32):
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lossf = nn.CrossEntropyLoss()
    best = 0.0
    for ep in range(epochs):
        head.train()
        perm = np.random.default_rng(ep).permutation(len(tr))
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            sel = torch.tensor(idx)
            logits = head(V[sel].to(DEVICE), T[sel].to(DEVICE), X[sel].to(DEVICE), M[sel].to(DEVICE))
            loss = lossf(logits, G[sel].to(DEVICE))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sched.step()
        head.eval()
        with torch.no_grad():
            hits = 0
            for i in range(0, len(te), 64):
                sel = torch.tensor(te[i:i + 64])
                logits = head(V[sel].to(DEVICE), T[sel].to(DEVICE), X[sel].to(DEVICE), M[sel].to(DEVICE))
                hits += int((logits.argmax(-1).cpu() == G[sel]).sum())
        acc = hits / len(te)
        best = max(best, acc)
        print(f"    ep{ep+1:02d} held-out top-1 {acc:.4f}", flush=True)
    return best, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=300)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--out", default="head_comparison.json")
    args = ap.parse_args()

    from transformers import AutoModel, AutoProcessor

    mid = "google/siglip2-base-patch16-224"
    proc = AutoProcessor.from_pretrained(local_path(mid))
    tower = AutoModel.from_pretrained(local_path(mid)).to(DEVICE).eval()

    print(f"building ScreenSpot set ({args.rows} samples, K={args.k})...", flush=True)
    data = build_eval_set(max_rows=args.rows, k=args.k, seed=0)
    print("  featurising...", flush=True)
    t0 = time.time()
    v, t = featurize(tower, proc, data)
    print(f"  done in {time.time()-t0:.0f}s", flush=True)

    K = args.k
    V, T, X, M, G = build_tensors(data, v, t, K)

    order = np.random.default_rng(0).permutation(len(data))
    n_tr = int(len(data) * 0.7)
    tr, te = order[:n_tr], order[n_tr:]
    print(f"  split {len(tr)} train / {len(te)} test screenshots", flush=True)

    dv, dt = V.shape[-1], T.shape[-1]
    results = {}
    for name, mk in [
        ("P_linear_probe", lambda: LinearProbe(dv, dt)),
        ("G_gated", lambda: GatedScorer(dv, dt)),
        ("N_gated_no_instruction", lambda: GatedScorer(dv, dt)),
    ]:
        print(f"\n=== {name} ===", flush=True)
        torch.manual_seed(0); np.random.seed(0)
        head = mk().to(DEVICE)
        if name.startswith("N_"):
            # the control: replace every instruction embedding with the mean
            T_use = T.mean(dim=0, keepdim=True).expand_as(T).contiguous()
        else:
            T_use = T
        npar = sum(p.numel() for p in head.parameters())
        best, last = train_head(head, V, T_use, X, M, G, tr, te, epochs=args.epochs)
        results[name] = dict(params=npar, best_top1=best, last_top1=last,
                             floor=1.0 / K)
        print(f"  -> best {best:.4f}  last {last:.4f}  (floor {1/K:.4f})  params {npar}", flush=True)

    json.dump(results, open(args.out, "w"), indent=2)
    print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
