"""Train and score the Jaume VLM bridge on the basic-visual-competence suite.

Reports per-family accuracy so we get a competence *profile* rather than one
number: which abilities the bridge already has, and which it does not.
"""
import argparse
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
REPO = os.path.join(PROJECT_ROOT, "code", "repo")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jaume.device import resolve_device, seed_all, describe
from vision_suite import build

FAMILIES = ["attr", "count", "spatial", "compare", "scene", "multi_attr"]


class Tower(nn.Module):
    def __init__(self, d, n_tokens=48):
        super().__init__()
        self.n_tokens = n_tokens
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(128, d, 3, stride=2, padding=1), nn.GELU(),
        )
        self.proj = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, x):
        h = self.conv(x)
        t = h.flatten(2).transpose(1, 2)
        t = F.adaptive_avg_pool1d(t.transpose(1, 2), self.n_tokens).transpose(1, 2)
        return self.proj(t)


def to_tensor(img):
    import torchvision.transforms.functional as TF
    return TF.to_tensor(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-train", type=int, default=9000)
    ap.add_argument("--n-val", type=int, default=900)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--unfreeze-layers", type=int, default=4)
    ap.add_argument("--n-tokens", type=int, default=48)
    ap.add_argument("--res", type=int, default=96)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)
    print("[vis] device=%s (%s)" % (device, describe(device)), flush=True)

    from jaume.modeling.config import JaumeConfig
    from jaume.modeling.embedding import JaumeEmbeddingModel
    from jaume.modeling.checkpoint import load_state_dict
    from jaume.data.dataset import load_tokenizer

    cfg = JaumeConfig.load(os.path.join(args.model, "config.json"))
    cfg.num_layers = args.depth
    tok = load_tokenizer(os.path.join(args.model, "tokenizer"))
    tok.add_tokens(["[MASK]"])
    cfg.vocab_size = len(tok)
    cfg.causal = True

    enc = JaumeEmbeddingModel(cfg).to(device)
    pre = load_state_dict(os.path.join(args.model, "pytorch_model.bin"))
    own = enc.state_dict()
    merged, up = {}, 0
    for k in own:
        if k in pre and pre[k].shape == own[k].shape:
            merged[k] = pre[k]
        elif k.startswith("backbone.layers."):
            rest = k.split(".", 3)[3]
            s = "backbone.layers.15.%s" % rest
            if s in pre:
                merged[k] = pre[s]; up += 1
            else:
                merged[k] = own[k]
        else:
            merged[k] = own[k]
    enc.load_state_dict(merged, strict=True)
    print("[vis] warm start upcycled=%d depth=%d" % (up, cfg.num_layers),
          flush=True)

    for p in enc.parameters():
        p.requires_grad_(False)
    if args.unfreeze_layers > 0:
        for layer in list(enc.backbone.layers)[-args.unfreeze_layers:]:
            for p in layer.parameters():
                p.requires_grad_(True)
        for p in enc.backbone.norm.parameters():
            p.requires_grad_(True)

    d = cfg.hidden_size
    tower = Tower(d, n_tokens=args.n_tokens).to(device)
    head = nn.Linear(d, cfg.vocab_size).to(device)
    print("[vis] tower=%.2fM" % (sum(p.numel() for p in tower.parameters()) / 1e6),
          flush=True)

    t0 = time.time()
    tr = build(args.n_train, seed=11)
    va = build(args.n_val, seed=777)
    print("[vis] data %d/%d in %.0fs" % (len(tr), len(va), time.time() - t0),
          flush=True)

    # answer inventory + option sets per (family, which)
    ans_set = sorted({a for _, _, _, a, _, _ in tr + va})
    print("[vis] distinct answers=%d" % len(ans_set), flush=True)
    q_tok = {}
    for _, fam, which, _, _, q in tr + va:
        if q not in q_tok:
            q_tok[q] = torch.tensor(
                tok(q, add_special_tokens=False)["input_ids"][:24],
                dtype=torch.long, device=device)
    a_tok = {}
    for a in ans_set:
        a_tok[a] = torch.tensor(
            tok(" " + a, add_special_tokens=False)["input_ids"],
            dtype=torch.long, device=device)

    params = (list(tower.parameters()) + list(head.parameters())
              + [p for p in enc.parameters() if p.requires_grad])
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    def step(batch):
        imgs = torch.stack([to_tensor(b[0]) for b in batch]).to(device)
        if args.res != 96:
            imgs = torch.nn.functional.interpolate(
                imgs, size=(args.res, args.res), mode="bilinear",
                align_corners=False)
        qs = [b[5] for b in batch]
        ans = [b[3] for b in batch]
        v = tower(imgs)
        ql = max(len(q_tok[q]) for q in qs)
        al = max(len(a_tok[a]) for a in ans)
        L = ql + al
        ids = torch.zeros(len(batch), L, dtype=torch.long, device=device)
        qm = torch.zeros(len(batch), L, dtype=torch.long, device=device)
        am = torch.zeros(len(batch), L, dtype=torch.long, device=device)
        for i, q in enumerate(qs):
            n = len(q_tok[q]); ids[i, :n] = q_tok[q]; qm[i, :n] = 1
        for i, a in enumerate(ans):
            n = len(a_tok[a]); ids[i, ql:ql + n] = a_tok[a]; am[i, ql:ql + n] = 1
        emb = enc.backbone.embed_tokens(ids)
        x = torch.cat([v, emb], 1)
        mask = torch.cat([torch.ones(v.shape[0], v.shape[1], dtype=torch.long,
                                     device=device), qm], 1)
        h = x
        for layer in enc.backbone.layers:
            h, _aux = layer(h, mask)
        h = enc.backbone.norm(h)
        # position (nv + ql - 1) predicts the first answer token; we need
        # exactly  such positions, not everything to the end of the row.
        nv = v.shape[1]
        logits = head(h[:, nv + ql - 1:nv + ql - 1 + al, :].float())
        tgt = ids[:, ql:]
        mm = am[:, ql:]
        loss = F.cross_entropy(
            logits.reshape(-1, cfg.vocab_size)[mm.reshape(-1).bool()],
            tgt.reshape(-1)[mm.reshape(-1).bool()])
        pred = logits.argmax(-1)
        ok = []
        for i in range(len(batch)):
            n = len(a_tok[ans[i]])
            ok.append(pred[i, :n].tolist() == a_tok[ans[i]].tolist())
        return loss, ok

    def evaluate(items):
        tower.eval(); head.eval(); enc.eval()
        per = {f: [0, 0] for f in FAMILIES}
        with torch.no_grad():
            for i in range(0, len(items) - 8 + 1, 8):
                b = items[i:i + 8]
                _, ok = step(b)
                for j, o in enumerate(ok):
                    fam = b[j][1]
                    per[fam][1] += 1
                    per[fam][0] += int(o)
        return per

    for ep in range(args.epochs):
        tower.train(); head.train(); enc.train()
        perm = list(range(len(tr)))
        random.Random(ep).shuffle(perm)
        tot = nb = 0
        t0 = time.time()
        for i in range(0, len(tr) - args.batch_size + 1, args.batch_size):
            batch = [tr[j] for j in perm[i:i + args.batch_size]]
            loss, _ = step(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot += float(loss); nb += 1
        per = evaluate(va)
        parts = []
        gm = 0.0
        for f in FAMILIES:
            c, n = per[f]
            a = c / max(n, 1)
            gm += a
            parts.append("%s %.3f" % (f, a))
        print("[vis] ep %d/%d loss %.3f | %s | mean %.3f (%.0fs)" % (
            ep + 1, args.epochs, tot / max(nb, 1), "  ".join(parts),
            gm / len(FAMILIES), time.time() - t0), flush=True)

    per = evaluate(va)
    print("=== FINAL PROFILE ===", flush=True)
    for f in FAMILIES:
        c, n = per[f]
        print("  %-11s %.4f  (%d/%d)" % (f, c / max(n, 1), c, n), flush=True)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        torch.save({"tower": tower.state_dict(), "head": head.state_dict()},
                   os.path.join(args.out, "vision_head.pt"))
        print("[vis] saved -> %s" % args.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
