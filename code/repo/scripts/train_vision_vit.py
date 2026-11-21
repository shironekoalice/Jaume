"""Jaume VLM bridge with a real pretrained ViT visual encoder.

Design (LLaVA-style, the mainstream recipe):
  * ViT encodes the image into patch tokens (196 for 224px/patch16)
  * a projector maps them into Jaume's hidden space
  * a frozen (or top-k unfrozen) Jaume backbone consumes [vis | question]
  * a linear head predicts the answer tokens

Why this variant exists
-----------------------
The CNN tower (1.21M) saturated: raising its token count from 48 to 192 moved
the suite mean from 0.25 to 0.71, which says the *information* was there but
the encoder could not extract more of it. A pretrained ViT supplies real
visual features instead of learning them from 9k synthetic images.
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


def to_tensor_224(img):
    import torchvision.transforms.functional as TF
    x = TF.to_tensor(img)
    return TF.resize(x, [224, 224], antialias=True)


class ViTBridge(nn.Module):
    """timm ViT -> patch tokens -> linear projector to Jaume width."""

    def __init__(self, d_out, arch="vit_small_patch16_224", pretrained=False,
                 freeze=True, unfreeze_blocks=0, weights_path=None):
        super().__init__()
        import timm
        self.vit = timm.create_model(arch, pretrained=False, num_classes=0)
        if weights_path:
            # timm's own pretrained download needs HF, which is unreachable
            # from this host; we fetched the same checkpoint from ModelScope.
            from safetensors.torch import load_file
            sd = load_file(weights_path)
            own = self.vit.state_dict()
            hit, miss = {}, []
            for k, v in sd.items():
                if k in own and own[k].shape == v.shape:
                    hit[k] = v
                else:
                    miss.append(k)
            own.update(hit)
            self.vit.load_state_dict(own)
            print("[vit] loaded %d/%d tensors from %s (skipped %d)" % (
                len(hit), len(own), weights_path, len(miss)), flush=True)
        elif pretrained:
            raise SystemExit("use --weights instead of --pretrained on this host")
        nf = self.vit.num_features
        # drop the CLS token: keep the spatial patch grid
        self.n_prefix = getattr(self.vit, "num_prefix_tokens", 1)
        self.proj = nn.Sequential(
            nn.LayerNorm(nf),
            nn.Linear(nf, d_out),
            nn.GELU(),
            nn.Linear(d_out, d_out),
        )
        if freeze:
            for p in self.vit.parameters():
                p.requires_grad_(False)
            if unfreeze_blocks > 0:
                blocks = list(self.vit.blocks)
                for blk in blocks[-unfreeze_blocks:]:
                    for p in blk.parameters():
                        p.requires_grad_(True)
                if hasattr(self.vit, "norm"):
                    for p in self.vit.norm.parameters():
                        p.requires_grad_(True)

    def forward(self, x):
        t = self.vit.forward_features(x)
        if t.dim() == 3:
            t = t[:, self.n_prefix:, :]
        return self.proj(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-train", type=int, default=9000)
    ap.add_argument("--n-val", type=int, default=1200)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--vit-lr", type=float, default=1e-4)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--unfreeze-layers", type=int, default=4)
    ap.add_argument("--arch", default="vit_small_patch16_224")
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--vit-unfreeze-blocks", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="vit")
    args = ap.parse_args()

    device = resolve_device(args.device)
    print("[vit] device=%s (%s)" % (device, describe(device)), flush=True)

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
    for p in enc.parameters():
        p.requires_grad_(False)
    if args.unfreeze_layers > 0:
        for layer in list(enc.backbone.layers)[-args.unfreeze_layers:]:
            for p in layer.parameters():
                p.requires_grad_(True)
        for p in enc.backbone.norm.parameters():
            p.requires_grad_(True)
    print("[vit] warm start up=%d depth=%d" % (up, cfg.num_layers), flush=True)

    d = cfg.hidden_size
    bridge = ViTBridge(d, args.arch,
                       pretrained=args.pretrained, freeze=True,
                       unfreeze_blocks=args.vit_unfreeze_blocks,
                       weights_path=args.weights).to(device)
    head = nn.Linear(d, cfg.vocab_size).to(device)
    n_tr = sum(p.numel() for p in bridge.parameters() if p.requires_grad)
    print("[vit] arch=%s pretrained=%s vit_trainable=%.2fM" % (
        args.arch, args.pretrained, n_tr / 1e6), flush=True)

    t0 = time.time()
    tr = build(args.n_train, seed=11)
    va = build(args.n_val, seed=777)
    print("[vit] data %d/%d %.0fs" % (len(tr), len(va), time.time() - t0),
          flush=True)

    q_tok, a_tok = {}, {}
    for _, fam, which, ans, _, q in tr + va:
        if q not in q_tok:
            q_tok[q] = torch.tensor(
                tok(q, add_special_tokens=False)["input_ids"][:24],
                dtype=torch.long, device=device)
        if ans not in a_tok:
            a_tok[ans] = torch.tensor(
                tok(" " + ans, add_special_tokens=False)["input_ids"],
                dtype=torch.long, device=device)

    vit_params = [p for p in bridge.vit.parameters() if p.requires_grad]
    proj_params = list(bridge.proj.parameters())
    enc_params = [p for p in enc.parameters() if p.requires_grad]
    groups = [{"params": proj_params + list(head.parameters()), "lr": args.lr}]
    if enc_params:
        groups.append({"params": enc_params, "lr": args.lr})
    if vit_params:
        groups.append({"params": vit_params, "lr": args.vit_lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    all_params = proj_params + list(head.parameters()) + enc_params + vit_params

    def step(batch):
        imgs = torch.stack([to_tensor_224(b[0]) for b in batch]).to(device)
        qs = [b[5] for b in batch]
        ans = [b[3] for b in batch]
        v = bridge(imgs)
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
        nv = v.shape[1]
        logits = head(h[:, nv + ql - 1:nv + ql - 1 + al, :].float())
        tgt = ids[:, ql:]
        mm = am[:, ql:]
        loss = F.cross_entropy(
            logits.reshape(-1, cfg.vocab_size)[mm.reshape(-1).bool()],
            tgt.reshape(-1)[mm.reshape(-1).bool()])
        pred = logits.argmax(-1)
        ok = [pred[i, :len(a_tok[ans[i]])].tolist() == a_tok[ans[i]].tolist()
              for i in range(len(batch))]
        return loss, ok

    def evaluate():
        bridge.eval(); head.eval(); enc.eval()
        per = {f: [0, 0] for f in FAMILIES}
        with torch.no_grad():
            for i in range(0, len(va) - 8 + 1, 8):
                b = va[i:i + 8]
                _, ok = step(b)
                for j, o in enumerate(ok):
                    fam = b[j][1]
                    per[fam][1] += 1
                    per[fam][0] += int(o)
        return per

    tag = args.tag
    for ep in range(args.epochs):
        bridge.train(); head.train(); enc.train()
        perm = list(range(len(tr)))
        random.Random(ep).shuffle(perm)
        tot = nb = 0
        t0 = time.time()
        for i in range(0, len(tr) - args.batch_size + 1, args.batch_size):
            batch = [tr[j] for j in perm[i:i + args.batch_size]]
            loss, _ = step(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            opt.step()
            tot += float(loss); nb += 1
        per = evaluate()
        parts = ["%s %.3f" % (f, per[f][0] / max(per[f][1], 1)) for f in FAMILIES]
        mean = sum(per[f][0] / max(per[f][1], 1) for f in FAMILIES) / len(FAMILIES)
        print("[vit] ep %d/%d loss %.3f | %s | mean %.3f (%.0fs)" % (
            ep + 1, args.epochs, tot / max(nb, 1), "  ".join(parts), mean,
            time.time() - t0), flush=True)

    per = evaluate()
    print("=== FINAL %s ===" % tag, flush=True)
    for f in FAMILIES:
        c, n = per[f]
        print("  %-11s %.4f  (%d/%d)" % (f, c / max(n, 1), c, n), flush=True)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        torch.save({"vit": bridge.state_dict(), "head": head.state_dict()},
                   os.path.join(args.out, "vision_vit.pt"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
