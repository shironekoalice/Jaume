"""Jaume VLM bridge with Google SigLIP 2 as the visual encoder.

Why SigLIP 2 over an ImageNet ViT
--------------------------------
An ImageNet-supervised ViT is trained to name objects. SigLIP 2 is trained on
image-text pairs with a sigmoid contrastive objective, so its features are
already aligned to *language*. For a VLM bridge whose consumer is a text
backbone, that alignment is the property we actually want, and it is the
reason the CLIP family dominates real VLM stacks.

torchaudio note
---------------
This host has a broken torchaudio ABI (undefined symbol). transformers
imports it eagerly, so we install a stub module before the import. We do not
use audio anywhere; the stub is deliberate and safe here.
"""
import argparse
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- torchaudio stub (must precede any transformers import) ----------------
if "torchaudio" not in sys.modules:
    import importlib.machinery as _im
    import types

    def _stub(name):
        m = types.ModuleType(name)
        m.__version__ = "0.0.0-stub"
        # find_spec() is called by transformers' availability probe; the
        # module must advertise a real spec or it raises ValueError.
        m.__spec__ = _im.ModuleSpec(name, loader=None)
        m.__loader__ = None
        m.__path__ = []
        m.__getattr__ = lambda attr: None
        sys.modules[name] = m
        return m

    _stub("torchaudio")
    for _sub in ("torchaudio.transforms", "torchaudio.functional",
                 "torchaudio.models", "torchaudio.utils"):
        _stub(_sub)

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
REPO = os.path.join(PROJECT_ROOT, "code", "repo")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jaume.device import resolve_device, seed_all, describe
from vision_suite import build

FAMILIES = ["attr", "count", "spatial", "compare", "scene", "multi_attr"]


class SiglipBridge(nn.Module):
    """SigLIP 2 vision tower -> patch tokens -> projector to Jaume width."""

    def __init__(self, d_out, path, unfreeze_blocks=0):
        super().__init__()
        # The checkpoint is the full SigLIP2 (vision + text); loading the
        # vision class alone would mismatch on keys. Load the full model and
        # keep only the vision tower, which is what the bridge consumes.
        # The checkpoint ships a legacy  config with SigLIP2 weights,
        # so the SiglipModel class is the one that matches the key layout.
        from transformers import SiglipModel
        full = SiglipModel.from_pretrained(path, dtype=torch.float32)
        self.tower = full.vision_model
        hidden = self.tower.config.hidden_size
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, d_out),
            nn.GELU(),
            nn.Linear(d_out, d_out),
        )
        for p in self.tower.parameters():
            p.requires_grad_(False)
        if unfreeze_blocks > 0:
            blocks = self.tower.encoder.layers
            for blk in list(blocks)[-unfreeze_blocks:]:
                for p in blk.parameters():
                    p.requires_grad_(True)
            for name in ("post_layernorm", "head"):
                mod = getattr(self.tower, name, None)
                if mod is not None:
                    for p in mod.parameters():
                        p.requires_grad_(True)

    def forward(self, pixel_values):
        out = self.tower(pixel_values=pixel_values)
        # (B, n_patches, hidden)
        t = out.last_hidden_state
        return self.proj(t)


def preprocess(images):
    """SigLIP 2 preprocessing, done on tensors (no torchvision dependency)."""
    import torchvision.transforms.functional as TF
    xs = []
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    for img in images:
        x = TF.to_tensor(img)
        x = TF.resize(x, [224, 224], antialias=True)
        x = (x - mean) / std
        xs.append(x)
    return torch.stack(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--siglip-path", required=True)
    ap.add_argument("--n-train", type=int, default=9000)
    ap.add_argument("--n-val", type=int, default=1200)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--siglip-lr", type=float, default=2e-5)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--unfreeze-layers", type=int, default=4)
    ap.add_argument("--siglip-unfreeze-blocks", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="siglip2")
    args = ap.parse_args()

    device = resolve_device(args.device)
    print("[sig] device=%s (%s)" % (device, describe(device)), flush=True)

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
    print("[sig] jaume warm start up=%d depth=%d" % (up, cfg.num_layers),
          flush=True)

    d = cfg.hidden_size
    bridge = SiglipBridge(d, args.siglip_path,
                          unfreeze_blocks=args.siglip_unfreeze_blocks).to(device)
    head = nn.Linear(d, cfg.vocab_size).to(device)
    n_vis = sum(p.numel() for p in bridge.tower.parameters())
    n_vis_tr = sum(p.numel() for p in bridge.tower.parameters()
                   if p.requires_grad)
    print("[sig] siglip2 %.1fM total, %.1fM trainable" % (
        n_vis / 1e6, n_vis_tr / 1e6), flush=True)

    t0 = time.time()
    tr = build(args.n_train, seed=11)
    va = build(args.n_val, seed=777)
    print("[sig] data %d/%d %.0fs" % (len(tr), len(va), time.time() - t0),
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

    vis_tr_params = [p for p in bridge.tower.parameters() if p.requires_grad]
    proj_params = list(bridge.proj.parameters())
    enc_params = [p for p in enc.parameters() if p.requires_grad]
    groups = [{"params": proj_params + list(head.parameters()), "lr": args.lr}]
    if enc_params:
        groups.append({"params": enc_params, "lr": args.lr})
    if vis_tr_params:
        groups.append({"params": vis_tr_params, "lr": args.siglip_lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    all_params = proj_params + list(head.parameters()) + enc_params + vis_tr_params

    def step(batch):
        imgs = preprocess([b[0] for b in batch]).to(device)
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
        print("[sig] ep %d/%d loss %.3f | %s | mean %.3f (%.0fs)" % (
            ep + 1, args.epochs, tot / max(nb, 1), "  ".join(parts), mean,
            time.time() - t0), flush=True)

    per = evaluate()
    print("=== FINAL %s ===" % args.tag, flush=True)
    for f in FAMILIES:
        c, n = per[f]
        print("  %-11s %.4f  (%d/%d)" % (f, c / max(n, 1), c, n), flush=True)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        torch.save({"bridge": bridge.state_dict(), "head": head.state_dict()},
                   os.path.join(args.out, "vision_siglip.pt"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
