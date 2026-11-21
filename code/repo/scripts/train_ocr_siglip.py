"""OCR probe on the Jaume + SigLIP2 bridge, with free-form string answers.

Unlike the multiple-choice basic-vision suite, OCR answers are arbitrary
strings (an invoice number, a 4-char code, "box 3"). We therefore score:
  * exact match, and
  * per-family accuracy
and we keep the answer set finite at build time so the decode is a prefix
comparison rather than open-ended generation.
"""
import argparse
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

if "torchaudio" not in sys.modules:
    import importlib.machinery as _im
    import types

    def _stub(name):
        m = types.ModuleType(name)
        m.__version__ = "0.0.0-stub"
        m.__spec__ = _im.ModuleSpec(name, loader=None)
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

from jaume.device import resolve_device, describe
from ocr_suite import build, OCR_FAMILIES


def preprocess(images, size=320):
    import torchvision.transforms.functional as TF
    xs = []
    mean = torch.tensor([0.5] * 3).view(3, 1, 1)
    std = torch.tensor([0.5] * 3).view(3, 1, 1)
    for img in images:
        x = TF.to_tensor(img)
        x = TF.resize(x, [size, size], antialias=True)
        xs.append((x - mean) / std)
    return torch.stack(xs)


def _interpolate_pos_embed(tower, n_patches_target):
    """Resize SigLIP's learned position embedding to a new patch grid.

    SigLIP2-base is fixed at 224px / patch16 -> 196 positions. OCR benefits
    from a larger canvas, which changes the patch count and would otherwise
    crash inside the embedding addition. Bicubic interpolation of the learned
    positional grid is the standard fix (ViT-22B / SAM both do this).
    """
    import math
    import torch.nn.functional as F
    pe = tower.embeddings.position_embedding.weight
    old_n, dim = pe.shape
    old_g = int(round(math.sqrt(old_n)))
    new_g = int(round(math.sqrt(n_patches_target)))
    if old_g * old_g != old_n or new_g * new_g != n_patches_target:
        return False, old_n, n_patches_target
    if new_g == old_g:
        return True, old_n, n_patches_target
    grid = pe.reshape(1, old_g, old_g, dim).permute(0, 3, 1, 2)
    grid = F.interpolate(grid, size=(new_g, new_g), mode="bicubic",
                         align_corners=False)
    new_pe = grid.permute(0, 2, 3, 1).reshape(new_g * new_g, dim)
    with torch.no_grad():
        tower.embeddings.position_embedding = torch.nn.Embedding(
            new_g * new_g, dim)
        tower.embeddings.position_embedding.weight.copy_(new_pe)
    if hasattr(tower.embeddings, "position_ids"):
        tower.embeddings.register_buffer(
            "position_ids",
            torch.arange(new_g * new_g).unsqueeze(0), persistent=False)
    if hasattr(tower, "config"):
        tower.config.image_size = new_g * tower.config.patch_size
    return True, old_n, n_patches_target


class SiglipBridge(nn.Module):
    def __init__(self, d_out, path, unfreeze_blocks=0, img_size=224):
        super().__init__()
        from transformers import SiglipModel
        full = SiglipModel.from_pretrained(path, dtype=torch.float32)
        self.tower = full.vision_model
        psz = self.tower.config.patch_size
        n_target = (img_size // psz) ** 2
        ok, old_n, new_n = _interpolate_pos_embed(self.tower, n_target)
        print("[ocr] pos-embed %d -> %d patches (ok=%s)" % (old_n, new_n, ok),
              flush=True)
        hidden = self.tower.config.hidden_size
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, d_out), nn.GELU(),
            nn.Linear(d_out, d_out))
        for p in self.tower.parameters():
            p.requires_grad_(False)
        if unfreeze_blocks > 0:
            for blk in list(self.tower.encoder.layers)[-unfreeze_blocks:]:
                for p in blk.parameters():
                    p.requires_grad_(True)
            for nm in ("post_layernorm", "head"):
                mod = getattr(self.tower, nm, None)
                if mod is not None:
                    for p in mod.parameters():
                        p.requires_grad_(True)

    def forward(self, x):
        return self.proj(self.tower(pixel_values=x).last_hidden_state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--siglip-path", required=True)
    ap.add_argument("--n-train", type=int, default=9000)
    ap.add_argument("--n-val", type=int, default=900)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--siglip-lr", type=float, default=2e-5)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--unfreeze-layers", type=int, default=4)
    ap.add_argument("--siglip-unfreeze-blocks", type=int, default=0)
    ap.add_argument("--img-size", type=int, default=320)
    ap.add_argument("--max-ans", type=int, default=10)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="ocr")
    ap.add_argument("--zero-shot", action="store_true",
                    help="evaluate the untrained bridge once, then exit")
    args = ap.parse_args()

    device = resolve_device(args.device)
    print("[ocr] device=%s (%s)" % (device, describe(device)), flush=True)

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
    own, merged, up = enc.state_dict(), {}, 0
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

    d = cfg.hidden_size
    bridge = SiglipBridge(d, args.siglip_path,
                          unfreeze_blocks=args.siglip_unfreeze_blocks,
                          img_size=args.img_size).to(device)
    enc_head = nn.Linear(d, cfg.vocab_size).to(device)
    print("[ocr] jaume up=%d depth=%d siglip=%.1fM" % (
        up, cfg.num_layers,
        sum(p.numel() for p in bridge.tower.parameters()) / 1e6), flush=True)

    t0 = time.time()
    tr = build(args.n_train, seed=31)
    va = build(args.n_val, seed=909)
    print("[ocr] data %d/%d %.0fs" % (len(tr), len(va), time.time() - t0),
          flush=True)

    q_tok, a_tok = {}, {}
    for _, fam, which, ans, _, q in tr + va:
        if q not in q_tok:
            q_tok[q] = torch.tensor(
                tok(q, add_special_tokens=False)["input_ids"][:32],
                dtype=torch.long, device=device)
        if ans not in a_tok:
            ids = tok(" " + ans, add_special_tokens=False)["input_ids"]
            a_tok[ans] = torch.tensor(ids[:args.max_ans],
                                      dtype=torch.long, device=device)
    print("[ocr] distinct answers=%d queries=%d" % (len(a_tok), len(q_tok)),
          flush=True)

    vis_tr = [p for p in bridge.tower.parameters() if p.requires_grad]
    proj = list(bridge.proj.parameters())
    encp = [p for p in enc.parameters() if p.requires_grad]
    groups = [{"params": proj + list(enc_head.parameters()), "lr": args.lr}]
    if encp:
        groups.append({"params": encp, "lr": args.lr})
    if vis_tr:
        groups.append({"params": vis_tr, "lr": args.siglip_lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    allp = proj + list(enc_head.parameters()) + encp + vis_tr

    def step(batch):
        imgs = preprocess([b[0] for b in batch], args.img_size).to(device)
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
        logits = enc_head(h[:, nv + ql - 1:nv + ql - 1 + al, :].float())
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
        bridge.eval(); enc_head.eval(); enc.eval()
        per = {f: [0, 0] for f in OCR_FAMILIES}
        with torch.no_grad():
            for i in range(0, len(va) - 8 + 1, 8):
                b = va[i:i + 8]
                _, ok = step(b)
                for j, o in enumerate(ok):
                    per[b[j][1]][1] += 1
                    per[b[j][1]][0] += int(o)
        return per

    if args.zero_shot:
        per = evaluate()
        print("=== ZERO-SHOT %s (no training) ===" % args.tag, flush=True)
        for f in OCR_FAMILIES:
            c, n = per[f]
            print("  %-12s %.4f  (%d/%d)" % (f, c / max(n, 1), c, n),
                  flush=True)
        return 0

    for ep in range(args.epochs):
        bridge.train(); enc_head.train(); enc.train()
        perm = list(range(len(tr)))
        random.Random(ep).shuffle(perm)
        tot = nb = 0
        t0 = time.time()
        for i in range(0, len(tr) - args.batch_size + 1, args.batch_size):
            batch = [tr[j] for j in perm[i:i + args.batch_size]]
            loss, _ = step(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(allp, 1.0)
            opt.step()
            tot += float(loss); nb += 1
        per = evaluate()
        parts = ["%s %.3f" % (f, per[f][0] / max(per[f][1], 1))
                 for f in OCR_FAMILIES]
        mean = sum(per[f][0] / max(per[f][1], 1) for f in OCR_FAMILIES) / len(OCR_FAMILIES)
        print("[ocr] ep %d/%d loss %.3f | %s | mean %.3f (%.0fs)" % (
            ep + 1, args.epochs, tot / max(nb, 1), "  ".join(parts), mean,
            time.time() - t0), flush=True)

    per = evaluate()
    print("=== FINAL %s ===" % args.tag, flush=True)
    for f in OCR_FAMILIES:
        c, n = per[f]
        print("  %-12s %.4f  (%d/%d)" % (f, c / max(n, 1), c, n), flush=True)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        torch.save({"bridge": bridge.state_dict(),
                    "head": enc_head.state_dict()},
                   os.path.join(args.out, "ocr_head.pt"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
