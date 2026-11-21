"""Tiled OCR: keep native resolution, tile the page, encode each tile.

Why tiles
---------
The diagnosis (docs/ocr-diagnosis.md) showed a linear probe on raw SigLIP2
features scores 0.19 on a 4-way text-location task -- at 224px a 20px glyph
shrinks to ~11px, smaller than one 16x16 patch. The information is destroyed
by downscaling, not by the bridge.

Fix: never downscale the page. Split it into overlapping tiles, resize EACH
TILE to 224 (so a 160x200 tile keeps its native detail), encode each with
SigLIP2, and concatenate the tile tokens. This is the "keep resolution,
compress tokens" principle from DeepSeek-OCR, implemented without changing
the vision tower.
"""
import argparse
import os
import sys
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

if "torchaudio" not in sys.modules:
    import importlib.machinery as _im
    import types

    def _stub(n):
        m = types.ModuleType(n)
        m.__version__ = "0"
        m.__spec__ = _im.ModuleSpec(n, loader=None)
        m.__path__ = []
        m.__getattr__ = lambda a: None
        sys.modules[n] = m
        return m

    _stub("torchaudio")
    for _s in ("torchaudio.transforms", "torchaudio.functional",
               "torchaudio.models", "torchaudio.utils"):
        _stub(_s)

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
REPO = os.path.join(PROJECT_ROOT, "code", "repo")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jaume.device import resolve_device, describe
from ocr_suite import build, OCR_FAMILIES

import string
CHARS = list(string.digits) + list(string.ascii_letters) + ["$", " ", ".", "-"]
C2I = {c: i for i, c in enumerate(CHARS)}
PAD_C = len(CHARS)
MAX_LEN = 8


def encode_answer(ans):
    ids = [C2I[c] for c in ans if c in C2I][:MAX_LEN]
    return ids + [PAD_C] * (MAX_LEN - len(ids))


def split_tiles(img, grid, overlap=0.15):
    """Split a PIL image into grid x grid overlapping tiles."""
    W, H = img.size
    tw, th = W / grid, H / grid
    ow, oh = tw * overlap, th * overlap
    tiles = []
    for gy in range(grid):
        for gx in range(grid):
            x0 = max(0, int(gx * tw - ow))
            y0 = max(0, int(gy * th - oh))
            x1 = min(W, int((gx + 1) * tw + ow))
            y1 = min(H, int((gy + 1) * th + oh))
            tiles.append(img.crop((x0, y0, x1, y1)))
    return tiles


class TiledBridge(nn.Module):
    def __init__(self, d_out, path, grid=2):
        super().__init__()
        from transformers import SiglipModel
        full = SiglipModel.from_pretrained(path, dtype=torch.float32)
        self.tower = full.vision_model
        self.grid = grid
        hidden = self.tower.config.hidden_size
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, d_out), nn.GELU(),
            nn.Linear(d_out, d_out))
        for p in self.tower.parameters():
            p.requires_grad_(False)

    def forward(self, tiles):
        """tiles: [B*G, 3, 224, 224] -> [B, G*196, d]"""
        t = self.tower(pixel_values=tiles).last_hidden_state  # [B*G,196,h]
        t = self.proj(t)
        b_g, n, d = t.shape
        b = b_g // (self.grid ** 2)
        return t.view(b, self.grid ** 2 * n, d)


def prep_tiles(images, grid, size=224):
    import torchvision.transforms.functional as TF
    out = []
    for img in images:
        for tile in split_tiles(img, grid):
            x = TF.to_tensor(tile)
            x = TF.resize(x, [size, size], antialias=True)
            out.append((x - 0.5) / 0.5)
    return torch.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--siglip-path", required=True)
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-val", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--unfreeze-layers", type=int, default=4)
    ap.add_argument("--grid", type=int, default=2)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="tile")
    args = ap.parse_args()

    device = resolve_device(args.device)
    print("[tile] device=%s (%s) grid=%dx%d tokens/box=%d" % (
        device, describe(device), args.grid, args.grid,
        args.grid ** 2 * 196), flush=True)

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
    bridge = TiledBridge(d, args.siglip_path, grid=args.grid).to(device)
    n_c = len(CHARS) + 1
    char_head = nn.Linear(d, n_c).to(device)

    t0 = time.time()
    tr = build(args.n_train, seed=31)
    va = build(args.n_val, seed=909)
    print("[tile] data %d/%d %.0fs" % (len(tr), len(va), time.time() - t0),
          flush=True)

    q_tok = {}
    for _, fam, which, ans, _, q in tr + va:
        if q not in q_tok:
            q_tok[q] = torch.tensor(
                tok(q, add_special_tokens=False)["input_ids"][:32],
                dtype=torch.long, device=device)

    proj = list(bridge.proj.parameters())
    encp = [p for p in enc.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(proj + list(char_head.parameters()) + encp,
                            lr=args.lr, weight_decay=0.01)
    allp = proj + list(char_head.parameters()) + encp

    def step(batch):
        tiles = prep_tiles([b[0] for b in batch], args.grid).to(device)
        qs = [b[5] for b in batch]
        ans = [b[3] for b in batch]
        v = bridge(tiles)                      # [B, G*196, d]
        B = len(batch)
        ql = max(len(q_tok[q]) for q in qs)
        ans_ids = [torch.tensor(encode_answer(a), device=device) for a in ans]

        qids = torch.zeros(B, ql, dtype=torch.long, device=device)
        qm = torch.zeros(B, ql, dtype=torch.long, device=device)
        for i, q in enumerate(qs):
            n = len(q_tok[q]); qids[i, :n] = q_tok[q]; qm[i, :n] = 1
        a_in = torch.full((B, MAX_LEN), PAD_C, dtype=torch.long, device=device)
        for i, ids in enumerate(ans_ids):
            a_in[i, 1:] = ids[:-1]

        emb = enc.backbone.embed_tokens(torch.cat([qids, a_in], dim=1))
        x = torch.cat([v, emb], 1)
        mask = torch.cat([
            torch.ones(v.shape[0], v.shape[1], dtype=torch.long, device=device),
            qm,
            torch.ones(B, MAX_LEN, dtype=torch.long, device=device)], dim=1)
        h = x
        for layer in enc.backbone.layers:
            h, _a = layer(h, mask)
        h = enc.backbone.norm(h)
        nv = v.shape[1]
        logits = char_head(h[:, nv + ql:nv + ql + MAX_LEN, :].float())
        tgt = torch.stack(ans_ids)
        loss = F.cross_entropy(logits.reshape(-1, n_c), tgt.reshape(-1))
        pred = logits.argmax(-1)
        ok = [all(pred[i, j].item() == tgt[i, j].item()
                  for j in range(len(encode_answer(ans[i]))))
              for i in range(B)]
        return loss, ok

    def evaluate():
        bridge.eval(); char_head.eval(); enc.eval()
        per = {f: [0, 0] for f in OCR_FAMILIES}
        with torch.no_grad():
            for i in range(0, len(va) - 4 + 1, 4):
                b = va[i:i + 4]
                _, ok = step(b)
                for j, o in enumerate(ok):
                    per[b[j][1]][1] += 1
                    per[b[j][1]][0] += int(o)
        return per

    for ep in range(args.epochs):
        bridge.train(); char_head.train(); enc.train()
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
        mean = sum(per[f][0] / max(per[f][1], 1)
                   for f in OCR_FAMILIES) / len(OCR_FAMILIES)
        print("[tile] ep %d/%d loss %.3f | %s | mean %.3f (%.0fs)" % (
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
                    "char_head": char_head.state_dict()},
                   os.path.join(args.out, "ocr_tile.pt"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
