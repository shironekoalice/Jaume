"""S2b: task-conditioned modulation -- one model, multiple internal paths.

Problem S2 exposed
------------------
Mixed training gave CIFAR 0.958 / VIS 0.950 but text_locate fell 1.00 -> 0.46.
The conflict diagnostic showed it is NOT simple gradient interference:
training OCR alone afterwards did not restore text_locate (0.34), and training
semantics alone wiped OCR entirely (0.000). The shared representation gets
*overwritten* by whichever task dominates.

Fix: task-conditioned modulation
--------------------------------
The question is already in the sequence, so it can tell the model which
internal path to take:

    g = sigmoid(MLP(pool(question_embedding)))      # per-task gate, dim d
    fused = g * sem_tokens + (1 - g) * ocr_tokens

Each task learns its own gate: semantic tasks push g toward sem, OCR tasks
toward ocr. One model, one checkpoint, but the two feature streams are no
longer forced through the same weighted sum. The gate is conditioned on the
*question*, so it generalises to unseen task mixes rather than being a
hard-coded task id.

A residual path keeps the un-gated concatenation available too:
    x = [fused | sem | ocr | q | a]
so information is not destroyed even when the gate is saturated.
"""
import argparse
import os
import pickle
import random
import string
import sys
import time

import numpy as np
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

from jaume.device import resolve_device, seed_all, describe
from ocr_suite import build as build_ocr, OCR_FAMILIES
from vision_suite import build as build_vis

VIS_FAMILIES = ["attr", "count", "spatial", "compare", "scene", "multi_attr"]
CHARS = list(string.digits) + list(string.ascii_letters) + ["$", " ", ".", "-"]
C2I = {c: i for i, c in enumerate(CHARS)}
PAD_C = len(CHARS)
N_C = len(CHARS) + 1
MAX_LEN = 8


def encode_answer(a):
    ids = [C2I[c] for c in a if c in C2I][:MAX_LEN]
    return ids + [PAD_C] * (MAX_LEN - len(ids))


class DualTowerTCM(nn.Module):
    """Dual encoder + question-conditioned gate between the two streams."""

    def __init__(self, d, siglip_path, trocr_path):
        super().__init__()
        from transformers import SiglipModel, VisionEncoderDecoderModel

        full = SiglipModel.from_pretrained(siglip_path, dtype=torch.float32)
        self.sem = full.vision_model
        hs = self.sem.config.hidden_size
        self.proj_sem = nn.Sequential(
            nn.LayerNorm(hs), nn.Linear(hs, d), nn.GELU(), nn.Linear(d, d))

        trocr = VisionEncoderDecoderModel.from_pretrained(
            trocr_path, dtype=torch.float32)
        self.ocr = trocr.encoder
        ho = self.ocr.config.hidden_size
        self.proj_ocr = nn.Sequential(
            nn.LayerNorm(ho), nn.Linear(ho, d), nn.GELU(), nn.Linear(d, d))

        # question-conditioned gate: one scalar weight per channel
        self.gate = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
            nn.Linear(d, d), nn.Sigmoid())

        for p in self.sem.parameters():
            p.requires_grad_(False)
        for p in self.ocr.parameters():
            p.requires_grad_(False)

    def forward(self, sem_px, ocr_px, q_pooled):
        """q_pooled: [B, d] mean-pooled question embedding."""
        s = self.proj_sem(self.sem(pixel_values=sem_px).last_hidden_state)
        o = self.proj_ocr(self.ocr(pixel_values=ocr_px).last_hidden_state)
        g = self.gate(q_pooled).unsqueeze(1)          # [B, 1, d]
        # the two towers emit different token counts (196 vs 577); align them
        # to a common length before the gated sum so the gate acts per token
        L = min(s.shape[1], o.shape[1])
        s_a = F.adaptive_avg_pool1d(s.transpose(1, 2), L).transpose(1, 2)
        o_a = F.adaptive_avg_pool1d(o.transpose(1, 2), L).transpose(1, 2)
        fused = g * s_a + (1.0 - g) * o_a
        return s, o, fused, g


def prep(images, size):
    import torchvision.transforms.functional as TF
    xs = []
    for im in images:
        x = TF.to_tensor(im.convert("RGB"))
        x = TF.resize(x, [size, size], antialias=True)
        xs.append((x - 0.5) / 0.5)
    return torch.stack(xs)


def load_cifar(root):
    import glob
    xs, ys = [], []
    for f in sorted(glob.glob(os.path.join(root, "data_batch_*"))):
        with open(f, "rb") as fh:
            d = pickle.load(fh, encoding="bytes")
        xs.append(d[b"data"]); ys.extend(d[b"labels"])
    return np.concatenate(xs).reshape(-1, 3, 32, 32), np.array(ys)


def cifar_images(arr):
    from PIL import Image
    return [Image.fromarray(a.transpose(1, 2, 0)) for a in arr]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--siglip-path", required=True)
    ap.add_argument("--trocr-path", required=True)
    ap.add_argument("--cifar-root", default=None)
    ap.add_argument("--s1-ckpt", default=None)
    ap.add_argument("--n-ocr", type=int, default=3000)
    ap.add_argument("--n-vis", type=int, default=2000)
    ap.add_argument("--n-cifar", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr-proj", type=float, default=5e-4)
    ap.add_argument("--lr-backbone", type=float, default=5e-5)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--unfreeze-layers", type=int, default=4)
    ap.add_argument("--ocr-weight", type=float, default=1.5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)
    print("[tcm] device=%s (%s)" % (device, describe(device)), flush=True)
    seed_all(device, 0)

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
    tower = DualTowerTCM(d, args.siglip_path, args.trocr_path).to(device)
    char_head = nn.Linear(d, N_C).to(device)
    cifar_head = nn.Linear(d, 10).to(device)

    if args.s1_ckpt and os.path.exists(args.s1_ckpt):
        ck = torch.load(args.s1_ckpt, map_location="cpu", weights_only=False)
        sd = tower.state_dict()
        hit = {k: v for k, v in ck["tower"].items()
               if k in sd and sd[k].shape == v.shape}
        sd.update(hit)
        tower.load_state_dict(sd)
        char_head.load_state_dict(ck["char_head"])
        print("[tcm] warm start S1: %d tensors" % len(hit), flush=True)
    print("[tcm] jaume up=%d gate params=%.2fM" % (
        up, sum(p.numel() for p in tower.gate.parameters()) / 1e6), flush=True)

    t0 = time.time()
    ocr_tr = build_ocr(args.n_ocr, seed=31)
    ocr_va = build_ocr(300, seed=909)
    vis_tr = build_vis(args.n_vis, seed=11)
    vis_va = build_vis(300, seed=777)
    cifar_tr = cifar_va = None
    if args.cifar_root and os.path.isdir(args.cifar_root):
        xa, ya = load_cifar(args.cifar_root)
        idx = np.arange(len(xa)); np.random.RandomState(0).shuffle(idx)
        n = min(args.n_cifar, len(idx) // 2)
        cifar_tr = (cifar_images(xa[idx[:n]]), torch.tensor(ya[idx[:n]]))
        cifar_va = (cifar_images(xa[idx[n:n + 500]]),
                    torch.tensor(ya[idx[n:n + 500]]))
    print("[tcm] data ocr %d vis %d cifar %s (%.0fs)" % (
        len(ocr_tr), len(vis_tr),
        len(cifar_tr[0]) if cifar_tr else 0, time.time() - t0), flush=True)

    q_tok = {}
    for _, _, _, _, _, q in ocr_tr + ocr_va + vis_tr + vis_va:
        if q not in q_tok:
            q_tok[q] = torch.tensor(
                tok(q, add_special_tokens=False)["input_ids"][:32],
                dtype=torch.long, device=device)

    proj_params = (list(tower.proj_sem.parameters())
                   + list(tower.proj_ocr.parameters())
                   + list(tower.gate.parameters())
                   + list(char_head.parameters())
                   + list(cifar_head.parameters()))
    enc_params = [p for p in enc.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [{"params": proj_params, "lr": args.lr_proj},
         {"params": enc_params, "lr": args.lr_backbone}], weight_decay=0.01)
    allp = proj_params + enc_params

    def q_pool(batch):
        """Mean-pooled question embedding, used to condition the gate."""
        B = len(batch)
        ql = max(len(q_tok[b[5]]) for b in batch)
        ids = torch.zeros(B, ql, dtype=torch.long, device=device)
        m = torch.zeros(B, ql, dtype=torch.long, device=device)
        for i, b in enumerate(batch):
            n = len(q_tok[b[5]]); ids[i, :n] = q_tok[b[5]]; m[i, :n] = 1
        e = enc.backbone.embed_tokens(ids)
        return (e * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp(min=1)

    def run_text(batch):
        sp = prep([b[0] for b in batch], 224).to(device)
        op = prep([b[0] for b in batch], 384).to(device)
        qp = q_pool(batch)
        s, o, fused, g = tower(sp, op, qp)
        B = len(batch)
        qs = [b[5] for b in batch]
        ans = [b[3] for b in batch]
        ql = max(len(q_tok[q]) for q in qs)
        aid = [torch.tensor(encode_answer(a), device=device) for a in ans]
        qids = torch.zeros(B, ql, dtype=torch.long, device=device)
        qm = torch.zeros(B, ql, dtype=torch.long, device=device)
        for i, q in enumerate(qs):
            n = len(q_tok[q]); qids[i, :n] = q_tok[q]; qm[i, :n] = 1
        a_in = torch.full((B, MAX_LEN), PAD_C, dtype=torch.long, device=device)
        for i, ids in enumerate(aid):
            a_in[i, 1:] = ids[:-1]
        emb = enc.backbone.embed_tokens(torch.cat([qids, a_in], dim=1))
        # [fused | sem | ocr | q | a] -- fused is the gated path, the raw
        # streams stay available as a residual so nothing is destroyed
        x = torch.cat([fused, s, o, emb], 1)
        mask = torch.cat([
            torch.ones(B, fused.shape[1], dtype=torch.long, device=device),
            torch.ones(B, s.shape[1], dtype=torch.long, device=device),
            torch.ones(B, o.shape[1], dtype=torch.long, device=device),
            qm,
            torch.ones(B, MAX_LEN, dtype=torch.long, device=device)], dim=1)
        h = x
        for layer in enc.backbone.layers:
            h, _a = layer(h, mask)
        h = enc.backbone.norm(h)
        base = fused.shape[1] + s.shape[1] + o.shape[1] + ql
        logits = char_head(h[:, base:base + MAX_LEN, :].float())
        tgt = torch.stack(aid)
        loss = F.cross_entropy(logits.reshape(-1, N_C), tgt.reshape(-1))
        pred = logits.argmax(-1)
        ok = [all(pred[i, j].item() == tgt[i, j].item()
                  for j in range(len(encode_answer(ans[i]))))
              for i in range(B)]
        return loss, ok, float(g.mean())

    def run_cifar(imgs, labels):
        sp = prep(imgs, 224).to(device)
        op = prep(imgs, 384).to(device)
        B = len(imgs)
        # CIFAR has no question: use a learned-free neutral gate (0.5)
        qp = torch.zeros(B, d, device=device)
        s, o, fused, g = tower(sp, op, qp)
        x = torch.cat([fused, s, o], 1)
        mask = torch.ones(B, x.shape[1], dtype=torch.long, device=device)
        h = x
        for layer in enc.backbone.layers:
            h, _a = layer(h, mask)
        h = enc.backbone.norm(h)
        logits = cifar_head(h.mean(1).float())
        loss = F.cross_entropy(logits, labels.to(device))
        acc = (logits.argmax(-1) == labels.to(device)).float().mean().item()
        return loss, acc

    def evaluate():
        tower.eval(); char_head.eval(); cifar_head.eval(); enc.eval()
        per = {f: [0, 0] for f in OCR_FAMILIES + VIS_FAMILIES}
        gates = {}
        with torch.no_grad():
            for src in (ocr_va, vis_va):
                for i in range(0, len(src) - 4 + 1, 4):
                    b = src[i:i + 4]
                    _, ok, gm = run_text(b)
                    for j, o_ in enumerate(ok):
                        per[b[j][1]][1] += 1
                        per[b[j][1]][0] += int(o_)
                        gates.setdefault(b[j][1], []).append(gm)
        ca = None
        if cifar_va:
            imgs, labs = cifar_va
            c = 0
            with torch.no_grad():
                for i in range(0, len(imgs), 32):
                    _, a_ = run_cifar(imgs[i:i+32], labs[i:i+32])
                    c += a_ * min(32, len(imgs) - i)
            ca = c / len(imgs)
        return per, ca, {k: float(np.mean(v)) for k, v in gates.items()}

    for ep in range(args.epochs):
        tower.train(); char_head.train(); cifar_head.train(); enc.train()
        perm_o = list(range(len(ocr_tr))); random.Random(ep).shuffle(perm_o)
        perm_v = list(range(len(vis_tr))); random.Random(ep + 100).shuffle(perm_v)
        perm_c = list(range(len(cifar_tr[0]))) if cifar_tr else []
        if perm_c:
            random.Random(ep + 200).shuffle(perm_c)
        io_ = iv_ = ic_ = 0
        tot = nb = 0
        t0 = time.time()
        n_steps = min(len(ocr_tr), len(vis_tr)) // args.batch_size
        for _ in range(n_steps):
            r = random.Random(ep * 7919 + nb).random()
            # OCR is weighted: 1.5x relative to semantics
            w_ocr = args.ocr_weight / (args.ocr_weight + 1.0)
            if r < w_ocr:
                batch = [ocr_tr[j] for j in perm_o[io_:io_ + args.batch_size]]
                io_ += args.batch_size
                if len(batch) < args.batch_size: break
                loss, _, _ = run_text(batch)
            elif r < w_ocr + (1 - w_ocr) * 0.7 or not perm_c:
                batch = [vis_tr[j] for j in perm_v[iv_:iv_ + args.batch_size]]
                iv_ += args.batch_size
                if len(batch) < args.batch_size: break
                loss, _, _ = run_text(batch)
            else:
                sel = perm_c[ic_:ic_ + args.batch_size]
                ic_ += args.batch_size
                if len(sel) < args.batch_size: break
                loss, _ = run_cifar([cifar_tr[0][j] for j in sel],
                                    cifar_tr[1][sel])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(allp, 1.0)
            opt.step()
            tot += float(loss); nb += 1
        per, ca, gates = evaluate()
        om = sum(per[f][0] / max(per[f][1], 1) for f in OCR_FAMILIES) / 3
        vm = sum(per[f][0] / max(per[f][1], 1) for f in VIS_FAMILIES) / 6
        print("[tcm] ep %d/%d loss %.3f | OCR %.3f (locate %.2f) | VIS %.3f"
              " | CIFAR %s | gate(sem) %.2f (%.0fs)"
              % (ep + 1, args.epochs, tot / max(nb, 1), om,
                 per["text_locate"][0] / max(per["text_locate"][1], 1), vm,
                 ("%.3f" % ca) if ca is not None else "n/a",
                 np.mean([gates.get(f, 0.5) for f in VIS_FAMILIES]),
                 time.time() - t0), flush=True)

    per, ca, gates = evaluate()
    print("=== FINAL S2b ===", flush=True)
    for f in OCR_FAMILIES:
        c, n = per[f]
        print("  OCR %-12s %.4f (%d/%d)" % (f, c / max(n, 1), c, n), flush=True)
    for f in VIS_FAMILIES:
        c, n = per[f]
        print("  VIS %-12s %.4f (%d/%d)" % (f, c / max(n, 1), c, n), flush=True)
    if ca is not None:
        print("  CIFAR        %.4f" % ca, flush=True)
    print("  gates: " + " ".join("%s=%.2f" % (k, v) for k, v in gates.items()),
          flush=True)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        torch.save({"tower": tower.state_dict(),
                    "char_head": char_head.state_dict(),
                    "cifar_head": cifar_head.state_dict()},
                   os.path.join(args.out, "dual_s2b.pt"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
