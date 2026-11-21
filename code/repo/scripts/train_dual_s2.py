"""S2: mixed training -- keep semantics while retaining OCR.

S1 proved the dual tower can read (OCR mean 0.925). S2 asks the harder
question: does adding semantic tasks (basic-visual suite + real CIFAR images)
degrade OCR, and does OCR training degrade semantics?

Three datasets are mixed in every batch:
  * OCR suite   (doc_field / char_read / text_locate)  -- glyphs
  * vision suite(attr / count / spatial / compare / scene / multi_attr)
                                                        -- semantics
  * CIFAR-10    (10-class real images)                 -- real-world semantics

Each is evaluated separately every epoch, so a regression in any direction
is visible immediately.
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


class DualTower(nn.Module):
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
        for p in self.sem.parameters():
            p.requires_grad_(False)
        for p in self.ocr.parameters():
            p.requires_grad_(False)

    def forward(self, sem_px, ocr_px):
        s = self.sem(pixel_values=sem_px).last_hidden_state
        o = self.ocr(pixel_values=ocr_px).last_hidden_state
        return self.proj_sem(s), self.proj_ocr(o)


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
    x = np.concatenate(xs).reshape(-1, 3, 32, 32)
    return x, np.array(ys)


def cifar_images(arr):
    from PIL import Image
    out = []
    for a in arr:
        img = Image.fromarray(a.transpose(1, 2, 0))
        out.append(img)
    return out


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
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr-proj", type=float, default=5e-4)
    ap.add_argument("--lr-backbone", type=float, default=5e-5)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--unfreeze-layers", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)
    print("[s2] device=%s (%s)" % (device, describe(device)), flush=True)
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
    tower = DualTower(d, args.siglip_path, args.trocr_path).to(device)
    char_head = nn.Linear(d, N_C).to(device)
    cifar_head = nn.Linear(d, 10).to(device)

    if args.s1_ckpt and os.path.exists(args.s1_ckpt):
        try:
            ck = torch.load(args.s1_ckpt, map_location="cpu", weights_only=False)
        except TypeError:
            ck = torch.load(args.s1_ckpt, map_location="cpu")
        tower.load_state_dict(ck["tower"], strict=False)
        char_head.load_state_dict(ck["char_head"])
        print("[s2] warm start from S1 checkpoint", flush=True)
    print("[s2] jaume up=%d" % up, flush=True)

    t0 = time.time()
    ocr_tr = build_ocr(args.n_ocr, seed=31)
    ocr_va = build_ocr(max(300, args.n_ocr // 10), seed=909)
    vis_tr = build_vis(args.n_vis, seed=11)
    vis_va = build_vis(max(300, args.n_vis // 10), seed=777)
    print("[s2] ocr %d/%d  vis %d/%d  %.0fs" % (
        len(ocr_tr), len(ocr_va), len(vis_tr), len(vis_va),
        time.time() - t0), flush=True)

    cifar_tr = cifar_va = None
    if args.cifar_root and os.path.isdir(args.cifar_root):
        xa, ya = load_cifar(args.cifar_root)
        idx = np.arange(len(xa)); np.random.RandomState(0).shuffle(idx)
        n = min(args.n_cifar, len(idx) // 2)
        cifar_tr = (cifar_images(xa[idx[:n]]), torch.tensor(ya[idx[:n]]))
        cifar_va = (cifar_images(xa[idx[n:n + max(500, n // 5)]]),
                    torch.tensor(ya[idx[n:n + max(500, n // 5)]]))
        print("[s2] cifar %d/%d" % (len(cifar_tr[0]), len(cifar_va[0])),
              flush=True)

    q_tok = {}
    for _, fam, which, ans, _, q in ocr_tr + ocr_va:
        if q not in q_tok:
            q_tok[q] = torch.tensor(
                tok(q, add_special_tokens=False)["input_ids"][:32],
                dtype=torch.long, device=device)
    for _, fam, which, ans, opts, q in vis_tr + vis_va:
        if q not in q_tok:
            q_tok[q] = torch.tensor(
                tok(q, add_special_tokens=False)["input_ids"][:32],
                dtype=torch.long, device=device)

    proj_params = (list(tower.proj_sem.parameters())
                   + list(tower.proj_ocr.parameters())
                   + list(char_head.parameters())
                   + list(cifar_head.parameters()))
    enc_params = [p for p in enc.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [{"params": proj_params, "lr": args.lr_proj},
         {"params": enc_params, "lr": args.lr_backbone}], weight_decay=0.01)
    allp = proj_params + enc_params

    def run_text(batch):
        """Shared path for OCR and vision-suite items (both are QA)."""
        sem_px = prep([b[0] for b in batch], 224).to(device)
        ocr_px = prep([b[0] for b in batch], 384).to(device)
        s, o = tower(sem_px, ocr_px)
        B = len(batch)
        qs = [b[5] for b in batch]
        ans = [b[3] for b in batch]
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
        x = torch.cat([s, o, emb], 1)
        mask = torch.cat([
            torch.ones(B, s.shape[1], dtype=torch.long, device=device),
            torch.ones(B, o.shape[1], dtype=torch.long, device=device),
            qm,
            torch.ones(B, MAX_LEN, dtype=torch.long, device=device)], dim=1)
        h = x
        for layer in enc.backbone.layers:
            h, _a = layer(h, mask)
        h = enc.backbone.norm(h)
        base = s.shape[1] + o.shape[1] + ql
        logits = char_head(h[:, base:base + MAX_LEN, :].float())
        tgt = torch.stack(ans_ids)
        loss = F.cross_entropy(logits.reshape(-1, N_C), tgt.reshape(-1))
        pred = logits.argmax(-1)
        ok = [all(pred[i, j].item() == tgt[i, j].item()
                  for j in range(len(encode_answer(ans[i]))))
              for i in range(B)]
        return loss, ok

    def run_cifar(imgs, labels):
        sem_px = prep(imgs, 224).to(device)
        ocr_px = prep(imgs, 384).to(device)
        s, o = tower(sem_px, ocr_px)
        B = len(imgs)
        x = torch.cat([s, o], 1)
        mask = torch.ones(B, x.shape[1], dtype=torch.long, device=device)
        h = x
        for layer in enc.backbone.layers:
            h, _a = layer(h, mask)
        h = enc.backbone.norm(h)
        pooled = h.mean(1)
        logits = cifar_head(pooled.float())
        loss = F.cross_entropy(logits, labels.to(device))
        acc = (logits.argmax(-1) == labels.to(device)).float().mean().item()
        return loss, acc

    def evaluate():
        tower.eval(); char_head.eval(); cifar_head.eval(); enc.eval()
        per = {f: [0, 0] for f in OCR_FAMILIES + VIS_FAMILIES}
        with torch.no_grad():
            for i in range(0, len(ocr_va) - 4 + 1, 4):
                b = ocr_va[i:i + 4]
                _, ok = run_text(b)
                for j, o_ in enumerate(ok):
                    per[b[j][1]][1] += 1
                    per[b[j][1]][0] += int(o_)
            for i in range(0, len(vis_va) - 4 + 1, 4):
                b = vis_va[i:i + 4]
                _, ok = run_text(b)
                for j, o_ in enumerate(ok):
                    per[b[j][1]][1] += 1
                    per[b[j][1]][0] += int(o_)
        cifar_acc = None
        if cifar_va is not None:
            imgs, labels = cifar_va
            correct = 0
            with torch.no_grad():
                for i in range(0, len(imgs), 32):
                    _, a = run_cifar(imgs[i:i+32], labels[i:i+32])
                    correct += a * min(32, len(imgs) - i)
            cifar_acc = correct / len(imgs)
        return per, cifar_acc

    ocr_keys = set(OCR_FAMILIES)
    for ep in range(args.epochs):
        tower.train(); char_head.train(); cifar_head.train(); enc.train()
        # interleave: each step draws one batch from a randomly chosen source
        pools = ["ocr", "vis"] + (["cifar"] if cifar_tr is not None else [])
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
            src = random.Random(ep * 7919 + nb).choice(pools)
            if src == "ocr":
                batch = [ocr_tr[j] for j in perm_o[io_:io_ + args.batch_size]]
                io_ += args.batch_size
                if len(batch) < args.batch_size: break
                loss, _ = run_text(batch)
            elif src == "vis":
                batch = [vis_tr[j] for j in perm_v[iv_:iv_ + args.batch_size]]
                iv_ += args.batch_size
                if len(batch) < args.batch_size: break
                loss, _ = run_text(batch)
            else:
                sel = perm_c[ic_:ic_ + args.batch_size]
                ic_ += args.batch_size
                if len(sel) < args.batch_size: break
                imgs = [cifar_tr[0][j] for j in sel]
                labs = cifar_tr[1][sel]
                loss, _ = run_cifar(imgs, labs)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(allp, 1.0)
            opt.step()
            tot += float(loss); nb += 1
        per, cifar_acc = evaluate()
        ocr_mean = sum(per[f][0] / max(per[f][1], 1) for f in OCR_FAMILIES) / len(OCR_FAMILIES)
        vis_mean = sum(per[f][0] / max(per[f][1], 1) for f in VIS_FAMILIES) / len(VIS_FAMILIES)
        print("[s2] ep %d/%d loss %.3f | OCR %.3f (%s) | VIS %.3f | CIFAR %s (%.0fs)"
              % (ep + 1, args.epochs, tot / max(nb, 1), ocr_mean,
                 " ".join("%s=%.2f" % (f, per[f][0] / max(per[f][1], 1))
                          for f in OCR_FAMILIES),
                 vis_mean,
                 ("%.3f" % cifar_acc) if cifar_acc is not None else "n/a",
                 time.time() - t0), flush=True)

    per, cifar_acc = evaluate()
    print("=== FINAL S2 ===", flush=True)
    for f in OCR_FAMILIES:
        c, n = per[f]
        print("  OCR %-12s %.4f (%d/%d)" % (f, c / max(n, 1), c, n), flush=True)
    for f in VIS_FAMILIES:
        c, n = per[f]
        print("  VIS %-12s %.4f (%d/%d)" % (f, c / max(n, 1), c, n), flush=True)
    if cifar_acc is not None:
        print("  CIFAR        %.4f" % cifar_acc, flush=True)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        torch.save({"tower": tower.state_dict(),
                    "char_head": char_head.state_dict(),
                    "cifar_head": cifar_head.state_dict()},
                   os.path.join(args.out, "dual_s2.pt"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
