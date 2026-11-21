"""Train Jaume VL on detection / counting with coordinate tokens.

This is the WeVisOne-style capability: one head emits explicit coordinates
over a 32x32 grid, so counting, boxes and points all come from the same
decoder. Answers are generated autoregressively (the answer prefix is fed
back in), which is required because the number of objects varies per image.
"""
import argparse
import os
import random
import string
import sys
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

from jaume.device import resolve_device, seed_all, describe
from detect_env import build as build_det, DET_FAMILIES

CHARS = list(string.digits) + [" ", "-", "."]
C2I = {c: i for i, c in enumerate(CHARS)}
PAD_C = len(CHARS)
N_C = len(CHARS) + 1
MAX_LEN = 40          # up to 5 objects x 8 digits + separators


def enc_ans(a):
    ids = [C2I[c] for c in a if c in C2I][:MAX_LEN]
    return ids + [PAD_C] * (MAX_LEN - len(ids))


def parse_boxes(txt, bins):
    """Parse 'x1y1x2y2 x1y1x2y2 ...' into a list of 4-tuples."""
    out = []
    for tok in (txt or "").split():
        if len(tok) == 4 and tok.isdigit():
            out.append(tuple(int(c) for c in tok))
    return out


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = max(0, ax2 - ax1) * max(0, ay2 - ay1) +          max(0, bx2 - bx1) * max(0, by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def box_score(pred_txt, gt_txt, thr=0.3):
    """Mean best-IoU match per ground-truth box.

    Exact match is the wrong metric for detection: requiring all 20 digits of
    five boxes to be right reports 0.00 even when the model places every box
    approximately correctly. This mirrors standard detection evaluation.
    """
    gt = parse_boxes(gt_txt, 8)
    pr = parse_boxes(pred_txt, 8)
    if not gt:
        return 0.0
    if not pr:
        return 0.0
    hits = 0
    for g in gt:
        if max((iou(g, p) for p in pr), default=0.0) >= thr:
            hits += 1
    return hits / len(gt)


def point_score(pred_txt, gt_txt):
    """Fraction of predicted points that land on some ground-truth point."""
    def pts(t):
        out = []
        for tok in (t or "").split():
            if len(tok) == 2 and tok.isdigit():
                out.append((int(tok[0]), int(tok[1])))
        return out
    g, p = pts(gt_txt), pts(pred_txt)
    if not g:
        return 0.0
    hit = sum(1 for (x, y) in p
              if any(abs(x - gx) <= 1 and abs(y - gy) <= 1 for gx, gy in g))
    return hit / len(g)


class DualTCM(nn.Module):
    def __init__(self, d, sig, tro):
        super().__init__()
        from transformers import SiglipModel, VisionEncoderDecoderModel
        f = SiglipModel.from_pretrained(sig, dtype=torch.float32)
        self.sem = f.vision_model
        self.proj_sem = nn.Sequential(
            nn.LayerNorm(self.sem.config.hidden_size),
            nn.Linear(self.sem.config.hidden_size, d), nn.GELU(),
            nn.Linear(d, d))
        t = VisionEncoderDecoderModel.from_pretrained(tro, dtype=torch.float32)
        self.ocr = t.encoder
        self.proj_ocr = nn.Sequential(
            nn.LayerNorm(self.ocr.config.hidden_size),
            nn.Linear(self.ocr.config.hidden_size, d), nn.GELU(),
            nn.Linear(d, d))
        self.gate = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
                                  nn.Linear(d, d), nn.Sigmoid())
        for p in self.sem.parameters():
            p.requires_grad_(False)
        for p in self.ocr.parameters():
            p.requires_grad_(False)

    def forward(self, sp, op, qp):
        s = self.proj_sem(self.sem(pixel_values=sp).last_hidden_state)
        o = self.proj_ocr(self.ocr(pixel_values=op).last_hidden_state)
        g = self.gate(qp).unsqueeze(1)
        L = min(s.shape[1], o.shape[1])
        sa = F.adaptive_avg_pool1d(s.transpose(1, 2), L).transpose(1, 2)
        oa = F.adaptive_avg_pool1d(o.transpose(1, 2), L).transpose(1, 2)
        return s, o, g * sa + (1 - g) * oa, g


def prep(imgs, size):
    import torchvision.transforms.functional as TF
    xs = []
    for im in imgs:
        x = TF.to_tensor(im.convert("RGB"))
        x = TF.resize(x, [size, size], antialias=True)
        xs.append((x - 0.5) / 0.5)
    return torch.stack(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--siglip-path", required=True)
    ap.add_argument("--trocr-path", required=True)
    ap.add_argument("--s3-ckpt", default=None)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-val", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr-proj", type=float, default=5e-4)
    ap.add_argument("--lr-backbone", type=float, default=5e-5)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--unfreeze-layers", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)
    print("[det] device=%s (%s)" % (device, describe(device)), flush=True)
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
    for layer in list(enc.backbone.layers)[-args.unfreeze_layers:]:
        for p in layer.parameters():
            p.requires_grad_(True)
    for p in enc.backbone.norm.parameters():
        p.requires_grad_(True)

    d = cfg.hidden_size
    tower = DualTCM(d, args.siglip_path, args.trocr_path).to(device)
    chead = nn.Linear(d, N_C).to(device)
    if args.s3_ckpt and os.path.exists(args.s3_ckpt):
        ck = torch.load(args.s3_ckpt, map_location="cpu", weights_only=False)
        sd = tower.state_dict()
        hit = {k: v for k, v in ck["tower"].items()
               if k in sd and sd[k].shape == v.shape}
        sd.update(hit); tower.load_state_dict(sd)
        print("[det] warm start from %s (%d tensors)" % (args.s3_ckpt, len(hit)),
              flush=True)
    print("[det] jaume up=%d  out_vocab=%d  max_len=%d" % (
        up, N_C, MAX_LEN), flush=True)

    t0 = time.time()
    tr = build_det(args.n_train, seed=41)
    va = build_det(args.n_val, seed=909)
    print("[det] data %d/%d (%.0fs)" % (len(tr), len(va), time.time() - t0),
          flush=True)

    qt = {}
    for _, _, _, _, _, q in tr + va:
        if q not in qt:
            qt[q] = torch.tensor(
                tok(q, add_special_tokens=False)["input_ids"][:40],
                dtype=torch.long, device=device)

    def run(b):
        sp = prep([x[0] for x in b], 224).to(device)
        op = prep([x[0] for x in b], 384).to(device)
        B = len(b)
        ql = max(len(qt[x[5]]) for x in b)
        qids = torch.zeros(B, ql, dtype=torch.long, device=device)
        qm = torch.zeros_like(qids)
        for i, x in enumerate(b):
            n = len(qt[x[5]]); qids[i, :n] = qt[x[5]]; qm[i, :n] = 1
        e = enc.backbone.embed_tokens(qids)
        qp = (e * qm.unsqueeze(-1)).sum(1) / qm.sum(1, keepdim=True).clamp(min=1)
        s, o, fused, g = tower(sp, op, qp)
        ans = [x[3] for x in b]
        aid = [torch.tensor(enc_ans(x), device=device) for x in ans]
        ain = torch.full((B, MAX_LEN), PAD_C, dtype=torch.long, device=device)
        for i, ids in enumerate(aid):
            ain[i, 1:] = ids[:-1]
        emb = enc.backbone.embed_tokens(torch.cat([qids, ain], 1))
        x = torch.cat([fused, s, o, emb], 1)
        mask = torch.cat([
            torch.ones(B, fused.shape[1], dtype=torch.long, device=device),
            torch.ones(B, s.shape[1], dtype=torch.long, device=device),
            torch.ones(B, o.shape[1], dtype=torch.long, device=device), qm,
            torch.ones(B, MAX_LEN, dtype=torch.long, device=device)], 1)
        h = x
        for L in enc.backbone.layers:
            h, _ = L(h, mask)
        h = enc.backbone.norm(h)
        base = fused.shape[1] + s.shape[1] + o.shape[1] + ql
        logits = chead(h[:, base:base + MAX_LEN, :].float())
        tgt = torch.stack(aid)
        loss = F.cross_entropy(logits.reshape(-1, N_C), tgt.reshape(-1))
        pred = logits.argmax(-1)
        # decode each prediction back to a string, then score it with the
        # metric appropriate to the task family
        i2c = {v: k for k, v in C2I.items()}
        scores = []
        for i in range(B):
            n = len(enc_ans(ans[i]))
            got = "".join(i2c.get(int(t), "") for t in pred[i, :n])
            fam = b[i][1]
            if fam == "det_count":
                scores.append(1.0 if got.strip() == ans[i].strip() else 0.0)
            elif fam == "det_boxes":
                scores.append(box_score(got, ans[i]))
            else:
                scores.append(point_score(got, ans[i]))
        return loss, scores

    def ev():
        tower.eval(); chead.eval(); enc.eval()
        per = {f: [0, 0] for f in DET_FAMILIES}
        with torch.no_grad():
            for i in range(0, len(va) - 4 + 1, 4):
                b = va[i:i + 4]
                _, ok = run(b)
                for j, o_ in enumerate(ok):
                    per[b[j][1]][1] += 1
                    per[b[j][1]][0] += int(o_)
        return per

    ps = (list(tower.proj_sem.parameters()) + list(tower.proj_ocr.parameters())
          + list(tower.gate.parameters()) + list(chead.parameters()))
    ep_ = [p for p in enc.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": ps, "lr": args.lr_proj},
                             {"params": ep_, "lr": args.lr_backbone}],
                            weight_decay=0.01)
    allp = ps + ep_

    for e in range(args.epochs):
        tower.train(); chead.train(); enc.train()
        perm = list(range(len(tr)))
        random.Random(e).shuffle(perm)
        tot = nb = 0
        t0 = time.time()
        for i in range(0, len(tr) - args.batch_size + 1, args.batch_size):
            b = [tr[j] for j in perm[i:i + args.batch_size]]
            loss, _ = run(b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(allp, 1.0)
            opt.step()
            tot += float(loss); nb += 1
        per = ev()
        m = sum(per[f][0] / max(per[f][1], 1) for f in DET_FAMILIES) / 3
        print("[det] ep %d/%d loss %.3f | mean %.3f  %s (%.0fs)" % (
            e + 1, args.epochs, tot / max(nb, 1), m,
            " ".join("%s=%.2f" % (f, per[f][0] / max(per[f][1], 1))
                     for f in DET_FAMILIES), time.time() - t0), flush=True)

    per = ev()
    print("=== FINAL DET ===", flush=True)
    for f in DET_FAMILIES:
        c, n = per[f]
        print("  %-11s %.4f (%d/%d)" % (f, c / max(n, 1), c, n), flush=True)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        torch.save({"tower": tower.state_dict(),
                    "char_head": chead.state_dict()},
                   os.path.join(args.out, "dual_det.pt"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
