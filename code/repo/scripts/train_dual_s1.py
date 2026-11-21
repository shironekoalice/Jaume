"""S1: align the dual-tower VLM (SigLIP2 semantics + TrOCR glyphs) to Jaume.

Architecture (docs/vlm-dual-tower-spec.md):
    image -> SigLIP2  -> 196 semantic tokens -> proj_sem -> d
          -> TrOCR enc -> ~577 glyph tokens   -> proj_ocr -> d
    concat([sem | ocr | question | answer[:-1]]) -> Jaume -> char head

The two towers are frozen. Only the two projectors and Jaume's top-k layers
train, so S1 is cheap and cannot destroy the pretrained encoders.

Loss: next-character cross-entropy over the answer, teacher forced. This is
the same objective that made TrOCR's own decoder work, but here the decoder
is Jaume, which is what lets the two modalities share one language space.
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

import string
CHARS = list(string.digits) + list(string.ascii_letters) + ["$", " ", ".", "-"]
C2I = {c: i for i, c in enumerate(CHARS)}
PAD_C = len(CHARS)
N_C = len(CHARS) + 1
MAX_LEN = 8


def encode_answer(a):
    ids = [C2I[c] for c in a if c in C2I][:MAX_LEN]
    return ids + [PAD_C] * (MAX_LEN - len(ids))


class DualTower(nn.Module):
    """SigLIP2 (semantics) + TrOCR encoder (glyphs) -> two token streams."""

    def __init__(self, d, siglip_path, trocr_path):
        super().__init__()
        from transformers import SiglipModel, VisionEncoderDecoderModel

        full = SiglipModel.from_pretrained(siglip_path, dtype=torch.float32)
        self.sem = full.vision_model
        hs = self.sem.config.hidden_size
        self.proj_sem = nn.Sequential(
            nn.LayerNorm(hs), nn.Linear(hs, d), nn.GELU(), nn.Linear(d, d))

        # TrOCR = encoder + decoder; we keep ONLY the encoder features
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
        s = self.sem(pixel_values=sem_px).last_hidden_state     # [B,196,hs]
        o = self.ocr(pixel_values=ocr_px).last_hidden_state     # [B,T,ho]
        return self.proj_sem(s), self.proj_ocr(o)


def prep_sem(images, size=224):
    import torchvision.transforms.functional as TF
    xs = []
    for im in images:
        x = TF.to_tensor(im)
        x = TF.resize(x, [size, size], antialias=True)
        xs.append((x - 0.5) / 0.5)
    return torch.stack(xs)


def prep_ocr(images, size=384):
    import torchvision.transforms.functional as TF
    xs = []
    for im in images:
        x = TF.to_tensor(im.convert("RGB"))
        # TrOCR expects grayscale-ish input; keep 3 channels, normalise to [-1,1]
        x = TF.resize(x, [size, size], antialias=True)
        xs.append((x - 0.5) / 0.5)
    return torch.stack(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--siglip-path", required=True)
    ap.add_argument("--trocr-path", required=True)
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-val", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr-proj", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", type=float, default=1e-4)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--unfreeze-layers", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)
    print("[dual] device=%s (%s)" % (device, describe(device)), flush=True)
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
    n_tr = sum(p.numel() for p in tower.parameters() if p.requires_grad)
    print("[dual] trainable tower params=%.2fM, jaume up=%d" % (n_tr / 1e6, up),
          flush=True)

    t0 = time.time()
    tr = build_ocr(args.n_train, seed=31)
    va = build_ocr(args.n_val, seed=909)
    print("[dual] data %d/%d %.0fs" % (len(tr), len(va), time.time() - t0),
          flush=True)

    q_tok = {}
    for _, fam, which, ans, _, q in tr + va:
        if q not in q_tok:
            q_tok[q] = torch.tensor(
                tok(q, add_special_tokens=False)["input_ids"][:32],
                dtype=torch.long, device=device)

    proj_params = (list(tower.proj_sem.parameters())
                   + list(tower.proj_ocr.parameters())
                   + list(char_head.parameters()))
    enc_params = [p for p in enc.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [{"params": proj_params, "lr": args.lr_proj},
         {"params": enc_params, "lr": args.lr_backbone}], weight_decay=0.01)
    allp = proj_params + enc_params

    def step(batch):
        sem_px = prep_sem([b[0] for b in batch]).to(device)
        ocr_px = prep_ocr([b[0] for b in batch]).to(device)
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

    def evaluate():
        tower.eval(); char_head.eval(); enc.eval()
        per = {f: [0, 0] for f in OCR_FAMILIES}
        with torch.no_grad():
            for i in range(0, len(va) - 4 + 1, 4):
                b = va[i:i + 4]
                _, ok = step(b)
                for j, o_ in enumerate(ok):
                    per[b[j][1]][1] += 1
                    per[b[j][1]][0] += int(o_)
        return per

    for ep in range(args.epochs):
        tower.train(); char_head.train(); enc.train()
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
        print("[dual] ep %d/%d loss %.3f | %s | mean %.3f (%.0fs)" % (
            ep + 1, args.epochs, tot / max(nb, 1), "  ".join(parts), mean,
            time.time() - t0), flush=True)

    per = evaluate()
    print("=== FINAL S1 ===", flush=True)
    for f in OCR_FAMILIES:
        c, n = per[f]
        print("  %-12s %.4f  (%d/%d)" % (f, c / max(n, 1), c, n), flush=True)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        torch.save({"tower": tower.state_dict(),
                    "char_head": char_head.state_dict()},
                   os.path.join(args.out, "dual_s1.pt"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
