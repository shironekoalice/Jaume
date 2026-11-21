"""OCR with a character-level head — the fix for the failed OCR probe.

The failed probe (train_ocr_siglip.py) classified a whole answer string as
one class out of 151647: a 6-digit invoice number had ~1 training example per
class. DeepSeek-OCR's decoder is autoregressive over text tokens for exactly
this reason.

Fix: answers are generated character by character with a small (<=64 class)
character head. Task difficulty becomes "at most 8 steps of 50-way
classification" instead of "1 draw from a 4800-way distribution".
Everything else (SigLIP2 bridge, frozen Jaume, top-k unfreeze) is unchanged.
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
        m.__getattr__ = lambda a: None
        sys.modules[name] = m
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

# Fixed character inventory across all three OCR sub-tasks.
import string
CHARS = (list(string.digits) + list(string.ascii_letters)
         + ["$", " ", ".", "-"])
C2I = {c: i for i, c in enumerate(CHARS)}
I2C = {i: c for c, i in C2I.items()}
PAD_C = len(CHARS)          # padding class
MAX_LEN = 8                 # longest answer we support ("box 4" = 5, 6-digit = 6)


def encode_answer(ans):
    ids = [C2I[c] for c in ans if c in C2I][:MAX_LEN]
    ids += [PAD_C] * (MAX_LEN - len(ids))
    return ids


def preprocess(images, size=224):
    import torchvision.transforms.functional as TF
    xs = []
    mean = torch.tensor([0.5] * 3).view(3, 1, 1)
    std = torch.tensor([0.5] * 3).view(3, 1, 1)
    for img in images:
        x = TF.to_tensor(img)
        x = TF.resize(x, [size, size], antialias=True)
        xs.append((x - mean) / std)
    return torch.stack(xs)


class SiglipBridge(nn.Module):
    def __init__(self, d_out, path, unfreeze_blocks=0):
        super().__init__()
        from transformers import SiglipModel
        full = SiglipModel.from_pretrained(path, dtype=torch.float32)
        self.tower = full.vision_model
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
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--teacher-forcing", action="store_true", default=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="charhead")
    args = ap.parse_args()

    device = resolve_device(args.device)
    print("[chc] device=%s (%s)" % (device, describe(device)), flush=True)

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
                          unfreeze_blocks=args.siglip_unfreeze_blocks).to(device)
    n_c = len(CHARS) + 1
    # one readout position predicts ALL answer characters: the sequence only
    # contains [vision | question], so there is exactly one position left to
    # read from. A multi-character head avoids needing answer tokens in the
    # input (which teacher forcing would otherwise require).
    char_head = nn.Linear(d, n_c).to(device)
    print("[chc] jaume up=%d depth=%d | char vocab=%d max_len=%d" % (
        up, cfg.num_layers, len(CHARS) + 1, MAX_LEN), flush=True)

    t0 = time.time()
    tr = build(args.n_train, seed=31)
    va = build(args.n_val, seed=909)
    print("[chc] data %d/%d %.0fs" % (len(tr), len(va), time.time() - t0),
          flush=True)

    q_tok = {}
    for _, fam, which, ans, _, q in tr + va:
        if q not in q_tok:
            q_tok[q] = torch.tensor(
                tok(q, add_special_tokens=False)["input_ids"][:32],
                dtype=torch.long, device=device)

    vis_tr = [p for p in bridge.tower.parameters() if p.requires_grad]
    proj = list(bridge.proj.parameters())
    encp = [p for p in enc.parameters() if p.requires_grad]
    groups = [{"params": proj + list(char_head.parameters()) + encp,
               "lr": args.lr}]
    if vis_tr:
        groups.append({"params": vis_tr, "lr": args.siglip_lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    allp = proj + list(char_head.parameters()) + encp + vis_tr

    def step(batch):
        imgs = preprocess([b[0] for b in batch], args.img_size).to(device)
        qs = [b[5] for b in batch]
        ans = [b[3] for b in batch]
        v = bridge(imgs)
        ql = max(len(q_tok[q]) for q in qs)
        B = len(batch)

        # TRUE AUTOREGRESSIVE teacher forcing:
        #   input  = [vision | question | answer[:-1]]
        #   target = answer
        # so the head sees a different position per character instead of
        # squeezing all 8 characters through one readout position.
        ans_ids = [torch.tensor(encode_answer(a), device=device) for a in ans]

        qids = torch.zeros(B, ql, dtype=torch.long, device=device)
        qm = torch.zeros(B, ql, dtype=torch.long, device=device)
        for i, q in enumerate(qs):
            n = len(q_tok[q]); qids[i, :n] = q_tok[q]; qm[i, :n] = 1

        # input answer = shifted right by one with a BOS-like pad
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
            h, _aux = layer(h, mask)
        h = enc.backbone.norm(h)

        nv = v.shape[1]
        # positions [nv+ql .. nv+ql+MAX_LEN-1] predict answer chars 0..MAX_LEN-1
        logits = char_head(
            h[:, nv + ql:nv + ql + MAX_LEN, :].float())   # [B, MAX_LEN, n_c]
        tgt = torch.stack(ans_ids)                        # [B, MAX_LEN]
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
            for i in range(0, len(va) - 8 + 1, 8):
                b = va[i:i + 8]
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
        print("[chc] ep %d/%d loss %.3f | %s | mean %.3f (%.0fs)" % (
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
                   os.path.join(args.out, "ocr_char.pt"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
