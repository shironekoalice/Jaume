"""Train Jaume as a document-understanding VLM on locally rendered invoices.

Design notes that matter
------------------------
* The visual tower is a real CNN over a 384x480 grayscale-ish page, projected
  to 64 x d tokens, then prepended to the question tokens.
* Answers are short numeric/word strings. We score EXACT MATCH on the answer
  span, so a model that reads the wrong row or the wrong column gets 0.
* Question kinds are cycled so every kind appears equally: total, subtotal,
  invoice number, row count, first-line quantity. A model that always guesses
  the most common answer cannot beat 1/kind-count by much, and the answers
  vary per image anyway.
"""
import argparse
import math
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
from docvqa_env import make_invoice, make_question, QUESTION_KINDS


class DocTower(nn.Module):
    """384x480 page -> 64 patch tokens of width d."""

    def __init__(self, d):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(128, d, 3, stride=2, padding=1), nn.GELU(),
        )
        self.proj = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, x):
        h = self.conv(x)
        B, C, Hh, Ww = h.shape
        t = h.flatten(2).transpose(1, 2)
        n = t.shape[1]
        if n != 64:
            t = F.adaptive_avg_pool1d(t.transpose(1, 2), 64).transpose(1, 2)
        return self.proj(t)


def img_to_tensor(img, size=(96, 120)):
    import torchvision.transforms.functional as TF
    x = TF.to_tensor(img.convert("L"))
    x = TF.resize(x, list(size), antialias=True)
    return x


def encode_q(tok, kind, device, max_len=24):
    q = make_question(kind)
    ids = tok(q, add_special_tokens=False)["input_ids"][:max_len - 1]
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-val", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--unfreeze-layers", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)
    print("[doc] device=%s (%s)" % (device, describe(device)), flush=True)

    from jaume.modeling.config import JaumeConfig
    from jaume.modeling.embedding import JaumeEmbeddingModel
    from jaume.modeling.checkpoint import load_state_dict
    from jaume.data.dataset import load_tokenizer

    cfg = JaumeConfig.load(os.path.join(args.model, "config.json"))
    cfg.num_layers = args.depth
    tokdir = os.path.join(args.model, "tokenizer")
    tok = load_tokenizer(tokdir)
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
    print("[doc] warm start upcycled=%d depth=%d" % (up, cfg.num_layers),
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
    tower = DocTower(d).to(device)
    head = nn.Linear(d, cfg.vocab_size).to(device)

    print("[doc] building data ...", flush=True)
    t0 = time.time()
    items = []
    for i in range(args.n_train + args.n_val):
        seed = 700000 + i
        img, gt = make_invoice(seed)
        kind = QUESTION_KINDS[i % len(QUESTION_KINDS)][0]
        items.append((img_to_tensor(img), kind, str(gt[kind])))
    tr = items[:args.n_train]
    va = items[args.n_train:]
    print("[doc] data %d/%d in %.0fs" % (len(tr), len(va), time.time() - t0),
          flush=True)

    # Pre-tokenise questions (finite set) and answers.
    q_ids = {}
    for kind, _q, _ in QUESTION_KINDS:
        ids = encode_q(tok, kind, device)
        q_ids[kind] = torch.tensor(ids, dtype=torch.long, device=device)
    ans_ids = {}
    for _, kind, ans in tr + va:
        if ans not in ans_ids:
            a = tok(" " + ans, add_special_tokens=False)["input_ids"]
            ans_ids[ans] = torch.tensor(a, dtype=torch.long, device=device)
    print("[doc] distinct answers=%d" % len(ans_ids), flush=True)

    params = (list(tower.parameters()) + list(head.parameters())
              + [p for p in enc.parameters() if p.requires_grad])
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    print("[doc] trainable tensors=%d" % len(params), flush=True)

    def run(batch, train=True):
        imgs = torch.stack([b[0] for b in batch]).to(device)
        kinds = [b[1] for b in batch]
        answers = [b[2] for b in batch]
        v = tower(imgs)
        ql = max(len(q_ids[k]) for k in kinds)
        al = max(len(ans_ids[a]) for a in answers)
        L = ql + al
        ids = torch.zeros(len(batch), L, dtype=torch.long, device=device)
        qmask = torch.zeros(len(batch), L, dtype=torch.long, device=device)
        amask = torch.zeros(len(batch), L, dtype=torch.long, device=device)
        for i, k in enumerate(kinds):
            n = len(q_ids[k])
            ids[i, :n] = q_ids[k]
            qmask[i, :n] = 1
        for i, a in enumerate(answers):
            n = len(ans_ids[a])
            base = ql
            ids[i, base:base + n] = ans_ids[a]
            amask[i, base:base + n] = 1
        emb = enc.backbone.embed_tokens(ids)
        x = torch.cat([v, emb], 1)
        mask = torch.cat([torch.ones(v.shape[0], v.shape[1], dtype=torch.long,
                                     device=device), qmask], 1)
        h = x
        for layer in enc.backbone.layers:
            h, _aux = layer(h, mask)
        h = enc.backbone.norm(h)
        logits = head(h[:, v.shape[1] + ql - 1:, :].float())
        tgt = ids[:, ql:]
        am = amask[:, ql:]
        loss = F.cross_entropy(
            logits.reshape(-1, cfg.vocab_size)[am.reshape(-1).bool()],
            tgt.reshape(-1)[am.reshape(-1).bool()])
        pred = logits.argmax(-1)
        correct = 0
        for i in range(len(batch)):
            n = len(ans_ids[answers[i]])
            got = pred[i, :n].tolist()
            if got == ans_ids[answers[i]].tolist():
                correct += 1
        return loss, correct / len(batch)

    for ep in range(args.epochs):
        tower.train(); head.train(); enc.train()
        perm = list(range(len(tr)))
        random.Random(ep).shuffle(perm)
        tot = acc = 0.0
        nb = 0
        t0 = time.time()
        for i in range(0, len(tr) - args.batch_size + 1, args.batch_size):
            batch = [tr[j] for j in perm[i:i + args.batch_size]]
            loss, a = run(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot += float(loss); acc += a; nb += 1
        tower.eval(); head.eval(); enc.eval()
        with torch.no_grad():
            vt = va_ = 0.0
            vn = 0
            for i in range(0, len(va) - 8 + 1, 8):
                loss, a = run(va[i:i + 8], train=False)
                vt += float(loss); va_ += a; vn += 1
        print("[doc] ep %d/%d train loss %.3f acc %.3f | VAL loss %.3f acc %.3f (%.0fs)"
              % (ep + 1, args.epochs, tot / max(nb, 1), acc / max(nb, 1),
                 vt / max(vn, 1), va_ / max(vn, 1), time.time() - t0),
              flush=True)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        torch.save({"tower": tower.state_dict(), "head": head.state_dict()},
                   os.path.join(args.out, "docvqa_head.pt"))
        print("[doc] saved -> %s" % args.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
