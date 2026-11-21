"""Train a causal LM on multi-digit addition and evaluate exact-match."""

import argparse
import math
import os
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
from carry_env import build_docs


def encode(docs, tok, max_length):
    out = []
    for prefix, ans, n_carry in docs:
        p = tok(prefix, add_special_tokens=False)["input_ids"]
        a = tok(ans, add_special_tokens=False)["input_ids"]
        ids = p + a
        if len(ids) > max_length:
            continue
        out.append({"ids": ids, "plen": len(p), "len": len(ids),
                    "n_carry": n_carry})
    return out


def span_loss(logits, ids, plen, lengths):
    B, L, V = logits.shape
    tgt = ids[:, 1:]
    lg = logits[:, :-1, :].reshape(-1, V)
    tg = tgt.reshape(-1)
    pos = torch.arange(L - 1, device=ids.device).unsqueeze(0)
    lo = (plen - 1).unsqueeze(1)
    hi = (lengths - 1).unsqueeze(1)
    mask = ((pos >= lo) & (pos < hi)).reshape(-1)
    if not bool(mask.any()):
        return logits.sum() * 0.0
    idx = mask.nonzero(as_tuple=True)[0]
    return F.cross_entropy(lg.index_select(0, idx), tg.index_select(0, idx))


@torch.no_grad()
def evaluate(enc, head, ids_all, plen_all, len_all, idx, pad, bs=16):
    enc.eval(); head.eval()
    right = total = 0
    for i in range(0, len(idx), bs):
        sel = idx[i:i + bs]
        ids = ids_all.index_select(0, sel)
        h, _ = enc.backbone(ids, (ids != pad).long())
        logits = head(h.float())
        pred = logits[:, :-1].argmax(-1)
        for j in range(ids.shape[0]):
            row = sel[j].item()
            p = int(plen_all[row])
            ln = int(len_all[row])
            got = pred[j, p - 1:ln - 1].tolist()
            want = ids[j, p:ln].tolist()
            if got == want:
                right += 1
            total += 1
    enc.train(); head.train()
    return right / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--init-from", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-digits", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--depth", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)
    print("[carry] device=%s (%s)" % (device, describe(device)), flush=True)

    from jaume.modeling.config import JaumeConfig
    from jaume.modeling.embedding import JaumeEmbeddingModel
    from jaume.modeling.checkpoint import load_state_dict
    from jaume.data.dataset import load_tokenizer

    cfg = JaumeConfig.load(args.config)
    if args.depth > 0:
        cfg.num_layers = args.depth
    tok = load_tokenizer(os.path.join(REPO, "jaume_tokenizer"))
    tok.add_tokens(["[MASK]"])
    cfg.vocab_size = len(tok)
    cfg.causal = True

    enc = JaumeEmbeddingModel(cfg).to(device)
    if args.init_from:
        pre = load_state_dict(os.path.join(args.init_from, "pytorch_model.bin"))
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
        print("[carry] warm start, upcycled %d" % up, flush=True)
    print("[carry] params=%.1fM depth=%d" % (
        sum(p.numel() for p in enc.parameters()) / 1e6, cfg.num_layers),
        flush=True)

    head = nn.Linear(cfg.hidden_size, cfg.vocab_size).to(device)
    if args.init_from:
        sd = load_state_dict(os.path.join(args.init_from, "pytorch_model.bin"))
        if "lm_head.weight" in sd:
            head.weight.data = sd["lm_head.weight"].to(device)

    seed_all(device, args.seed)
    t0 = time.time()
    tr = encode(build_docs(args.n_train, args.n_digits, args.seed),
                tok, args.max_length)
    va = encode(build_docs(args.n_val, args.n_digits, args.seed + 999),
                tok, args.max_length)
    print("[carry] train=%d val=%d in %.0fs" % (
        len(tr), len(va), time.time() - t0), flush=True)

    L = max(e["len"] for e in tr + va)
    pad = cfg.pad_token_id
    def pack(items):
        ids = torch.full((len(items), L), pad, dtype=torch.long)
        pl = torch.zeros(len(items), dtype=torch.long)
        ln = torch.zeros(len(items), dtype=torch.long)
        for i, e in enumerate(items):
            ids[i, :e["len"]] = torch.tensor(e["ids"], dtype=torch.long)
            pl[i] = e["plen"]; ln[i] = e["len"]
        return ids, pl, ln
    tr_ids, tr_pl, tr_ln = pack(tr)
    va_ids, va_pl, va_ln = pack(va)
    print("[carry] dense %s" % (tuple(tr_ids.shape),), flush=True)

    tr_ids = tr_ids.to(device); tr_pl = tr_pl.to(device)
    tr_ln = tr_ln.to(device)
    va_ids_d = va_ids.to(device); va_pl_d = va_pl.to(device)
    va_ln_d = va_ln.to(device)
    val_idx = torch.arange(len(va), device=device)

    params = list(enc.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05,
                            betas=(0.9, 0.95))
    spe = max(1, len(tr) // (args.batch_size * args.grad_accum))
    total = spe * args.epochs

    def lr_at(s):
        w = 100
        if s < w:
            return args.lr * (s + 1) / w
        p = (s - w) / max(1, total - w)
        return args.lr * (0.1 + 0.9 * 0.5 *
                          (1 + math.cos(math.pi * min(p, 1.0))))

    step = 0
    t0 = time.time()
    for ep in range(args.epochs):
        enc.train(); head.train()
        perm = torch.randperm(len(tr), device=device)
        opt.zero_grad(set_to_none=True)
        for bi, i in enumerate(range(0, len(tr) - args.batch_size + 1,
                                     args.batch_size)):
            sel = perm[i:i + args.batch_size]
            ids = tr_ids.index_select(0, sel)
            h, _ = enc.backbone(ids, (ids != pad).long())
            logits = head(h.float())
            loss = span_loss(logits, ids, tr_pl.index_select(0, sel),
                             tr_ln.index_select(0, sel))
            (loss / args.grad_accum).backward()
            if (bi + 1) % args.grad_accum == 0:
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % args.log_every == 0:
                    print("[carry] ep %d step %d/%d loss %.4f" % (
                        ep, step, total, float(loss.detach())), flush=True)
                if args.eval_every and step % args.eval_every == 0:
                    acc = evaluate(enc, head, va_ids_d, va_pl_d, va_ln_d,
                                   val_idx, pad)
                    print("[carry] ep %d step %d VAL_ACC %.4f" % (
                        ep, step, acc), flush=True)
    acc = evaluate(enc, head, va_ids_d, va_pl_d, va_ln_d, val_idx, pad)
    print("[carry] FINAL_ACC %.4f (%.0fs)" % (acc, time.time() - t0),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
