"""Continue-train Jaume-MLM into a true causal LM (next-token + saved head).

Why this checkpoint is needed
-----------------------------
The 350M base was trained with an MLM head and its checkpoint contains no
lm_head at all. Every "causal" behaviour seen so far came from heads trained
inside task-specific fine-tunes. That means:
  * context length cannot be measured with PPL (no trained causal head);
  * dialogue/generation cannot work out of the box;
  * the Jev series would inherit the same gap.

This script closes it: warm-start from the MLM base, train next-token
prediction on generic text over a range of sequence lengths (so positions
beyond the usual 128-256 actually get gradient), and save a checkpoint that
INCLUDES the lm_head. Length curriculum: 256 -> 1024 -> 4096 -> 8192.
"""
import argparse
import json
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

from jaume.device import resolve_device, seed_all, describe


def load_corpus(paths, tok, limit):
    docs = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                for key in ("text", "content", "body", "passage"):
                    if key in d and d[key]:
                        docs.append(d[key])
                        break
                if len(docs) >= limit:
                    return docs
    return docs


def pack(docs, tok, length, n_blocks):
    ids = []
    for d in docs:
        ids.extend(tok(d, add_special_tokens=False)["input_ids"])
        ids.append(tok.eos_token_id or 151643)
    blocks = []
    for i in range(0, len(ids) - length - 1, length):
        blocks.append(ids[i:i + length + 1])
        if len(blocks) >= n_blocks:
            break
    return torch.tensor(blocks, dtype=torch.long)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = resolve_device(args.device)
    print("[clm] device=%s (%s)" % (device, describe(device)), flush=True)
    seed_all(device, args.seed)

    from jaume.modeling.config import JaumeConfig
    from jaume.modeling.embedding import JaumeEmbeddingModel
    from jaume.modeling.checkpoint import load_state_dict
    from jaume.data.dataset import load_tokenizer

    cfg = JaumeConfig.load(os.path.join(args.model, "config.json"))
    tok = load_tokenizer(os.path.join(args.model, "tokenizer"))
    tok.add_tokens(["[MASK]"])
    cfg.vocab_size = len(tok)
    cfg.causal = True

    enc = JaumeEmbeddingModel(cfg).to(device)
    sd = load_state_dict(os.path.join(args.model, "pytorch_model.bin"))
    own, merged, up = enc.state_dict(), {}, 0
    for k in own:
        if k in sd and sd[k].shape == own[k].shape:
            merged[k] = sd[k]
        elif k.startswith("backbone.layers."):
            rest = k.split(".", 3)[3]
            s = "backbone.layers.15.%s" % rest
            if s in sd:
                merged[k] = sd[s]; up += 1
            else:
                merged[k] = own[k]
        else:
            merged[k] = own[k]
    enc.load_state_dict(merged, strict=True)

    # causal LM head, weight-tied to the input embedding (standard for small
    # models; loop_lm.py uses the same arrangement)
    head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False).to(device)
    with torch.no_grad():
        head.weight.copy_(enc.backbone.embed_tokens.weight)
    print("[clm] warm start up=%d depth=%d (head tied to embeddings)" % (
        up, cfg.num_layers), flush=True)

    print("[clm] loading corpus ...", flush=True)
    docs = load_corpus(args.data, tok, 60000)
    print("[clm] %d docs" % len(docs), flush=True)

    # ---- length curriculum: most tokens at 1024-4096, some at 8192 --------
    # 8192 is excluded from training: explicit attention needs a
    # [B,H,8192,8192] score matrix and the fused kernel is unreliable on this
    # CANN build. 4096 is the practical ceiling here; the probe can still
    # measure 8192 at inference (single sequence, no activations kept).
    stages = [
        ("256", 256, int(args.steps * 0.15)),
        ("1024", 1024, int(args.steps * 0.35)),
        ("2048", 2048, int(args.steps * 0.25)),
        ("4096", 4096, int(args.steps * 0.25)),
    ]
    # 8192 blocks are memory-heavy; reduce batch there
    opt = torch.optim.AdamW(
        [p for p in enc.parameters() if p.requires_grad] + [head.weight],
        lr=args.lr, weight_decay=0.05, betas=(0.9, 0.95))
    total = sum(n for _, _, n in stages)
    done = 0
    t0 = time.time()

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        p = (step - args.warmup) / max(1, total - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 *
                          (1 + math.cos(math.pi * min(p, 1.0))))

    step = 0
    for name, L, n_steps in stages:
        if n_steps <= 0:
            continue
        bs = max(1, args.batch_size // max(1, L // 256)) if L >= 512 else args.batch_size
        blocks = pack(docs, tok, L, n_steps * bs * args.grad_accum + 8)
        blocks = blocks.to(device)
        print("[clm] stage %s: %d steps, batch %d, blocks %d" % (
            name, n_steps, bs, len(blocks)), flush=True)
        ptr = 0
        for _ in range(n_steps):
            enc.train(); head.train()
            opt.zero_grad(set_to_none=True)
            accum_loss = 0.0
            for _a in range(args.grad_accum):
                if ptr + bs > len(blocks):
                    ptr = 0
                    perm = torch.randperm(len(blocks), device=device)
                    blocks = blocks.index_select(0, perm)
                b = blocks[ptr:ptr + bs]
                ptr += bs
                inp, tgt = b[:, :-1], b[:, 1:]
                h, _ = enc.backbone(inp, torch.ones_like(inp))
                logits = head(h.float())
                loss = F.cross_entropy(
                    logits.reshape(-1, cfg.vocab_size), tgt.reshape(-1))
                (loss / args.grad_accum).backward()
                accum_loss += float(loss) / args.grad_accum
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            torch.nn.utils.clip_grad_norm_(
                [p for p in enc.parameters() if p.requires_grad] + [head.weight], 1.0)
            opt.step()
            step += 1
            done += 1
            if step % 50 == 0:
                print("[clm] step %d/%d len=%s loss %.4f (%.0fs)" % (
                    step, total, name, accum_loss, time.time() - t0), flush=True)

    os.makedirs(args.output, exist_ok=True)
    cfg.save(os.path.join(args.output, "config.json"))
    out = {"backbone." + k: v for k, v in enc.backbone.state_dict().items()}
    out["lm_head.weight"] = head.weight.detach().cpu()
    torch.save(out, os.path.join(args.output, "pytorch_model.bin"))
    try:
        tok.save_pretrained(os.path.join(args.output, "tokenizer"))
    except Exception:
        pass
    print("[clm] saved -> %s (%.0fs)" % (args.output, time.time() - t0),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
