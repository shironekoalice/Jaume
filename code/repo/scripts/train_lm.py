"""Stage 1 of the internal-loop programme: train a decoder-only LM on Jaume-Causal.

What this is
------------
Jaume-Causal is a causal twin of the Jaume embedding encoder: same layers,
same weights layout, causal mask instead of bidirectional.  It was trained
with a masked-LM objective, which makes it a reasonable -- not optimal --
starting point for a language model.  This script adds the missing piece, an
LM head, and trains next-token prediction.

The point of stage 1 is NOT to build a competitive language model.  It is to
establish a baseline whose depth can then be looped, so that the effect of
looping (stage 2) and the interior structure of the loop (stage 3) are
measured against a real, trained, non-looped reference.

Loop depth is a first-class argument (--num-loops), and the backbone can hand
back the hidden state after every loop round (--dump-loop-states), because
the next stage measures the interior of the loop rather than only its
output.

Usage
-----
    python scripts/train_lm.py --config configs/jaume-350m-loop.json \
        --data data/mlm_corpus.jsonl --output output/jaume-lm-350m \
        --num-loops 1 --epochs 1 --batch-size 32 --max-length 512 \
        --device npu

Then sweep the loop depth at inference time on the same checkpoint:

    python scripts/train_lm.py --eval-only --config ... --num-loops 4 ...
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jaume.device import resolve_device, seed_all, describe


def load_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def build_blocks(texts, tokenizer, block_len, max_blocks, seed=0):
    """Concatenate the corpus into a token stream and cut it into blocks.

    Blocks are packed back to back with no separator, so a block boundary cuts
    a sentence in half.  That is standard for LM pretraining and it keeps the
    token budget fully used.
    """
    rng = random.Random(seed)
    ids = []
    for t in texts:
        ids.extend(tokenizer(t, add_special_tokens=False)["input_ids"])
        if max_blocks and len(ids) >= block_len * max_blocks:
            break
    n = len(ids) // block_len
    if max_blocks:
        n = min(n, max_blocks)
    blocks = torch.tensor(ids[:n * block_len], dtype=torch.long).view(n, block_len)
    return blocks


def init_from_causal(model, ckpt_path):
    """Warm-start the loop backbone from a Jaume-Causal checkpoint."""
    from jaume.modeling.checkpoint import load_state_dict
    sd = load_state_dict(os.path.join(ckpt_path, "pytorch_model.bin"))
    src = {}
    for k, v in sd.items():
        if k.startswith("backbone."):
            src[k[len("backbone."):]] = v
    tgt = model.backbone.state_dict()
    loaded, skipped = [], []
    for k, v in src.items():
        if k in tgt and tgt[k].shape == v.shape:
            tgt[k] = v
            loaded.append(k)
        else:
            skipped.append(k)
    model.backbone.load_state_dict(tgt)
    if skipped:
        # A skipped embedding matrix means the vocab sizes differ, and the
        # LM head is sized from the tokenizer -- so this is a real problem,
        # not a detail.  Say so loudly instead of silently running with a
        # randomly initialised embedding.
        raise RuntimeError(
            f"warm start skipped {len(skipped)} tensors, e.g. {skipped[:5]}. "
            "Check that cfg.vocab_size == len(tokenizer) and that the "
            "checkpoint is a Jaume-Causal (model_type='jaume-causal') one.")
    return loaded, skipped


def ntp_loss(logits, labels, loss_chunk=0):
    """Next-token cross-entropy with an optional chunked vocabulary."""
    if loss_chunk and loss_chunk > 0:
        total = 0.0
        n = 0
        for i in range(0, logits.size(-1), loss_chunk):
            total = total + F.cross_entropy(
                logits[..., i:i + loss_chunk].reshape(-1, logits.size(-1)),
                labels.reshape(-1), reduction="sum")
            n += labels.numel()
        return total / max(n, 1)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))


@torch.no_grad()
def evaluate(model, val_blocks, device, loss_chunk=0, max_batches=8,
             batch_size=8):
    """Held-out loss. This is the number that decides whether a loop helps."""
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, min(val_blocks.shape[0], max_batches * batch_size),
                   batch_size):
        b = val_blocks[i:i + batch_size].to(device)
        inp, tgt = b[:, :-1], b[:, 1:]
        logits, aux = model(inp)
        loss = ntp_loss(logits, tgt, loss_chunk)
        tot += float(loss) * b.shape[0]
        n += b.shape[0]
    model.train()
    return tot / max(n, 1)


def loop_align_loss(loop_states, emb_weight, input_ids, offset=1):
    """LoopMTP-style latent soft alignment.

    LoopMTP (arXiv 2608.03624) observes that a model that loops T times can
    anticipate T future tokens, and supervises the loop by softly aligning
    the hidden state of loop t with the embedding of the token t positions
    ahead.  That gives the loop a dense, forward-looking signal, which plain
    next-token prediction does not provide to the intermediate rounds.

    The alignment here is the cosine between the L2-normalised loop-t hidden
    state at position i and the L2-normalised token embedding at position
    i+t+offset.  Only positions where the target is still inside the
    sequence contribute.  The last round is excluded: it carries the
    next-token prediction, and aligning it with the future as well would
    double-count the objective.

    offset: a causal model's hidden state at position i encodes the token at
    position i+1, so "the token t steps ahead" of what that state predicts
    is at position i+1+t -- that is offset=1, the default.  offset=0 aligns
    against the token already visible at position i, which is one step
    closer than LoopMTP's wording implies; it is kept switchable so the
    choice can be checked rather than assumed.
    """
    total = None
    n = 0
    B, L = input_ids.shape
    T = len(loop_states)
    for t in range(1, T):
        tgt_slice = input_ids[:, t + offset:]
        h = F.normalize(loop_states[t][:, : L - t - offset].float(), dim=-1)
        e = F.normalize(emb_weight[tgt_slice].float(), dim=-1)
        loss = 1.0 - (h * e).sum(-1).mean()
        total = loss if total is None else total + loss
        n += 1
    if total is None:
        return torch.zeros((), device=input_ids.device)
    return total / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--init-from", default=None,
                    help="Jaume-Causal checkpoint dir to warm-start from")
    ap.add_argument("--data", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--num-loops", type=int, default=1)
    ap.add_argument("--loop-form", choices=["full", "tail"], default="full")
    ap.add_argument("--loop-layers", type=int, default=0)
    ap.add_argument("--loop-position-mode", choices=["repeat", "increment"],
                    default="repeat")
    ap.add_argument("--loop-gate", action="store_true",
                    help="per-round learned gate on each round's residual "
                         "update. Round 0 stays pinned to 1.0 so at init the "
                         "model equals the plain loop.")
    ap.add_argument("--loop-gate-min", type=float, default=0.0)
    ap.add_argument("--loop-align-weight", type=float, default=0.0,
                    help="LoopMTP-style latent soft alignment: softly align the "
                         "hidden state of loop t with the token embedding t "
                         "positions ahead. 0 disables it.")
    ap.add_argument("--loop-align-offset", type=int, default=1,
                    help="0 aligns against the token already visible at the "
                         "same position; 1 (default) against the token one "
                         "step further, which matches 't steps ahead' for a "
                         "causal model whose hidden state predicts i+1.")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--max-blocks", type=int, default=None)
    ap.add_argument("--val-blocks", type=int, default=64)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--loss-chunk", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    device = resolve_device(args.device)
    print(f"[lm] device={device} ({describe(device)})", flush=True)

    from jaume.modeling.config import JaumeConfig
    from jaume.modeling.loop_lm import LoopLM
    from jaume.data.dataset import load_tokenizer

    cfg = JaumeConfig.load(args.config)
    cfg.num_loops = args.num_loops
    cfg.loop_form = args.loop_form
    cfg.loop_layers = args.loop_layers
    cfg.loop_position_mode = args.loop_position_mode
    cfg.loop_gate = args.loop_gate
    cfg.loop_gate_min = args.loop_gate_min
    cfg.causal = True
    cfg.model_type = "jaume-loop"

    tokdir = args.tokenizer
    if not tokdir:
        tokdir = os.path.join(REPO, "jaume_tokenizer")
    tokenizer = load_tokenizer(tokdir)
    # The causal twin was trained with a [MASK] token appended to the
    # vocabulary (151646 -> 151647), so the tokenizer must be resized the
    # same way before the embedding matrix can be warm-started.  This mirrors
    # train_mlm_causal.py; without it the embedding row count is off by one
    # and the warm start silently drops the whole embedding matrix.
    num_added = tokenizer.add_tokens(["[MASK]"])
    cfg.vocab_size = len(tokenizer)
    if num_added:
        print(f"[lm] tokenizer resized to {cfg.vocab_size} "
              f"(+{num_added} [MASK])", flush=True)

    model = LoopLM(cfg).to(device)
    print(f"[lm] params={model.num_parameters()/1e6:.1f}M  "
          f"effective_depth={model.backbone.effective_depth} "
          f"(layers={cfg.num_layers} loops={args.num_loops} "
          f"form={args.loop_form})", flush=True)

    if args.init_from:
        loaded, skipped = init_from_causal(model, args.init_from)
        print(f"[lm] warm start: {len(loaded)} tensors from {args.init_from}, "
              f"{len(skipped)} skipped", flush=True)

    if args.grad_ckpt:
        model.backbone.gradient_checkpointing_enable()

    if args.eval_only:
        print("[lm] eval-only: nothing else to do", flush=True)
        return 0

    texts = [r["text"] for r in load_jsonl(args.data) if "text" in r]
    if args.max_samples:
        texts = texts[:args.max_samples]
    print(f"[lm] {len(texts)} documents", flush=True)

    t0 = time.time()
    blocks = build_blocks(texts, tokenizer, args.max_length, args.max_blocks,
                          seed=args.seed)
    n_val = max(args.val_blocks, 8)
    val_blocks = blocks[:n_val]
    blocks = blocks[n_val:]
    print(f"[lm] {blocks.shape[0]} train blocks of {args.max_length} tokens, "
          f"{val_blocks.shape[0]} held out ({time.time()-t0:.0f}s)", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1,
                              betas=(0.9, 0.95))
    steps_per_epoch = max(1, blocks.shape[0] // (args.batch_size * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs

    def lr_at(step):
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        p = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))

    rng = random.Random(args.seed)
    seed_all(device, args.seed)
    step = 0
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(blocks.shape[0])
        opt.zero_grad(set_to_none=True)
        for bi, i in enumerate(range(0, blocks.shape[0] - args.batch_size + 1,
                                      args.batch_size)):
            b = blocks[perm[i:i + args.batch_size]].to(device)
            inp, tgt = b[:, :-1], b[:, 1:]
            if args.loop_align_weight > 0:
                logits, aux, loop_states = model(inp, return_loop_states=True)
            else:
                logits, aux = model(inp)
                loop_states = None
            loss = ntp_loss(logits, tgt, args.loss_chunk)
            if aux is not None:
                loss = loss + cfg.load_balance_weight * aux
            if loop_states is not None:
                loss = loss + args.loop_align_weight * loop_align_loss(
                    loop_states, model.backbone.embed_tokens.weight, inp,
                    args.loop_align_offset)
            (loss / args.grad_accum).backward()
            if (bi + 1) % args.grad_accum == 0:
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % args.log_every == 0:
                    tok_s = (step * args.batch_size * args.grad_accum
                             * args.max_length) / max(time.time() - t0, 1e-9)
                    print(f"[lm] epoch {ep} step {step}/{total_steps} "
                          f"loss {float(loss.detach()):.4f} "
                          f"lr {lr_at(step):.2e} tokens/s {tok_s:.0f}", flush=True)
                if args.eval_every and step % args.eval_every == 0:
                    vl = evaluate(model, val_blocks, device, args.loss_chunk,
                                  args.eval_batches, args.batch_size)
                    gate_str = ""
                    gp = getattr(model.backbone, "loop_gate_param", None)
                    if gp is not None:
                        gate_str = " gate[" + ",".join(
                            f"{float(v):.2f}" for v in gp.detach()) + "]"
                    print(f"[lm] epoch {ep} step {step} VAL loss {vl:.4f} "
                          f"ppl {math.exp(min(vl, 20)):.2f}{gate_str}",
                          flush=True)
            if args.save_every and step and step % args.save_every == 0:
                save(model, cfg, tokenizer, args.output, step)
    vl = evaluate(model, val_blocks, device, args.loss_chunk,
                  args.eval_batches, args.batch_size)
    print(f"[lm] FINAL VAL loss {vl:.4f} ppl {math.exp(min(vl, 20)):.2f}",
          flush=True)
    if args.output:
        save(model, cfg, tokenizer, args.output, step)
    print(f"[lm] done in {time.time()-t0:.0f}s", flush=True)
    return 0


def save(model, cfg, tokenizer, output, step):
    os.makedirs(output, exist_ok=True)
    cfg.save(os.path.join(output, "config.json"))
    sd = {"backbone." + k: v for k, v in model.backbone.state_dict().items()}
    sd["lm_head.weight"] = model.lm_head.weight.detach().cpu()
    torch.save(sd, os.path.join(output, "pytorch_model.bin"))
    try:
        tokenizer.save_pretrained(os.path.join(output, "tokenizer"))
    except Exception:
        pass
    print(f"[lm] saved step {step} -> {output}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
