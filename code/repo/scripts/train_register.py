"""Train on the chain task, with --depth to sweep layers (the 8.10 filter)."""

import argparse
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
REPO = os.path.join(PROJECT_ROOT, "code", "repo")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jaume.device import resolve_device, seed_all, describe
from register_env import build_docs


@torch.no_grad()
def evaluate(model, ids_all, ans_all, idx, bs=8, max_batches=4):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, min(len(idx), max_batches * bs), bs):
        sel = idx[i:i + bs].to(ids_all.device)
        ids = ids_all.index_select(0, sel)
        tgt = ans_all.index_select(0, sel)
        logits, _ = model(ids)
        loss = F.cross_entropy(logits[:, -1, :].float(), tgt)
        tot += float(loss) * len(sel)
        n += len(sel)
    model.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--init-from", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--n-tasks", type=int, default=30000)
    ap.add_argument("--n-ops", type=int, default=16)
    ap.add_argument("--num-loops", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--max-length", type=int, default=192)
    ap.add_argument("--val-docs", type=int, default=256)
    ap.add_argument("--eval-every", type=int, default=300)
    ap.add_argument("--log-every", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)
    print(f"[ch] device={device} ({describe(device)})", flush=True)

    from jaume.modeling.config import JaumeConfig
    from jaume.modeling.loop_lm import LoopLM
    from jaume.modeling.checkpoint import load_state_dict
    from jaume.data.dataset import load_tokenizer

    cfg = JaumeConfig.load(args.config)
    cfg.num_layers = args.depth
    cfg.num_loops = args.num_loops
    tokdir = os.path.join(REPO, "jaume_tokenizer")
    tokenizer = load_tokenizer(tokdir)
    tokenizer.add_tokens(["[MASK]"])
    cfg.vocab_size = len(tokenizer)
    cfg.causal = True
    cfg.model_type = "jaume-loop"

    model = LoopLM(cfg).to(device)
    if args.init_from:
        sd = load_state_dict(os.path.join(args.init_from, "pytorch_model.bin"))
        tgt = model.backbone.state_dict()
        src = {k[len("backbone."):]: v for k, v in sd.items()
               if k.startswith("backbone.")}
        loaded = [k for k, v in src.items()
                  if k in tgt and tgt[k].shape == v.shape]
        for k in loaded:
            tgt[k] = src[k]
        model.backbone.load_state_dict(tgt)
        if "lm_head.weight" in sd:
            model.lm_head.weight.data = sd["lm_head.weight"].to(device)
        print(f"[ch] warm start {len(loaded)} tensors", flush=True)
    print(f"[ch] depth={args.depth} params={model.num_parameters()/1e6:.1f}M "
          f"n_ops={args.n_ops}", flush=True)

    seed_all(device, args.seed)
    docs = build_docs(args.n_tasks, args.n_ops, args.seed)
    print(f"[ch] {len(docs)} docs", flush=True)

    ans_ids = tokenizer([d["answer"] for d in docs],
                        add_special_tokens=False)["input_ids"]
    ids_list = []
    for d, a in zip(docs, ans_ids):
        p = tokenizer(d["prefix"], add_special_tokens=False)["input_ids"]
        ids = p + a
        if len(ids) > args.max_length:
            ids = ids[-args.max_length:]
        ids_list.append(ids)
    L = max(len(x) for x in ids_list)
    ids_all = torch.full((len(ids_list), L), cfg.pad_token_id, dtype=torch.long)
    ans_all = torch.zeros(len(ids_list), dtype=torch.long)
    for k, (ids, a) in enumerate(zip(ids_list, ans_ids)):
        ids_all[k, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        ans_all[k] = a[0]
    ids_dev = ids_all.to(device)
    ans_dev = ans_all.to(device)
    print(f"[ch] dense {tuple(ids_all.shape)}", flush=True)

    train_idx = torch.arange(args.val_docs, len(ids_list), device=device)
    val_idx = torch.arange(0, args.val_docs, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1,
                            betas=(0.9, 0.95))
    steps = max(1, len(train_idx) // (args.batch_size * args.grad_accum))
    total = steps * args.epochs

    def lr_at(s):
        if s < args.warmup_steps:
            return args.lr * (s + 1) / args.warmup_steps
        p = (s - args.warmup_steps) / max(1, total - args.warmup_steps)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))

    step = 0
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = train_idx[torch.randperm(len(train_idx), device=device)]
        opt.zero_grad(set_to_none=True)
        for bi, i in enumerate(range(0, len(train_idx) - args.batch_size + 1,
                                      args.batch_size)):
            sel = perm[i:i + args.batch_size].to(device)
            ids = ids_dev.index_select(0, sel)
            tgt = ans_dev.index_select(0, sel)
            logits, aux = model(ids)
            loss = F.cross_entropy(logits[:, -1, :].float(), tgt)
            if aux is not None:
                loss = loss + cfg.load_balance_weight * aux
            (loss / args.grad_accum).backward()
            if (bi + 1) % args.grad_accum == 0:
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % args.log_every == 0:
                    print(f"[ch] step {step}/{total} loss "
                          f"{float(loss.detach()):.4f}", flush=True)
                if args.eval_every and step % args.eval_every == 0:
                    vl = evaluate(model, ids_dev, ans_dev, val_idx)
                    print(f"[ch] step {step} VAL {vl:.4f}", flush=True)

    vl = evaluate(model, ids_dev, ans_dev, val_idx)
    print(f"[ch] FINAL VAL {vl:.4f} (depth={args.depth} n_ops={args.n_ops})",
          flush=True)
    os.makedirs(args.output, exist_ok=True)
    cfg.save(os.path.join(args.output, "config.json"))
    out = {"backbone." + k: v for k, v in model.backbone.state_dict().items()}
    out["lm_head.weight"] = model.lm_head.weight.detach().cpu()
    torch.save(out, os.path.join(args.output, "pytorch_model.bin"))
    print(f"[ch] saved ({time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
