"""Train the looped LM on a task whose stage 2 depends on stage 1's result.

The corpus has two distinct target kinds in one document, and the order
matters:

    derive: <givens> => <implied facts>
    place:  <remaining items>

`derive` can only be produced by running the propagation; `place` operates on
whatever `derive` left undetermined.  A one-pass model that never revisits
its own conclusion has strictly less information than one that does -- which
is exactly the property the previous single-step task lacked.
"""

import argparse
import math
import os
import random
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
from derive_env import make_puzzle, make_chain_puzzle, render


def build_docs(n_tasks, n_items, n_slots, n_fixed, n_same, n_notsame, seed,
               chain_len=0):
    """One document per puzzle. target mask covers BOTH target spans."""
    docs = []
    for t in range(n_tasks):
        if chain_len > 0:
            cons, env = make_chain_puzzle(n_items, n_slots, chain_len,
                                          seed * 1000003 + t)
        else:
            cons, env = make_puzzle(n_items, n_slots, n_fixed, n_same,
                                    n_notsame, seed * 1000003 + t)
        if env is None:
            continue
        r = render(env)
        prefix = (f"puzzle: {r['given']} derive: ")
        target1 = f"{r['implied']} " if r["implied"] else "none "
        mid = "place: "
        target2 = " ".join(f"item{i}" for i in r["remaining"])
        if not target2:
            target2 = "none"
        docs.append({
            "prefix": prefix,
            "target1": target1,
            "mid": mid,
            "target2": target2,
            "depth": r["depth"],
        })
    return docs


def encode(docs, tok, max_length):
    out = []
    for d in docs:
        p = tok(d["prefix"], add_special_tokens=False)["input_ids"]
        t1 = tok(d["target1"], add_special_tokens=False)["input_ids"]
        m = tok(d["mid"], add_special_tokens=False)["input_ids"]
        t2 = tok(d["target2"], add_special_tokens=False)["input_ids"]
        ids = p + t1 + m + t2
        if len(ids) > max_length:
            keep = max_length - (len(t1) + len(m) + len(t2))
            if keep < 8:
                continue
            ids = p[-keep:] + t1 + m + t2
        s1 = len(p) - (len(p) - min(len(p), max_length - len(t1) - len(m) - len(t2))) if len(p) + len(t1) + len(m) + len(t2) > max_length else len(p)
        # recompute cleanly: positions in the final list
        np_ = len(ids) - (len(t1) + len(m) + len(t2))
        start1 = np_ - 1          # target1 begins here (predict np_)
        start2 = np_ + len(t1) + len(m) - 1
        out.append({"ids": ids, "start1": start1, "start2": start2,
                    "len": len(ids)})
    return out


def staged_loss(logits, ids, start1, start2, lengths):
    """CE on the two target spans only.

    target1 spans [start1+1, start1+len_t1]; target2 spans
    [start2+1, len-1].  Both are contiguous, so two range masks suffice.
    Everything is tensor ops -- the earlier per-row Python loop was what
    stalled the run at step 0.
    """
    B, L, V = logits.shape
    tgt = ids[:, 1:]
    lg = logits[:, :-1, :].reshape(-1, V)
    tg = tgt.reshape(-1)
    pos = torch.arange(L - 1, device=ids.device).unsqueeze(0)
    lo1 = (start1 + 1).unsqueeze(1)
    hi1 = (start2).unsqueeze(1)
    lo2 = (start2 + 1).unsqueeze(1)
    hi2 = (lengths - 1).unsqueeze(1)
    m1 = (pos >= lo1) & (pos < hi1)
    m2 = (pos >= lo2) & (pos < hi2)
    mask = (m1 | m2).reshape(-1)
    if not bool(mask.any()):
        return logits.sum() * 0.0
    idx = mask.nonzero(as_tuple=True)[0]
    sel = lg.index_select(0, idx)
    return F.cross_entropy(sel, tg.index_select(0, idx))


@torch.no_grad()
def evaluate(model, ids_all, s1_all, s2_all, len_all, idx, bs=8, max_batches=4):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, min(len(idx), max_batches * bs), bs):
        sel = idx[i:i + bs].to(ids_all.device)
        ids = ids_all.index_select(0, sel)
        logits, aux = model(ids)
        loss = staged_loss(logits, ids, s1_all.index_select(0, sel),
                           s2_all.index_select(0, sel),
                           len_all.index_select(0, sel))
        tot += float(loss) * len(sel)
        n += len(sel)
    model.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--init-from", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-tasks", type=int, default=12000)
    ap.add_argument("--n-items", type=int, default=6)
    ap.add_argument("--n-slots", type=int, default=4)
    ap.add_argument("--n-fixed", type=int, default=3)
    ap.add_argument("--n-same", type=int, default=2)
    ap.add_argument("--n-notsame", type=int, default=3)
    ap.add_argument("--chain-len", type=int, default=0,
                    help="if >0, path of same-constraints with exact depth")
    ap.add_argument("--num-loops", type=int, default=1)
    ap.add_argument("--loop-gate", action="store_true")
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
    ap.add_argument("--depth", type=int, default=0,
                    help="override config num_layers (0 = keep config)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)
    print(f"[dp] device={device} ({describe(device)})", flush=True)

    from jaume.modeling.config import JaumeConfig
    from jaume.modeling.loop_lm import LoopLM
    from jaume.modeling.checkpoint import load_state_dict
    from jaume.data.dataset import load_tokenizer

    cfg = JaumeConfig.load(args.config)
    if args.depth > 0:
        cfg.num_layers = args.depth
    tokdir = os.path.join(REPO, "jaume_tokenizer")
    tokenizer = load_tokenizer(tokdir)
    tokenizer.add_tokens(["[MASK]"])
    cfg.vocab_size = len(tokenizer)
    cfg.num_loops = args.num_loops
    cfg.loop_gate = args.loop_gate
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
        print(f"[dp] warm start {len(loaded)} tensors", flush=True)
    print(f"[dp] params={model.num_parameters()/1e6:.1f}M "
          f"depth={model.backbone.effective_depth}", flush=True)

    seed_all(device, args.seed)
    t0 = time.time()
    docs = build_docs(args.n_tasks, args.n_items, args.n_slots,
                      args.n_fixed, args.n_same, args.n_notsame, args.seed,
                      args.chain_len)
    print(f"[dp] {len(docs)} puzzles in {time.time()-t0:.0f}s", flush=True)
    enc = encode(docs, tokenizer, args.max_length)
    print(f"[dp] {len(enc)} encoded", flush=True)

    L = max(e["len"] for e in enc)
    ids_all = torch.full((len(enc), L), cfg.pad_token_id, dtype=torch.long)
    s1_all = torch.zeros(len(enc), dtype=torch.long)
    s2_all = torch.zeros(len(enc), dtype=torch.long)
    len_all = torch.zeros(len(enc), dtype=torch.long)
    for k, e in enumerate(enc):
        ids_all[k, :e["len"]] = torch.tensor(e["ids"], dtype=torch.long)
        s1_all[k] = e["start1"]
        s2_all[k] = e["start2"]
        len_all[k] = e["len"]
    print(f"[dp] dense {tuple(ids_all.shape)} "
          f"({ids_all.numel()*ids_all.element_size()/1e6:.0f} MB)", flush=True)

    n_val = args.val_docs
    ids_all_dev = ids_all.to(device)
    s1_dev = s1_all.to(device)
    s2_dev = s2_all.to(device)
    len_dev = len_all.to(device)
    train_idx = torch.arange(n_val, len(enc), device=device)
    val_idx = torch.arange(0, n_val, device=device)
    print(f"[dp] train={len(train_idx)} val={len(val_idx)}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1,
                            betas=(0.9, 0.95))
    steps_per_epoch = max(1, len(train_idx) // (args.batch_size * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs

    def lr_at(step):
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        p = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
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
            ids = ids_all_dev.index_select(0, sel)
            logits, aux = model(ids)
            loss = staged_loss(logits, ids, s1_dev.index_select(0, sel),
                               s2_dev.index_select(0, sel),
                               len_dev.index_select(0, sel))
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
                    print(f"[dp] epoch {ep} step {step}/{total_steps} "
                          f"loss {float(loss.detach()):.4f}", flush=True)
                if args.eval_every and step % args.eval_every == 0:
                    vl = evaluate(model, ids_all_dev, s1_dev, s2_dev, len_dev,
                                  val_idx)
                    gs = ""
                    gp = getattr(model.backbone, "loop_gate_param", None)
                    if gp is not None:
                        gs = " gate[" + ",".join(f"{float(v):.2f}"
                                                 for v in gp.detach()) + "]"
                    print(f"[dp] epoch {ep} step {step} VAL {vl:.4f}{gs}",
                          flush=True)

    vl = evaluate(model, ids_all_dev, s1_dev, s2_dev, len_dev, val_idx)
    gs = ""
    gp = getattr(model.backbone, "loop_gate_param", None)
    if gp is not None:
        gs = " gate[" + ",".join(f"{float(v):.2f}" for v in gp.detach()) + "]"
    print(f"[dp] FINAL VAL {vl:.4f}{gs}", flush=True)

    os.makedirs(args.output, exist_ok=True)
    cfg.save(os.path.join(args.output, "config.json"))
    out = {"backbone." + k: v for k, v in model.backbone.state_dict().items()}
    out["lm_head.weight"] = model.lm_head.weight.detach().cpu()
    torch.save(out, os.path.join(args.output, "pytorch_model.bin"))
    try:
        tokenizer.save_pretrained(os.path.join(args.output, "tokenizer"))
    except Exception:
        pass
    print(f"[dp] saved -> {args.output} ({time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
