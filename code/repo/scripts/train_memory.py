"""Jaume-Memory stage-01 trainer: one script, all arms (B0-B6, E1-E9).

Fairness rules from docs/memory-plan.md §3, enforced here:
  * every arm starts from the SAME MLM base checkpoint (--init-from);
  * same data, optimiser, steps, max_length and seed (seed is required);
  * MoE stays on but its aux weight is kept constant and is NOT ablated;
  * the extra memory parameters are trained with the same optimiser.

Usage:
    python scripts/train_memory.py \
        --config configs/jaume-350m.json --pooling latent_memory --num-slots 8 \
        --init-from 350m-stage3 --train-data data/all_nli.jsonl \
        --output output/mem/B3-latent --seed 1 --epochs 1
"""
import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jaume.device import resolve_device, has_amp, make_scaler, empty_cache, max_memory_allocated, reset_peak_memory_stats, total_memory, seed_all

from jaume.modeling.config import JaumeConfig
from jaume.modeling.embedding import JaumeEmbeddingModel
from jaume.modeling.memory import SlotDiversityLoss
from jaume.losses.contrastive import MultipleNegativesRankingLoss, MatryoshkaLoss
from jaume.losses.gradcache import gradcache_step
from jaume.data.dataset import PairDataset, load_tokenizer
from jaume.data.collator import PairCollator
from jaume.inference.encode import save_pretrained

MEMORY_POOLINGS = (
    "mean", "cls", "attention", "attention_mlp",
    "latent_memory", "recurrent_memory", "recurrent_memory_gate",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--train-data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--init-from", required=True,
                   help="checkpoint dir; the base is IDENTICAL for every arm")
    p.add_argument("--pooling", required=True, choices=MEMORY_POOLINGS)
    p.add_argument("--num-slots", type=int, default=8)
    p.add_argument("--slot-heads", type=int, default=8)
    p.add_argument("--slot-pool", default="mean", choices=["mean", "attention"])
    p.add_argument("--attn-mlp-width", type=int, default=2048,
                   help="width of the parameter-matched B2 scorer")
    p.add_argument("--no-learn-slots", action="store_true", help="E2: fixed random slots")
    p.add_argument("--slot-interaction", action="store_true", help="E4: slot self-attention")
    p.add_argument("--slot-competition", action="store_true",
                   help="Slot Attention (Locatello 2020, arXiv:2006.15055): softmax over slots")
    p.add_argument("--slot-iterations", type=int, default=1,
                   help="iterative slot refinement rounds (Slot Attention uses 3)")
    p.add_argument("--no-slot-gru", action="store_true", help="plain residual slot update")
    p.add_argument("--slot-div-weight", type=float, default=0.0,
                   help="anti-collapse diversity loss weight on slot states")
    p.add_argument("--memory-output", default="new_slots",
                   choices=["new_slots", "carry"],
                   help="experiment A: pool the UPDATED state ('carry') instead "
                        "of this chunk's fresh slots ('new_slots'). Only "
                        "differs for gated recurrent arms.")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--auto-batch", action="store_true",
                   help="grow the micro-batch until VRAM is nearly full "
                        "(keeps batch_size*grad_accum constant by folding the "
                        "accumulation into the batch)")
    p.add_argument("--vram-frac", type=float, default=0.92,
                   help="target fraction of total VRAM the auto-batch may use")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--use-matryoshka", action="store_true")
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--gc-chunk", type=int, default=0)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    device = resolve_device()
    seed_all(device, args.seed)

    # the base checkpoint's own config is authoritative (MLM resizes the vocab)
    config = JaumeConfig.load(os.path.join(args.init_from, "config.json"))
    config.pooling = args.pooling
    config.memory_output = args.memory_output
    config.num_slots = args.num_slots
    config.slot_heads = args.slot_heads
    config.slot_pool = args.slot_pool
    config.learn_slots = not args.no_learn_slots
    config.slot_interaction = args.slot_interaction
    config.attn_mlp_width = args.attn_mlp_width
    config.slot_competition = args.slot_competition
    config.slot_iterations = args.slot_iterations
    config.slot_gru = not args.no_slot_gru

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tokenizer_path = args.tokenizer or os.path.join(repo_root, "jaume_tokenizer")
    tokenizer = load_tokenizer(tokenizer_path)

    model = JaumeEmbeddingModel(config).to(device)

    # load the shared base; the pooling head is arm-specific and is either
    # absent from the base (fresh init) or shape-compatible (classic arms).
    state = torch.load(
        os.path.join(args.init_from, "pytorch_model.bin"),
        map_location="cpu", weights_only=True,
    )
    model_state = model.state_dict()
    loaded, fresh = [], []
    for k, v in model_state.items():
        src = state.get(k)
        if src is not None and tuple(src.shape) == tuple(v.shape):
            v.copy_(src)
            loaded.append(k)
        else:
            fresh.append(k)
    print(f"[memory] base tensors reused: {len(loaded)}  freshly initialised: {len(fresh)}")
    if fresh:
        print(f"[memory]   fresh: {fresh[:12]}{' ...' if len(fresh) > 12 else ''}")

    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[memory] arm={args.pooling} seed={args.seed} params={n_params/1e6:.1f}M "
          f"trainable={n_train/1e6:.1f}M device={device}")

    base_loss = MultipleNegativesRankingLoss()
    loss_fn = MatryoshkaLoss(base_loss, config.matryoshka_dims) if args.use_matryoshka else base_loss
    if args.use_matryoshka:
        print(f"[memory] matryoshka dims: {config.matryoshka_dims}")
    div_fn = SlotDiversityLoss(args.slot_div_weight)
    use_state = args.slot_div_weight > 0
    if use_state:
        print(f"[memory] slot diversity loss weight = {args.slot_div_weight}")

    dataset = PairDataset(args.train_data, max_samples=args.max_samples)
    collator = PairCollator(tokenizer, max_length=args.max_length)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collator, num_workers=args.num_workers, drop_last=True,
    )
    print(f"[memory] training pairs: {len(dataset)}  steps/epoch: {len(loader)//args.grad_accum}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = make_scaler(device, enabled=args.gc_chunk == 0)
    if args.gc_chunk:
        args.grad_accum = 1
        print(f"[memory] gradcache enabled, chunk {args.gc_chunk} pairs")

    if args.auto_batch and has_amp(device):
        os.makedirs(args.output, exist_ok=True)
        eff = args.batch_size * args.grad_accum
        total = total_memory(device)
        budget = int(total * args.vram_frac)
        print(f"[memory] auto-batch: effective batch {eff}, VRAM budget "
              f"{budget/1024**2:.0f}MiB of {total/1024**2:.0f}MiB")
        cand = 32
        while cand * 2 <= eff:
            cand *= 2
        best = args.batch_size
        while True:
            try:
                empty_cache(device)
                reset_peak_memory_stats(device)
                probe = next(iter(loader)) if cand == args.batch_size else None
                if probe is None:
                    idx = torch.randperm(len(dataset))[:cand].tolist()
                    probe = collator([dataset[i] for i in idx])
                ids = probe["query_input_ids"].to(device)
                mask = probe["query_attention_mask"].to(device)
                with torch.autocast(device, dtype=torch.bfloat16, enabled=has_amp(device) and args.gc_chunk == 0):
                    q = model(ids, mask)[0]
                    p = model(ids, mask)[0]
                    loss = loss_fn(q, p)
                loss.backward()
                optimizer.zero_grad(set_to_none=True)
                peak = max_memory_allocated(device)
                del ids, mask, probe
                if peak > budget and cand > 8:
                    print(f"[memory]   bs={cand:4d} peak {peak/1024**2:6.0f}MiB > budget, stopping")
                    break
                print(f"[memory]   bs={cand:4d} peak {peak/1024**2:6.0f}MiB OK")
                best = cand
                if cand >= eff:
                    # never probe beyond the effective batch: the accumulated
                    # gradient is identical, so a larger micro-batch buys
                    # nothing and only risks OOM
                    break
                cand *= 2
            except (torch.OutOfMemoryError, RuntimeError) as exc:
                is_oom = isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()
                if not is_oom:
                    raise
                print(f"[memory]   bs={cand:4d} OOM, stopping")
                optimizer.zero_grad(set_to_none=True)
                empty_cache(device)
                break
            except StopIteration:
                break
        final = max(args.batch_size, best)
        print(f"[memory] auto-batch selected micro-batch {final} "
              f"(effective {final * args.grad_accum} pairs/step)")
        if final != args.batch_size:
            args.batch_size = final
            loader = DataLoader(
                dataset, batch_size=args.batch_size, shuffle=True,
                collate_fn=collator, num_workers=args.num_workers, drop_last=True,
            )
        model.zero_grad(set_to_none=True)
        empty_cache(device)
        reset_peak_memory_stats(device)
        print(f"[memory] training pairs: {len(dataset)}  "
              f"steps/epoch: {len(loader)//args.grad_accum}")
        with open(os.path.join(args.output, "run_config.json"), "w", encoding="utf-8") as f:
            json.dump({"args": vars(args), "config": config.to_dict()}, f,
                      indent=2, ensure_ascii=False)

    def lr_lambda(step):
        return step / max(1, args.warmup_steps) if step < args.warmup_steps else 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "config": config.to_dict()}, f, indent=2, ensure_ascii=False)

    from torch.utils.tensorboard import SummaryWriter

    tb = SummaryWriter(log_dir=os.path.join(args.output, "tb"))
    pad_id = config.pad_token_id

    def pad_cat(*pairs):
        max_len = max(ids.size(1) for ids, _ in pairs)
        ids_out, mask_out = [], []
        for ids, mask in pairs:
            pad = max_len - ids.size(1)
            if pad > 0:
                ids = torch.nn.functional.pad(ids, (0, pad), value=pad_id)
                mask = torch.nn.functional.pad(mask, (0, pad), value=0)
            ids_out.append(ids)
            mask_out.append(mask)
        return torch.cat(ids_out), torch.cat(mask_out)

    global_step = 0
    accum = 0
    t0 = time.time()
    running = 0.0
    peak_mb = 0.0
    if args.grad_ckpt:
        model.backbone.gradient_checkpointing_enable()
        print("[memory] gradient checkpointing enabled")
    model.train()

    for epoch in range(args.epochs):
        for batch in loader:
            input_ids = batch["query_input_ids"].to(device)
            mask = batch["query_attention_mask"].to(device)
            p_ids = batch["passage_input_ids"].to(device)
            p_mask = batch["passage_attention_mask"].to(device)
            has_hard = "hard_negative_input_ids" in batch
            if has_hard:
                n_ids = batch["hard_negative_input_ids"].to(device)
                n_mask = batch["hard_negative_attention_mask"].to(device)
                h = batch["num_hard_negatives"]

            if args.gc_chunk:
                loss = gradcache_step(
                    model, loss_fn, input_ids, mask, p_ids, p_mask,
                    n_ids if has_hard else None,
                    n_mask if has_hard else None,
                    num_hard=h if has_hard else 0,
                    chunk_size=args.gc_chunk,
                    load_balance_weight=config.load_balance_weight,
                    device_type=device,
                )
                running += loss.item()
            else:
                with torch.autocast(device, dtype=torch.bfloat16, enabled=has_amp(device)):
                    if has_hard:
                        all_ids, all_mask = pad_cat(
                            (input_ids, mask), (p_ids, p_mask), (n_ids, n_mask),
                        )
                        q_all, q_aux = model(all_ids, all_mask)
                        b = input_ids.size(0)
                        q_emb = q_all[:b]
                        p_emb = q_all[b : b + b]
                        n_emb = q_all[b + b :].view(b, h, -1)
                    else:
                        if use_state:
                            q_emb, q_aux, q_state = model(input_ids, mask, return_state=True)
                            p_emb, _, p_state = model(p_ids, p_mask, return_state=True)
                        else:
                            q_emb, q_aux = model(input_ids, mask)
                            p_emb, _ = model(p_ids, p_mask)
                            q_state = p_state = None
                        n_emb = None
                    loss = loss_fn(q_emb, p_emb, n_emb)
                    loss = loss + config.load_balance_weight * q_aux
                    if use_state:
                        loss = loss + div_fn(q_state) + div_fn(p_state)

                scaler.scale(loss / args.grad_accum).backward()
                running += loss.item()
            accum += 1

            if accum == args.grad_accum:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                accum = 0
                global_step += 1

                if has_amp(device):
                    peak_mb = max(peak_mb, max_memory_allocated(device) / 1024 ** 2)

                if global_step % args.log_every == 0:
                    el = time.time() - t0
                    speed = global_step * args.grad_accum * args.batch_size / el
                    print(f"[memory] epoch {epoch} step {global_step} "
                          f"loss {running / (args.log_every * args.grad_accum):.4f} "
                          f"lr {scheduler.get_last_lr()[0]:.2e} pairs/s {speed:.0f}")
                    tb.add_scalar("train/loss", running / (args.log_every * args.grad_accum), global_step)
                    tb.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
                    tb.add_scalar("train/pairs_per_s", speed, global_step)
                    running = 0.0

                if args.save_every and global_step % args.save_every == 0:
                    save_pretrained(model, tokenizer, os.path.join(args.output, f"step-{global_step}"))

    save_pretrained(model, tokenizer, os.path.join(args.output, "final"))
    elapsed = time.time() - t0
    summary = {
        "arm": args.pooling, "seed": args.seed, "params_M": round(n_params / 1e6, 2),
        "steps": global_step, "seconds": round(elapsed, 1),
        "pairs_per_s": round(global_step * args.grad_accum * args.batch_size / elapsed, 2),
        "peak_vram_mb": round(peak_mb, 1),
    }
    with open(os.path.join(args.output, "train_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[memory] done in {elapsed:.0f}s -> {args.output}/final")
    print(f"[memory] summary: {summary}")


if __name__ == "__main__":
    main()
