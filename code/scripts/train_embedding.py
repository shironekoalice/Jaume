"""Stage 2 embedding training: contrastive (MNRL) with Matryoshka support.

Usage:
    python scripts/train_embedding.py \
        --config configs/jaume-smoke.json \
        --train-data /root/autodl-tmp/data/stsb.jsonl \
        --output output/smoke \
        --epochs 1 --batch-size 32 --grad-accum 4
"""
import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jaume.modeling.config import JaumeConfig
from jaume.modeling.embedding import JaumeEmbeddingModel
from jaume.losses.contrastive import MultipleNegativesRankingLoss, MatryoshkaLoss
from jaume.losses.gradcache import gradcache_step
from jaume.data.dataset import PairDataset, load_tokenizer
from jaume.data.collator import PairCollator
from jaume.inference.encode import save_pretrained


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--tokenizer", default=None, help="defaults to <repo>/jaume_tokenizer")
    p.add_argument("--train-data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--use-matryoshka", action="store_true")
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--init-from", default=None, help="checkpoint dir to start from")
    p.add_argument("--gc-chunk", type=int, default=0,
                   help="GradCache pair-chunk size; enables large effective "
                        "batch with bounded activation memory (0 = off)")
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--log-every", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    config = JaumeConfig.load(args.config)
    if args.init_from:
        # the checkpoint's own config is authoritative (e.g. MLM-resized vocab)
        ckpt_config = os.path.join(args.init_from, "config.json")
        if os.path.exists(ckpt_config):
            config = JaumeConfig.load(ckpt_config)
            print(f"[jaume] config loaded from {ckpt_config}")
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tokenizer_path = args.tokenizer or os.path.join(repo_root, "jaume_tokenizer")
    tokenizer = load_tokenizer(tokenizer_path)

    model = JaumeEmbeddingModel(config).to(device)
    if args.init_from:
        state = torch.load(
            os.path.join(args.init_from, "pytorch_model.bin"),
            map_location="cpu", weights_only=True,
        )
        model.load_state_dict(state, strict=True)
        print(f"[jaume] initialized from {args.init_from}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[jaume] params: {n_params/1e6:.1f}M  device: {device}")
    print(f"[jaume] config: hidden={config.hidden_size} layers={config.num_layers} "
          f"experts={config.expert_num}x top{config.topk} emb={config.embedding_dim}D "
          f"pooling={config.pooling}")

    base_loss = MultipleNegativesRankingLoss()
    if args.use_matryoshka:
        loss_fn = MatryoshkaLoss(base_loss, config.matryoshka_dims)
        print(f"[jaume] matryoshka dims: {config.matryoshka_dims}")
    else:
        loss_fn = base_loss

    dataset = PairDataset(args.train_data, max_samples=args.max_samples)
    collator = PairCollator(tokenizer, max_length=args.max_length)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collator, num_workers=args.num_workers, drop_last=True,
    )
    print(f"[jaume] training pairs: {len(dataset)}  steps/epoch: {len(loader)//args.grad_accum}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    # GradCache produces raw (unscaled) gradients; bf16 does not need loss scaling
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda" and args.gc_chunk == 0)
    if args.gc_chunk:
        args.grad_accum = 1
        print(f"[jaume] gradcache enabled, chunk {args.gc_chunk} pairs")

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    os.makedirs(args.output, exist_ok=True)
    from torch.utils.tensorboard import SummaryWriter

    tb = SummaryWriter(log_dir=os.path.join(args.output, "tb"))
    pad_id = config.pad_token_id

    def pad_cat(*pairs):
        """Pads (input_ids, mask) groups to a common length and concatenates."""
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
    if args.grad_ckpt:
        model.backbone.gradient_checkpointing_enable()
        print("[jaume] gradient checkpointing enabled")
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
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
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
                        q_emb, q_aux = model(input_ids, mask)
                        p_emb, _ = model(p_ids, p_mask)
                        n_emb = None
                    loss = loss_fn(q_emb, p_emb, n_emb)
                    loss = loss + config.load_balance_weight * q_aux

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

                if global_step % args.log_every == 0:
                    speed = global_step * args.grad_accum * args.batch_size / (time.time() - t0)
                    print(f"[jaume] epoch {epoch} step {global_step} "
                          f"loss {running / (args.log_every * args.grad_accum):.4f} "
                          f"lr {scheduler.get_last_lr()[0]:.2e} pairs/s {speed:.0f}")
                    tb.add_scalar("train/loss", running / (args.log_every * args.grad_accum), global_step)
                    tb.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
                    tb.add_scalar("train/pairs_per_s", speed, global_step)
                    running = 0.0

                if args.save_every and global_step % args.save_every == 0:
                    save_pretrained(model, tokenizer, os.path.join(args.output, f"step-{global_step}"))

    save_pretrained(model, tokenizer, os.path.join(args.output, "final"))
    print(f"[jaume] done in {time.time()-t0:.0f}s, saved to {args.output}/final")


if __name__ == "__main__":
    main()
