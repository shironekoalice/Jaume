"""Stage 1: MLM (denoising) pretraining of the bidirectional encoder.

Standard 15% masking with 80/10/10 replace/random/keep. The Qwen-derived
tokenizer has no [MASK] token, so one is added and the embedding table is
resized. The checkpoint is saved in JaumeEmbeddingModel format (with a
fresh projection head) so Stage 2 can pick it up via --init-from.

Usage:
    python scripts/train_mlm.py \
        --config configs/jaume-350m.json \
        --data /root/autodl-tmp/data/mlm_corpus.jsonl \
        --output output/350m-mlm --max-samples 300000
"""
import argparse
import json
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jaume.modeling.config import JaumeConfig
from jaume.modeling.embedding import JaumeEmbeddingModel
from jaume.inference.encode import save_pretrained


class TextDataset(Dataset):
    def __init__(self, path, max_samples=None):
        texts = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                texts.append(json.loads(line)["text"])
                if max_samples and len(texts) >= max_samples:
                    break
        random.Random(42).shuffle(texts)
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


class MaskedCollator:
    """Tokenize + BERT-style 15% masking (80/10/10)."""

    def __init__(self, tokenizer, max_length=128, mask_prob=0.15):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prob = mask_prob
        self.mask_id = tokenizer.convert_tokens_to_ids("[MASK]")
        special = set(tokenizer.all_special_ids)
        self.candidate_ids = [i for i in range(tokenizer.vocab_size) if i not in special]

    def __call__(self, batch):
        enc = self.tokenizer(
            batch, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        input_ids = enc["input_ids"]
        special_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for sid in self.tokenizer.all_special_ids:
            special_mask |= input_ids == sid

        prob = torch.full(input_ids.shape, self.mask_prob)
        prob[special_mask] = 0.0
        masked = torch.bernoulli(prob).bool()

        labels = input_ids.clone()
        labels[~masked] = -100

        # of the selected 15%: 80% -> [MASK], 10% -> random, 10% -> keep
        r = torch.rand(input_ids.shape)
        to_random = masked & (r < 0.1)
        to_keep = masked & (r >= 0.1) & (r < 0.2)
        replaced = input_ids.clone()
        replaced[masked] = self.mask_id
        rand_idx = torch.randint(0, len(self.candidate_ids), (int(to_random.sum()),))
        replaced[to_random] = torch.tensor(self.candidate_ids, dtype=torch.long)[rand_idx]
        replaced[to_keep] = input_ids[to_keep]
        return {"input_ids": replaced, "attention_mask": enc["attention_mask"], "labels": labels}


class MLMHead(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states):
        return self.decoder(self.norm(F.gelu(self.dense(hidden_states))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--save-every", type=int, default=0,
                   help="save intermediate checkpoint every N optimizer steps")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--log-every", type=int, default=50)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tokenizer_path = args.tokenizer or os.path.join(repo_root, "jaume_tokenizer")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    num_added = tokenizer.add_tokens(["[MASK]"])

    config = JaumeConfig.load(args.config)
    config.vocab_size = len(tokenizer)  # + [MASK]
    model = JaumeEmbeddingModel(config).to(device)
    mlm_head = MLMHead(config.hidden_size, config.vocab_size).to(device)
    print(f"[mlm] vocab resized to {config.vocab_size} (+{num_added} [MASK])")

    dataset = TextDataset(args.data, args.max_samples)
    collator = MaskedCollator(tokenizer, max_length=args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collator, num_workers=args.num_workers, drop_last=True)
    print(f"[mlm] {len(dataset)} texts, {len(loader)} steps/epoch")

    params = list(model.parameters()) + list(mlm_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: min(1.0, (s + 1) / max(1, args.warmup_steps)),
    )

    if args.grad_ckpt:
        model.backbone.gradient_checkpointing_enable()
    model.train()
    mlm_head.train()
    os.makedirs(args.output, exist_ok=True)
    global_step, accum, running, t0 = 0, 0, 0.0, time.time()
    tokens_seen = 0
    from torch.utils.tensorboard import SummaryWriter

    tb = SummaryWriter(log_dir=os.path.join(args.output, "tb"))

    for epoch in range(args.epochs):
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            tokens_seen += int(mask.sum())

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                hidden, aux = model.backbone(input_ids, mask)
                # project only masked positions (~15%): the 151k-vocab output
                # is the bottleneck, and unmasked logits contribute nothing
                sel = labels.view(-1) != -100
                logits = mlm_head(hidden.view(-1, config.hidden_size)[sel])
                loss = F.cross_entropy(
                    logits.float(), labels.view(-1)[sel],
                    ignore_index=-100,
                )
                loss = loss + config.load_balance_weight * aux

            (loss / args.grad_accum).backward()
            running += loss.item()
            accum += 1
            if accum == args.grad_accum:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                accum = 0
                global_step += 1
                if global_step % args.log_every == 0:
                    dt = time.time() - t0
                    print(f"[mlm] epoch {epoch} step {global_step} "
                          f"loss {running / (args.log_every * args.grad_accum):.4f} "
                          f"lr {scheduler.get_last_lr()[0]:.2e} "
                          f"tokens/s {tokens_seen / dt:.0f}")
                    tb.add_scalar("train/loss", running / (args.log_every * args.grad_accum), global_step)
                    tb.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
                    tb.add_scalar("train/tokens_per_s", tokens_seen / dt, global_step)
                    running = 0.0

                if args.save_every and global_step % args.save_every == 0:
                    save_pretrained(model, tokenizer, os.path.join(args.output, f"step-{global_step}"))
                    print(f"[mlm] checkpoint saved at step {global_step}")

    os.makedirs(args.output, exist_ok=True)
    save_pretrained(model, tokenizer, os.path.join(args.output, "final"))
    print(f"[mlm] done in {time.time()-t0:.0f}s, saved to {args.output}/final")


if __name__ == "__main__":
    main()
