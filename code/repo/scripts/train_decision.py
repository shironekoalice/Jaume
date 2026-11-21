"""Stage 5: train Choice / Score / Noul heads on the encoder, then
calibrate (Stage 6 metrics).

Data (all derived from real datasets, single "state" text per example):
- noul:  NLI entailment -> 1, contradiction -> 0, state = "premise || hypothesis"
- score: STS 0-5 gold score, state = "sentence1 || sentence2" (STS-B + STS22)
- choice: STS22 language ID, state = sentence1 (monolingual rows only)

All three tasks share ONE encoder forward per step (SPEC §23). By default
the encoder is frozen and only the heads train; --unfreeze-layers N also
fine-tunes the top N backbone layers with a lower learning rate.

Usage:
    python scripts/train_decision.py \
        --model output/350m-stage3-hardneg/final \
        --noul-data /root/autodl-tmp/data/all_nli.jsonl \
        --score-data /root/autodl-tmp/data/stsb.jsonl,/root/autodl-tmp/data/sts22-crosslingual-sts.jsonl \
        --choice-data /root/autodl-tmp/data/sts22-crosslingual-sts.jsonl \
        --output output/350m-decision --epochs 2 \
        --head-hidden 512 --unfreeze-layers 4
"""
import argparse
import json
import os
import random
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jaume.device import resolve_device, has_amp, make_scaler, empty_cache, max_memory_allocated, reset_peak_memory_stats, total_memory, seed_all

from jaume.losses.calibration import calibration_report, TemperatureScaler
from jaume.modeling.config import JaumeConfig
from jaume.modeling.decision import JaumeDecisionModel
from jaume.data.dataset import load_tokenizer

SEP = " || "


def load_noul(path, max_samples):
    """Entailment -> 1, contradiction -> 0, class-balanced."""
    pos, neg = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            text = f"{row['premise']}{SEP}{row['hypothesis']}"
            if row["label"] == 0:
                pos.append((text, 1))
            elif row["label"] == 2:
                neg.append((text, 0))
    random.shuffle(pos)
    random.shuffle(neg)
    n = max_samples // 2 if max_samples else None
    pairs = pos[:n] + neg[:n]
    random.shuffle(pairs)
    return pairs


def load_score(paths, max_samples):
    pairs = []
    per = max_samples // len(paths) if max_samples else None
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if "sentence1" not in row:
                    continue
                pairs.append((f"{row['sentence1']}{SEP}{row['sentence2']}",
                              float(row["score"])))
                if per and len(pairs) >= per:
                    break
    return pairs


def load_choice(path, max_samples):
    """Monolingual STS22 rows: sentence1 -> language class."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if "-" not in row["lang"]:
                rows.append((row["sentence1"], row["lang"]))
    random.shuffle(rows)
    counts = {}
    for _, lang in rows:
        counts[lang] = counts.get(lang, 0) + 1
    langs = sorted(counts, key=counts.get, reverse=True)
    lang_to_idx = {lang: i for i, lang in enumerate(langs)}
    rows = [(text, lang_to_idx[lang]) for text, lang in rows]
    return rows[:max_samples] if max_samples else rows, langs


def split_train_val(pairs, val_ratio=0.1, seed=42):
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    return shuffled[n_val:], shuffled[:n_val]


class DecisionTrainer:
    def __init__(self, model, tokenizer, device, options):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.options = options
        self.frozen = options.unfreeze_layers == 0
        if not self.frozen:
            self.model.backbone.gradient_checkpointing_enable()

    def _pooled(self, texts, batch_size=128):
        """Shared pooled states; one forward serves all heads.

        Frozen mode: computed under no_grad (heads-only training).
        Unfrozen mode: full graph with gradient checkpointing.
        """
        outs = []
        self.model.train()
        if self.frozen:
            with torch.no_grad():
                for i in range(0, len(texts), batch_size):
                    enc = self.tokenizer(
                        texts[i : i + batch_size], padding=True, truncation=True,
                        max_length=self.options.max_length, return_tensors="pt",
                    ).to(self.device)
                    hidden, _ = self.model.backbone(enc["input_ids"], enc["attention_mask"])
                    outs.append(self.model.pooling(hidden, enc["attention_mask"]))
        else:
            for i in range(0, len(texts), batch_size):
                enc = self.tokenizer(
                    texts[i : i + batch_size], padding=True, truncation=True,
                    max_length=self.options.max_length, return_tensors="pt",
                ).to(self.device)
                hidden, _ = self.model.backbone(enc["input_ids"], enc["attention_mask"])
                outs.append(self.model.pooling(hidden, enc["attention_mask"]))
        pooled = torch.cat(outs)
        return pooled.detach() if self.frozen else pooled

    def _head_logits(self, pooled):
        ch = self.model.heads["category"]
        choice = ch.proj(ch.trunk(pooled))
        score = self.model.heads["importance"].proj(
            self.model.heads["importance"].trunk(pooled)
        ).squeeze(-1)
        noul = self.model.heads["relevance"].proj(
            self.model.heads["relevance"].trunk(pooled)
        ).squeeze(-1)
        return choice, score, noul

    def train(self, data, epochs, batch_size, lr_head, lr_backbone):
        choice, score, noul = data["choice"], data["score"], data["noul"]
        head_params = [p for n, p in self.model.named_parameters()
                       if p.requires_grad and n.startswith("heads")]
        backbone_params = [p for n, p in self.model.named_parameters()
                           if p.requires_grad and not n.startswith("heads")]
        opt = torch.optim.AdamW([
            {"params": head_params, "lr": lr_head},
            {"params": backbone_params, "lr": lr_backbone},
        ])
        n = max(len(choice["train"]), len(score["train"]), len(noul["train"]))
        steps_per_epoch = max(1, n // batch_size)
        rng = random.Random(0)
        for epoch in range(epochs):
            total, count = 0.0, 0
            for step in range(steps_per_epoch):
                opt.zero_grad(set_to_none=True)
                loss = 0.0
                for name, dataset in (("choice", choice), ("score", score), ("noul", noul)):
                    rows = dataset["train"]
                    batch = rng.sample(rows, min(batch_size, len(rows)))
                    pooled = self._pooled([t for t, _ in batch])
                    ch_logits, sc_raw, no_logit = self._head_logits(pooled)
                    if name == "choice":
                        labels = torch.tensor([l for _, l in batch], device=self.device)
                        loss = loss + F.cross_entropy(ch_logits, labels)
                    elif name == "score":
                        target = torch.tensor([v for _, v in batch], device=self.device)
                        pred = torch.sigmoid(sc_raw) * self.options.score_levels
                        loss = loss + F.mse_loss(pred, target)
                    else:
                        labels = torch.tensor([float(l) for _, l in batch], device=self.device)
                        loss = loss + F.binary_cross_entropy_with_logits(no_logit, labels)
                loss.backward()
                opt.step()
                total += loss.item()
                count += 1
                if (step + 1) % 50 == 0:
                    print(f"[decision] epoch {epoch} step {step+1}/{steps_per_epoch} "
                          f"loss {total/count:.4f}", flush=True)
            print(f"[decision] epoch {epoch} mean loss {total/max(1,count):.4f}", flush=True)

    @torch.no_grad()
    def evaluate(self, data):
        self.model.heads.eval()
        self.model.backbone.eval()
        report = {}

        rows = data["noul"]["val"]
        pooled = self._pooled([t for t, _ in rows])
        _, _, no_logit = self._head_logits(pooled)
        labels = torch.tensor([l for _, l in rows])
        probs = torch.sigmoid(no_logit.cpu())
        report["noul"] = calibration_report(torch.stack([1 - probs, probs], dim=1), labels)
        scaler = TemperatureScaler()
        binary_logits = torch.stack([-no_logit.cpu(), no_logit.cpu()], dim=1)
        scaler.fit(binary_logits, labels)
        probs_t = scaler(binary_logits)[:, 1]
        report["noul"]["ece_after_temp"] = calibration_report(
            torch.stack([1 - probs_t, probs_t], dim=1), labels
        )["ece"]
        report["noul"]["temperature"] = scaler.temperature

        rows = data["choice"]["val"]
        pooled = self._pooled([t for t, _ in rows])
        ch_logits, _, _ = self._head_logits(pooled)
        labels = torch.tensor([l for _, l in rows])
        probs = F.softmax(ch_logits.cpu(), dim=-1)
        report["choice"] = calibration_report(probs, labels)
        cscaler = TemperatureScaler()
        cscaler.fit(ch_logits.cpu(), labels)
        report["choice"]["accuracy"] = probs.argmax(-1).eq(labels).float().mean().item()
        report["choice"]["ece_after_temp"] = calibration_report(
            cscaler(ch_logits.cpu()), labels
        )["ece"]
        report["choice"]["temperature"] = cscaler.temperature

        rows = data["score"]["val"]
        pooled = self._pooled([t for t, _ in rows])
        _, sc_raw, _ = self._head_logits(pooled)
        pred = torch.sigmoid(sc_raw.cpu()) * self.options.score_levels
        target = torch.tensor([v for _, v in rows])
        report["score"] = {
            "mae": (pred - target).abs().mean().item(),
            "pearson": torch.corrcoef(torch.stack([pred, target]))[0, 1].item(),
        }
        return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="frozen encoder checkpoint dir")
    p.add_argument("--noul-data", required=True)
    p.add_argument("--score-data", required=True,
                   help="comma-separated jsonl paths with sentence1/sentence2/score")
    p.add_argument("--choice-data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3, help="head lr")
    p.add_argument("--lr-backbone", type=float, default=1e-5)
    p.add_argument("--unfreeze-layers", type=int, default=0)
    p.add_argument("--head-hidden", type=int, default=0,
                   help=">0: 2-layer MLP heads with this hidden size")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--score-levels", type=int, default=5)
    p.add_argument("--max-noul", type=int, default=20000)
    p.add_argument("--max-choice", type=int, default=None)
    p.add_argument("--max-score", type=int, default=None)
    args = p.parse_args()
    random.seed(42)
    device = resolve_device()

    config = JaumeConfig.load(os.path.join(args.model, "config.json"))
    decisions = {
        "category": {"type": "choice", "options": None},
        "importance": {"type": "score", "levels": args.score_levels},
        "relevance": {"type": "noul"},
    }
    if args.head_hidden > 0:
        for spec in decisions.values():
            spec["head_hidden"] = args.head_hidden

    noul_pairs = load_noul(args.noul_data, args.max_noul)
    score_pairs = load_score(args.score_data.split(","), args.max_score)
    choice_rows, langs = load_choice(args.choice_data, args.max_choice)
    print(f"[decision] noul {len(noul_pairs)}  score {len(score_pairs)}  "
          f"choice {len(choice_rows)} ({len(langs)} languages: {langs[:8]})", flush=True)
    decisions["category"]["options"] = langs

    model = JaumeDecisionModel(config, decisions)
    state = torch.load(os.path.join(args.model, "pytorch_model.bin"),
                       map_location="cpu", weights_only=True)
    encoder_state = {k: v for k, v in state.items()
                     if k.startswith(("backbone", "pooling"))}
    _, unexpected = model.load_state_dict(encoder_state, strict=False)
    assert not unexpected, unexpected
    print(f"[decision] loaded encoder ({len(encoder_state)} tensors)", flush=True)

    if args.unfreeze_layers > 0:
        n = config.num_layers
        for layer in model.backbone.layers[: n - args.unfreeze_layers]:
            for param in layer.parameters():
                param.requires_grad_(False)
        for param in model.backbone.embed_tokens.parameters():
            param.requires_grad_(False)
    else:
        for param in model.backbone.parameters():
            param.requires_grad_(False)
    for param in model.pooling.parameters():
        param.requires_grad_(False)
    n_train = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    print(f"[decision] trainable params: {n_train/1e6:.2f}M "
          f"(unfreeze top {args.unfreeze_layers} layers)", flush=True)

    data = {
        "noul": dict(zip(("train", "val"), split_train_val(noul_pairs))),
        "score": dict(zip(("train", "val"), split_train_val(score_pairs))),
        "choice": dict(zip(("train", "val"), split_train_val(choice_rows))),
    }

    tokenizer = load_tokenizer(os.path.join(args.model, "tokenizer"))
    trainer = DecisionTrainer(model, tokenizer, device, args)
    trainer.train(data, args.epochs, args.batch_size, args.lr, args.lr_backbone)

    report = trainer.evaluate(data)
    print("[decision] calibration report:")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    os.makedirs(args.output, exist_ok=True)
    config.save(os.path.join(args.output, "config.json"))
    torch.save(model.state_dict(), os.path.join(args.output, "pytorch_model.bin"))
    tokenizer.save_pretrained(os.path.join(args.output, "tokenizer"))
    with open(os.path.join(args.output, "decisions.json"), "w", encoding="utf-8") as f:
        json.dump({"decisions": decisions, "report": report}, f, ensure_ascii=False, indent=2)
    print(f"[decision] saved to {args.output}")


if __name__ == "__main__":
    main()
