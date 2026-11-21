"""Train the Jaume Omni adapters on precomputed SenseVoice/Mimi features.

This is the NPU process. It intentionally does not import funasr, torchaudio
or Mimi: all audio work happens in scripts/prepare_omni_features.py, so a
broken CPU audio ABI cannot take down a multi-hour training run.

Smoke:
    python scripts/train_omni.py --config configs/omni-qwen3vl-2b.json       --manifest /mnt/workspace/data/omni_smoke/train.jsonl       --output /mnt/workspace/output/omni-smoke --max-steps 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from jaume.device import resolve_device, describe, seed_all
from jaume.omni.config import OmniConfig
from jaume.omni.data import OmniFeatureDataset, collate_omni
from jaume.omni.model import JaumeOmniModel

LOG = "[omni]"


def _cpu_state(module):
    return {k: v.detach().cpu() for k, v in module.state_dict().items()}


def _to_cpu(value):
    """Recursively copy tensors to CPU for portable checkpoints."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(v) for v in value)
    return value


def _to_cpu(value):
    """Recursively copy tensors to CPU for portable checkpoints."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(v) for v in value)
    return value


def _checkpoint_state(path: str) -> dict:
    return torch.load(
        os.path.join(path, "omni_trainable.pt"), map_location="cpu",
        weights_only=True,
    )


def save_checkpoint(model: JaumeOmniModel, optimizer, path: str, step: int) -> None:
    """Save every trainable parameter plus optimizer state.

    Saving only the audio adapters silently drops the fine-tuned Thinker
    layers. The frozen 2B backbone is still referenced by config.thinker_path,
    so this stays much smaller than duplicating the full base model.
    """
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(model.config.to_dict(), f, indent=2, ensure_ascii=False)
    trainable = {
        name: p.detach().cpu()
        for name, p in model.named_parameters()
        if p.requires_grad
    }
    state = {
        "format_version": 2,
        "step": step,
        "trainable": trainable,
        "optimizer": _to_cpu(optimizer.state_dict()) if optimizer is not None else None,
    }
    torch.save(state, os.path.join(path, "omni_trainable.pt"))


def load_checkpoint(model: JaumeOmniModel, path: str) -> dict:
    """Restore trainable parameters; return the raw state for later opt load.

    Format v1 (adapters only) remains readable. It cannot restore unfrozen
    Thinker layers because those were never stored.
    """
    state = _checkpoint_state(path)
    if "trainable" in state:
        current = {name: p for name, p in model.named_parameters() if p.requires_grad}
        missing = sorted(set(current) - set(state["trainable"]))
        unexpected = sorted(set(state["trainable"]) - set(current))
        if missing or unexpected:
            raise RuntimeError(
                f"checkpoint parameter mismatch: missing={missing[:5]} "
                f"unexpected={unexpected[:5]}"
            )
        with torch.no_grad():
            for name, tensor in state["trainable"].items():
                current[name].copy_(tensor.to(current[name].device))
    else:
        model.audio_projector.load_state_dict(state["audio_projector"])
        model.bridge.load_state_dict(state["bridge"])
        model.codec_start.data.copy_(state["codec_start"])
        model.talker.load_state_dict(state["talker"])
        model.codebook_head.load_state_dict(state["codebook_head"])
        print(f"{LOG} warning: legacy adapter-only checkpoint; Thinker layers not restored",
              flush=True)
    return state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--max-text-length", type=int, default=128)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = OmniConfig.from_dict(json.load(f))
    device = resolve_device(args.device)
    seed_all(device, 1234)
    print(f"{LOG} device={device} ({describe(device)})", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(config.thinker_path, trust_remote_code=True)
    ds = OmniFeatureDataset(
        args.manifest, text_tokenizer=tokenizer,
        max_text_length=args.max_text_length,
        audio_dim=config.audio_hidden_size,
        num_codebooks=config.codec_num_codebooks,
    )
    print(f"{LOG} samples={len(ds)}", flush=True)

    model = JaumeOmniModel(config, load_thinker=True).to(device)
    model.train()
    resume_state = None
    if args.resume:
        resume_state = load_checkpoint(model, args.resume)
        step0 = int(resume_state.get("step", 0))
        print(f"{LOG} resumed at step {step0}", flush=True)
    else:
        step0 = 0

    trainable = model.trainable_parameters()
    adapter = (
        list(model.audio_projector.parameters())
        + list(model.bridge.parameters())
        + list(model.talker.parameters())
        + list(model.codebook_head.parameters())
    )
    n_adapter = sum(p.numel() for p in adapter)
    n_thinker = sum(p.numel() for p in model.thinker.parameters() if p.requires_grad)
    print(
        f"{LOG} trainable_total={(n_adapter+n_thinker)/1e6:.2f}M "
        f"adapter={n_adapter/1e6:.2f}M thinker_unfrozen={n_thinker/1e6:.2f}M",
        flush=True,
    )
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    if resume_state is not None and resume_state.get("optimizer") is not None:
        opt.load_state_dict(resume_state["optimizer"])
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=lambda b: collate_omni(b, audio_dim=config.audio_hidden_size),
    )

    step = step0
    accum = 0
    t0 = time.time()
    running = {"loss": 0.0, "text": 0.0, "audio": 0.0}
    n_log = 0
    stop = False
    for epoch in range(args.epochs):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            audio_features = batch["audio_features"]
            input_ids = batch["input_ids"]
            text_labels = input_ids.clone()
            text_labels[~batch["text_mask"]] = -100
            attention_mask = torch.cat([batch["text_mask"], batch["audio_input_mask"]], dim=1).long()
            total, losses = model(
                input_ids=input_ids,
                audio_features=audio_features,
                attention_mask=attention_mask,
                audio_input_mask=batch["audio_input_mask"],
                text_labels=text_labels,
                text_segment_len=input_ids.shape[1],
                audio_codes=batch["audio_codes"],
                audio_codes_mask=batch["audio_codes_mask"],
            )
            (total / args.grad_accum).backward()
            accum += 1
            running["loss"] += float(total.detach())
            running["text"] += float(losses.get("text", torch.tensor(0.0)))
            running["audio"] += float(losses.get("audio", torch.tensor(0.0)))
            n_log += 1
            if accum == args.grad_accum:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                accum = 0
                step += 1
                if step % args.log_every == 0:
                    dt = max(time.time() - t0, 1e-6)
                    print(
                        f"{LOG} step={step} loss={running['loss']/n_log:.4f} "
                        f"text={running['text']/n_log:.4f} audio={running['audio']/n_log:.4f} "
                        f"lr={opt.param_groups[0]['lr']:.2e} samples/s={n_log/dt:.2f} "
                        f"peak={torch.npu.max_memory_allocated()/2**30:.2f}GiB"
                        if device == "npu" else
                        f"{LOG} step={step} loss={running['loss']/n_log:.4f} "
                        f"text={running['text']/n_log:.4f} audio={running['audio']/n_log:.4f}",
                        flush=True,
                    )
                    running = {"loss": 0.0, "text": 0.0, "audio": 0.0}
                    n_log = 0
                    t0 = time.time()
                if args.save_every and step % args.save_every == 0:
                    save_checkpoint(model, opt, os.path.join(args.output, f"step-{step}"), step)
            if args.max_steps and step >= args.max_steps:
                stop = True
                break
        if stop:
            break

    save_checkpoint(model, opt, os.path.join(args.output, "final"), step)
    print(f"{LOG} done step={step} -> {args.output}/final", flush=True)


if __name__ == "__main__":
    main()
