"""Stage 1 (causal twin): MLM pretraining of the CAUSAL Jaume encoder.

This script is deliberately a separate file, not a flag on train_mlm.py, so
that the bidirectional and causal training pipelines cannot be confused:

  * it refuses to run unless the config declares `"causal": true`;
  * it refuses to run unless the config is tagged `"jaume-causal"`;
  * it refuses to write into an output directory whose name lacks "causal";
  * it logs under the `[causal-mlm]` prefix and its own TensorBoard dir.

Controlled-variable contract
----------------------------
The attention form must be the ONLY difference between the two twins. This
script therefore reuses the *exact* classes of the bidirectional pipeline --
`TextDataset`, `MaskedCollator`, `MLMHead` are imported from train_mlm.py
rather than re-implemented -- so the data pipeline, the 80/10/10 masking
scheme and the projection-head loss are identical by construction.

Note on the objective: the causal twin is trained with the same BERT-style
MLM objective as the bidirectional base, NOT with next-token prediction.
That keeps the training objective a controlled constant. The consequence is
recorded in docs/ttc-spec.md: the twin learns to fill in masked positions,
so it is not an autoregressive LM and cannot be used for token-level
on-policy distillation without a separate spec.

Usage:
    python scripts/train_mlm_causal.py \
        --config configs/jaume-350m-causal.json \
        --data ../../data/mlm_corpus.jsonl \
        --output output/350m-causal-mlm --max-samples 300000
"""

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # scripts/
sys.path.insert(0, REPO)                                        # jaume/

from jaume.device import resolve_device, has_amp, empty_cache, seed_all, describe
from jaume.modeling.config import JaumeConfig
from jaume.modeling.embedding import JaumeEmbeddingModel
from jaume.inference.encode import save_pretrained

# Reused verbatim from the bidirectional pipeline -- see controlled-variable
# contract in the module docstring.
from train_mlm import TextDataset, MaskedCollator, MLMHead

LOG = "[causal-mlm]"


def _step_dirs(output):
    """All step-* checkpoints in the output dir, newest first."""
    if not os.path.isdir(output):
        return []
    dirs = [d for d in os.listdir(output)
            if d.startswith("step-") and os.path.isdir(os.path.join(output, d))]
    return sorted(dirs, key=lambda d: int(d.split("-", 1)[1]), reverse=True)


def resolve_resume(args):
    """Return the checkpoint dir to resume from, or None."""
    if not args.resume:
        return None
    if args.resume == "auto":
        candidates = _step_dirs(args.output)
        if not candidates:
            raise SystemExit(
                f"{LOG} --resume auto found no step-* checkpoint in {args.output}"
            )
        return os.path.join(args.output, candidates[0])
    if not os.path.isdir(args.resume):
        raise SystemExit(f"{LOG} resume dir not found: {args.resume}")
    return args.resume


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
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--device", default=None)
    p.add_argument("--loss-chunk", type=int, default=0,
                   help="materialise the MLM logits in chunks of this many "
                        "masked positions (0 = one shot). The full projection "
                        "is [n_masked x 151646] in fp32: 745 MB forward, the "
                        "same again inside cross_entropy, and the same again "
                        "for the gradient -- about 2.2 GB for a batch of 48. "
                        "Chunking bounds that to chunk_size x 151646 x 4 B "
                        "at any moment.")
    p.add_argument("--resume", default=None,
                   help="checkpoint dir to resume from, or 'auto' to pick the "
                        "newest step-* inside --output")
    args = p.parse_args()

    # ---- anti-mix guards: refuse anything that is not the causal twin ----
    config = JaumeConfig.load(args.config)
    if not getattr(config, "causal", False):
        raise SystemExit(
            f"{LOG} REFUSING TO RUN: config {args.config} does not declare "
            "\"causal\": true. Use scripts/train_mlm.py for the bidirectional encoder."
        )
    if config.model_type != "jaume-causal":
        raise SystemExit(
            f"{LOG} REFUSING TO RUN: config model_type is {config.model_type!r}, "
            "expected 'jaume-causal'."
        )
    if "causal" not in os.path.basename(args.output.rstrip("/\\")).lower():
        raise SystemExit(
            f"{LOG} REFUSING TO RUN: output directory {args.output!r} must contain "
            "'causal' in its name so causal artefacts are never mixed with "
            "bidirectional ones."
        )

    device = resolve_device(args.device)
    seed_all(device, 42)
    print(f"{LOG} device={device} ({describe(device)})")
    print(f"{LOG} config={args.config} model_type={config.model_type} "
          f"causal={config.causal} layers={config.num_layers} "
          f"hidden={config.hidden_size} experts={config.expert_num}", flush=True)

    tokenizer_path = args.tokenizer or os.path.join(REPO, "jaume_tokenizer")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    num_added = tokenizer.add_tokens(["[MASK]"])

    config.vocab_size = len(tokenizer)  # + [MASK]
    model = JaumeEmbeddingModel(config).to(device)
    mlm_head = MLMHead(config.hidden_size, config.vocab_size).to(device)

    from jaume.modeling.backbone_causal import CausalBackbone
    assert isinstance(model.backbone, CausalBackbone), (
        f"{LOG} backbone is {type(model.backbone).__name__}, expected CausalBackbone"
    )
    print(f"{LOG} backbone={type(model.backbone).__name__} OK; "
          f"vocab resized to {config.vocab_size} (+{num_added} [MASK])", flush=True)

    dataset = TextDataset(args.data, args.max_samples)
    collator = MaskedCollator(tokenizer, max_length=args.max_length)

    # ---- resume: restore model + optimizer + scheduler + data position ----
    resume_dir = resolve_resume(args)
    start_step = 0
    samples_seen = 0
    if resume_dir:
        state_path = os.path.join(resume_dir, "trainer_state.pt")
        if not os.path.exists(state_path):
            raise SystemExit(
                f"{LOG} {resume_dir} has no trainer_state.pt -- it was not written "
                "by a resume-capable run; cannot resume safely."
            )
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(torch.load(
            os.path.join(resume_dir, "pytorch_model.bin"),
            map_location="cpu", weights_only=True))
        print(f"{LOG} resumed model weights from {resume_dir}", flush=True)
        start_step = int(state["global_step"])
        samples_seen = int(state["samples_seen"])
        batch_at_save = int(state["batch_size"])
        accum_at_save = int(state["grad_accum"])
        # The corpus order is fixed (TextDataset shuffles with seed 42), so the
        # resumed run must re-walk exactly the same examples.
        skip = samples_seen
        if skip > len(dataset):
            raise SystemExit(
                f"{LOG} checkpoint consumed {skip} samples but the corpus only "
                f"has {len(dataset)} -- resume would be inconsistent."
            )
        dataset = torch.utils.data.Subset(dataset, range(skip, len(dataset)))
        print(f"{LOG} resume: step {start_step}, {samples_seen} samples already "
              f"consumed (batch {batch_at_save} x accum {accum_at_save}), "
              f"{len(dataset)} samples left", flush=True)
    else:
        state = None

    # shuffle=False on purpose: TextDataset already fixes the corpus order
    # with seed 42, so this makes the exact training sequence deterministic
    # and identical between the two twins (the controlled-variable contract),
    # and it is what makes --resume reproduce the same order instead of
    # reshuffling the remaining rows.
    # A dedicated generator for the DataLoader. PyTorch draws a `base_seed`
    # from the GLOBAL RNG every time a loader iterator is created:
    #     self._base_seed = torch.empty((), dtype=torch.int64).random_(generator=loader.generator).item()
    # The iterator is created once per epoch, so an uninterrupted run consumes
    # that draw at epoch start while a resumed run would consume it again
    # after restoring the RNG state -- shifting the whole MLM mask sequence.
    # Passing an explicit generator keeps the draw OUT of the global RNG, which
    # makes --resume reproduce the uninterrupted run bit-for-bit, and also
    # makes the data-loading path identical between the two twins.
    loader_generator = torch.Generator()
    loader_generator.manual_seed(42)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collator, num_workers=args.num_workers, drop_last=True,
                        generator=loader_generator)
    print(f"{LOG} {len(dataset)} texts, {len(loader)} steps/epoch", flush=True)

    params = list(model.parameters()) + list(mlm_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: min(1.0, (s + 1) / max(1, args.warmup_steps)),
    )

    if state is not None:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        head_state = state.get("mlm_head_state")
        if head_state is not None:
            mlm_head.load_state_dict(head_state)
            print(f"{LOG} restored MLM head weights", flush=True)
        else:
            print(f"{LOG} WARNING: checkpoint has no MLM head weights; the "
                  "head would be randomly re-initialised and the run would "
                  "not match the uninterrupted one", flush=True)
        rng = state.get("torch_rng_state")
        if rng is not None:
            torch.set_rng_state(rng)
            print(f"{LOG} restored torch RNG state (mask sequence continues)", flush=True)
        else:
            print(f"{LOG} WARNING: checkpoint has no torch RNG state; the mask "
                  "sequence will differ from the uninterrupted run", flush=True)
        print(f"{LOG} restored optimizer + scheduler state", flush=True)

    if args.grad_ckpt:
        model.backbone.gradient_checkpointing_enable()
    model.train()
    mlm_head.train()
    os.makedirs(args.output, exist_ok=True)
    global_step = start_step
    accum, running, t0 = 0, 0.0, time.time()
    tokens_seen = 0
    from torch.utils.tensorboard import SummaryWriter

    tb = SummaryWriter(log_dir=os.path.join(args.output, "tb"))

    for epoch in range(args.epochs):
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            tokens_seen += int(mask.sum())        # real tokens (padding excluded)
            samples_seen += int(mask.shape[0])    # corpus rows consumed

            with torch.autocast(device, dtype=torch.bfloat16, enabled=has_amp(device)):
                hidden, aux = model.backbone(input_ids, mask)
                # identical to the bidirectional pipeline: project only the
                # ~15% masked positions, because the 151k-vocab output is the
                # bottleneck and unmasked logits contribute nothing
                sel = labels.view(-1) != -100
                # Packing the masked positions cannot use boolean indexing on
                # every backend: Ascend lowers `x[mask]` to `aclnnNonzeroV2`,
                # which faults the vector core in this CANN build. The
                # gather-then-filter below reaches the same tensor without any
                # data-dependent index generation -- sort the mask, take the
                # first n_sel rows, and reorder the labels by the same index.
                n_sel = int(sel.sum().item())
                if hidden.device.type == "npu":
                    order = torch.argsort(sel.to(torch.int8), descending=True)
                    h_packed = hidden.view(-1, config.hidden_size)[order]
                    l_packed = labels.view(-1)[order]
                    h_sel = h_packed[:n_sel]
                    l_sel = l_packed[:n_sel]
                else:
                    h_sel = hidden.view(-1, config.hidden_size)[sel]
                    l_sel = labels.view(-1)[sel]

            chunk = args.loss_chunk
            scale = float(args.grad_accum)

            if chunk and chunk > 0 and n_sel > chunk:
                # Chunked MLM loss.
                #
                # The projection is [n_masked x 151646]. In fp32 that is 745 MB
                # for a batch of 48 -- once for the logits, once for the
                # log-softmax inside cross_entropy, and once for the gradient:
                # roughly 2.2 GB for the loss alone, which is what pushed the
                # batch size down.
                #
                # Each chunk goes through torch.utils.checkpoint, so the chunk's
                # logits are NOT kept alive between the forward and the
                # backward pass; the head is recomputed on demand during
                # backward and only one chunk's logits exist at a time. The
                # chunks are summed and back-propagated once, so the backbone
                # graph is traversed a single time.
                # The whole head+loss must be inside the checkpointed
                # function. Wrapping only `mlm_head` would leave the fp32 copy
                # of the logits and the log-softmax alive in the autograd
                # graph, which is most of the memory the chunking is meant to
                # save.
                def _chunk_loss(h_c, l_c):
                    return F.cross_entropy(
                        mlm_head(h_c).float(), l_c, reduction="sum")

                losses = []
                for c0 in range(0, n_sel, chunk):
                    losses.append(torch.utils.checkpoint.checkpoint(
                        _chunk_loss, h_sel[c0:c0 + chunk], l_sel[c0:c0 + chunk],
                        use_reentrant=False))
                loss_c = torch.stack(losses).sum() / n_sel
                del losses
            else:
                logits = mlm_head(h_sel)
                loss_c = F.cross_entropy(
                    logits.float(), l_sel,
                    ignore_index=-100,
                )
                del logits

            # The MoE load-balance term must be part of the SAME backward
            # pass. A second .backward() would traverse the backbone graph
            # again after the first one already released it
            # ("Trying to backward through the graph a second time").
            if aux is not None and aux.requires_grad:
                loss_c = loss_c + config.load_balance_weight * aux

            (loss_c / scale).backward()
            loss_val = float(loss_c.detach())

            running += loss_val
            accum += 1
            if accum == args.grad_accum:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                accum = 0
                global_step += 1
                if global_step % args.log_every == 0:
                    dt = time.time() - t0
                    print(f"{LOG} epoch {epoch} step {global_step} "
                          f"loss {running / (args.log_every * args.grad_accum):.4f} "
                          f"lr {scheduler.get_last_lr()[0]:.2e} "
                          f"tokens/s {tokens_seen / dt:.0f}", flush=True)
                    tb.add_scalar("train/loss", running / (args.log_every * args.grad_accum), global_step)
                    tb.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
                    tb.add_scalar("train/tokens_per_s", tokens_seen / dt, global_step)
                    running = 0.0

                if args.save_every and global_step % args.save_every == 0:
                    ckpt_dir = os.path.join(args.output, f"step-{global_step}")
                    save_pretrained(model, tokenizer, ckpt_dir)
                    torch.save({
                        "global_step": global_step,
                        "samples_seen": samples_seen,
                        "batch_size": args.batch_size,
                        "grad_accum": args.grad_accum,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        # torch RNG state: the MLM mask pattern (15% selection
                        # plus the 80/10/10 replace decision) is drawn from the
                        # global RNG, so without this a resumed run would
                        # continue with a different mask sequence.
                        "torch_rng_state": torch.get_rng_state(),
                        # The MLM head is a separate nn.Module and is NOT part
                        # of the model's state_dict, so it has to be stored
                        # explicitly -- otherwise a resumed run would start
                        # from a freshly random head and the loss trajectory
                        # would diverge from the uninterrupted one.
                        "mlm_head_state": mlm_head.state_dict(),
                    }, os.path.join(ckpt_dir, "trainer_state.pt"))
                    print(f"{LOG} checkpoint saved at step {global_step} "
                          f"({samples_seen} samples consumed)", flush=True)

    os.makedirs(args.output, exist_ok=True)
    save_pretrained(model, tokenizer, os.path.join(args.output, "final"))
    print(f"{LOG} done in {time.time()-t0:.0f}s, saved to {args.output}/final", flush=True)


if __name__ == "__main__":
    main()
