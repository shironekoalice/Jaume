# Jaume Series Overview

> 中文版：[docs/overview.md](overview.md)
>
> Training progress and weights are published on ModelScope: [LightNeko/Jaume](https://modelscope.cn/models/LightNeko/Jaume) (this repo). Code mirror: [modemneko/Jaume](https://github.com/modemneko/Jaume).

Jaume is a family of lightweight, agent-oriented models. Each member is an independent model addressing one direction; members do not have to share a backbone. They are hosted in this single repository and organized by file layout.

## Family Members

| Member | Direction | Status | Docs |
|---|---|---|---|
| **Jaume-Embedding** | Text semantic representation + structured decision (Choice / Score / Noul) | v0.1 released (350M full pipeline); next-generation MLM base training in progress | [SPEC.md](../SPEC.md) · [PLAN.md](../PLAN.md) |
| **Jaume-Memory** | Latent memory-slot pooling (JM-S) and cross-input memory (JM-R) | **Evaluation complete; both mechanisms failed their preregistered criteria** | [Technical report](memory-report.en.md) · [Experiment plan](memory-plan.md) |
| **Neuro-Memory** (planned) | Brain-inspired hippocampal direction: structured relational memory and structural encoding (cf. Hippoformer / mm-TEM) | Direction defined, experiments not started | [Direction and first-round plan](neuro-memory-spec.md) |
| **Jaume-Temporal** | Temporal skeleton estimation and control (working name) | Task definition | [docs/temporal-spec.md](temporal-spec.md) |

**Standalone experiments outside the series above**:

| Experiment | Direction | Status | Docs |
|---|---|---|---|
| metamem | External memory write policy: deciding "what is worth remembering" with no query in sight (frozen backbone + classifier head, no architecture change) | **Paused, no valid results** (encoding aborted; to be rerun on a server) | [Parked record](memory-metamem-parked.md) · [Preregistration](memory-metamem-spec.md) |

> Grouping rule: Jaume-Memory studies the model's **internal** neural state channel; Neuro-Memory studies how **brain-inspired structure** carries relational memory; metamem studies **external** write-selection policy. Different questions, so they are not merged.


## Jaume-Embedding v0.1 (only released member)

Architecture: Bidirectional Transformer Encoder (GQA + RoPE + RMSNorm + MoE), Pooling → Projection → L2 normalize into unit vectors; Choice / Score / Noul decision heads attached to the same encoder representation — one encoding, multiple decisions.

### Released checkpoints (350M family)

| Checkpoint | Notes |
|---|---|
| `350m-mlm/` | Stage 1 MLM base (EN/ZH), loss 11.9 → 4.06 |
| `350m-mlm-v2/` | Multilingual continued MLM (+ DE/ES/TR/AR/PL/FR wiki) |
| `350m-stage2/` | Contrastive training (from `350m-mlm`); STS-B 0.6111 @512D |
| `350m-stage2-ml/` | Multilingual contrastive (from `350m-mlm-v2`) |
| `350m-stage3/` | Hard-negative retraining; **STS-B 0.6223 @512D / 0.6108 @128D**, STS22 0.3554 |
| `350m-stage3-ml/` | Multilingual hard-negative retraining; STS-B 0.6214 / 0.6107 @128D, STS22 0.3581 |
| `350m-decision-v2/` | Decision heads (on `350m-stage3`): Choice 99.75% (ECE 0.0017), Noul AUROC 0.837, Score Pearson 0.29 (weak) |

> STS scores above come from the historical training pipeline whose evaluation sets overlap with training; independent holdout results will follow the protocol in [docs/memory-plan.md](memory-plan.md).

The 700M model is not trained yet; all current experiments run at 350M scale first.

## Repository Layout

```text
├── README.md            Model card (Jaume-Embedding)
├── docs/overview.md     Series overview (this file; EN copy: overview.en.md)
├── SPEC.md              Jaume-Embedding architecture spec
├── PLAN.md              Jaume-Embedding training plan & log
├── docs/                Member-level planning docs (memory / temporal)
├── configs/             Model configs
├── jaume/               Python package (modeling / losses / data / inference)
├── jaume_tokenizer/     Loftey 151646-vocab BPE
├── scripts/             Training / eval / mining / sync scripts
├── tests/               Unit tests
├── 350m-*/              Released checkpoints
├── data/                Training & evaluation data
├── logs/                Training logs & TensorBoard
└── code/                Code snapshots (packed with checkpoints)
```

## Hardware Notes (RTX 3080 Ti 12GB)

- 350M + bf16 + gradient checkpointing + batch 8: ~6.3GB VRAM, 12 pairs/s (max_length 128)
- GradCache (chunk 96, no checkpointing): effective batch 256, 52 pairs/s, 17.2GB (on 22GB card)
- Single-GPU contrastive learning needs gradient accumulation or GradCache for large effective batch; multi-GPU for scale-up
