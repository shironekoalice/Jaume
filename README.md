---
frameworks:
- pytorch
tasks:
- sentence-embedding
- image-text-to-text
- object-detection
- ocr
- text-to-speech
---
# Jaume

A model family investigating what small models can and cannot do. Each member
answers one question; most answers turned out negative.

> Weights and logs: [LightNeko/Jaume](https://modelscope.cn/models/LightNeko/Jaume)
> Code: [shironekoalice/Jaume](https://github.com/shironekoalice/Jaume)
> Chinese overview: [docs/overview.md](docs/overview.md)

## Running on Ascend

Development ran on **Ascend 910B3 (64 GiB HBM)** with `torch_npu` / CANN. The
Actor direction is being validated on **Ascend 950PR**.

| Script | What it does |
|---|---|
| `code/repo/scripts/check_npu_env.py` | pre-flight: device, memory, SDPA mask support, bf16, tokens/s |
| `code/repo/scripts/setup_ascend.sh` | CANN environment setup |
| `tools/npu-ssh.ps1`, `tools/npu_probe.sh` | remote orchestration (tunnel host, port and key are local configuration, not in the repo) |
| `prototype/actor/run_npu.sh` | the Actor pipeline on NPU: `check` / `probe` / `ui` / `maze` |

The Actor's two stages map onto the hardware directly: stage 1 (locate) is one
forward pass over a single screenshot, and stage 2 (confirm) is K small OCR
calls. Stage 2 is throughput-bound rather than compute-bound, which is what
makes the newest Ascend hardware the interesting target for it.

`docs/ascend-950-application.md` holds the full 950PR application write-up,
including the measured numbers each claim rests on.

**The one result that transfers:** a representation either contains the
information a task needs or it does not. A three-minute linear probe on the
frozen encoder answers that before any training run, and it predicted every
result below.

## Architecture

One family, five questions, different backbones.

```
Jaume
├── Embedding  bidirectional 350M MoE  semantic + choice/score/noul
├── Memory     bidirectional 350M MoE  slot pooling / carried state
├── TTC        causal 350M             multi-round inference
├── Loop       causal 350M             depth by repetition
├── Vis        causal 350M + vision    one model, ten visual capabilities
└── Actor      small + vision          one forward per decision, perception + solver
```

### Jaume-Actor (prototyping)

> Full record: [docs/jaume-actor-prototype.md](docs/jaume-actor-prototype.md) ·
> Backbone selection: [docs/jaume-actor-backbone-selection.md](docs/jaume-actor-backbone-selection.md) ·
> Handoff guide: [docs/jaume-actor-handoff.md](docs/jaume-actor-handoff.md) ·
> Ascend 950 application: [docs/ascend-950-application.md](docs/ascend-950-application.md) ·
> Code: [prototype/actor/](prototype/actor/)

A browser/game agent that decides in **one forward pass** and never decodes an
action string. Five findings, all measured:

| Finding | Evidence |
|---|---|
| Score a shortlist of candidates, do not regress coordinates | candidate scoring reaches 1.000 on real-browser grounding (20/20) and 1.000 in closed-loop control (6/6) |
| Give planning to a solver, not to the network | perception + BFS **1.000** closed-loop (3/3 seeds) vs a learned pixel policy 0.890 |
| Any input path that bypasses vision breeds shortcut solutions | coordinate channels drop the worst seed from 1.0000 to 0.0213; head init scale 10 drops it to 0.0192 |
| A 93M vision tower beats a 2B VLM for grounding | on real ScreenSpot with a linear probe (held-out): SigLIP2-base **0.850** vs Qwen3-VL-2B 0.806, at 1/22 the parameters and 11x faster per crop |
| The instruction must modulate the visual features, not just be concatenated | with real UI distractors: gated head **0.648** vs linear probe 0.185 vs gated-without-instruction 0.183 (floor 0.167) |

The collapsed solution is always "look at nothing". The backbone was chosen
on measured data rather than family loyalty, and the answer was not the one
the project started with. Both were only visible because of controls: a
grounding benchmark whose "no instruction" control also scores 1.000 is
measuring nothing.

The maze detector and the UI actor are reported with the **worst seed**,
never a single run.

**Reproduce it.** Every number above comes from a script in
`prototype/actor/`, and `python prototype/actor/check_consistency.py`
re-checks all 52 of them against the json files they came from:

```bash
cd prototype/actor
python fetch_models.py                                    # weights (ModelScope)
python train_gated_head.py --rows 300                     # train the gated scorer
python real_distractor.py --rows 400 --k 6 --epochs 12     # the decisive comparison
python single_crop_ocr.py                                 # OCR capability, one crop at a time
python eval_browser_gated.py --ckpt gated_head.pt         # real DOM, real Chrome
python check_consistency.py                               # 52 checks
```

The same scripts run on Ascend NPU through `prototype/actor/run_npu.sh`
(`check` / `probe` / `ui` / `maze`).

### Jaume-Vis (the most complete member)

```
image ─┬─> SigLIP2  (frozen, 224px)    ─> 196 tokens ─┐
       └─> TrOCR    (frozen, 384px)    ─> 577 tokens ─┴─> gate(q) ─┐
                                                                   │
                        question ──────────────────────────────────┤
                                                                   ▼
                                         Jaume-Causal 350M (top-4 unfrozen)
                                                                   │
                                         answer (autoregressive decoding)
```

Three choices carry the results:

| Choice | Why |
|---|---|
| Two towers | Semantics and glyphs need different encoders. A single tower is 3.5x worse at reading. |
| Question-conditioned gate | Mixes the streams per task, keeps both raw streams as residuals. Without it OCR drops 0.89 -> 0.65. |
| Coordinate tokens | Counting and pointing decode grid indices instead of inferring position. |

## Results

All numbers below come from the protocol declared with each table. Frozen
towers mean the vision encoders never receive gradient; only the projectors,
the capability heads, and the top 4 Jaume layers train.

### Jaume-Vis capabilities (one 350M checkpoint)

Evaluated after S3c: 15 epochs, 1500 OCR + 1200 vision + 2500 FUNSD samples
per epoch, batch 8, lr 4e-4 (projectors) / 4e-5 (backbone).

| Capability | Score | Metric | Test set |
|---|---|---|---|
| Semantic understanding | **0.956** | accuracy | synthetic suite, 6 families x 200 |
| Real image classification | **0.954** | accuracy | CIFAR-10 official 10k test |
| OCR localisation | **0.970** | exact string | synthetic, 4-way, 200 samples |
| OCR reading | 0.791 | exact string | synthetic, 4-char codes, 200 |
| Document field | 0.612 | exact string | synthetic invoices, 200 |
| Real document reading | 0.281 | normalised exact | FUNSD, 300 samples |
| Real document field type | 0.493 | accuracy | FUNSD, 4-way, 300 |
| Object counting | **0.910** | exact string | 1-3 objects, 500 samples |
| Object detection | 0.347 | IoU>0.3 hit rate | 1-3 objects, 8x8 grid, 500 |
| Object pointing | 0.645 | +-1 cell hit rate | 1-3 objects, 8x8 grid, 500 |

Notes on interpretation:

- **CIFAR-10** uses the official test split with no training overlap.
- **FUNSD** answers are case-folded and whitespace-collapsed before comparison,
  because exact character match penalises formatting rather than reading.
- **Object detection** is scored as the fraction of ground-truth boxes whose
  best IoU with any prediction exceeds 0.3. Exact string match on the
  coordinate digits gives 0.058 for the same predictions; it measures digit
  agreement, not localisation.
- **Object pointing** counts a prediction as correct if it falls within one
  8x8 grid cell of a ground-truth centre. Exact match gives 0.109.
- The 8x8 grid bounds detection IoU: even a perfect predictor achieves only
  0.763 at IoU>0.3 because of quantisation. A 16x16 grid lifts that bound to
  0.999.

### Negative results

| Direction | Comparison | Result | Protocol |
|---|---|---|---|
| Loop, equal compute | C (10k, loop-4) vs A (40k, base) | **-0.99 nats** | identical wall-clock |
| Loop, two-stage derivation | loop-4 vs base at 4x compute | -0.0006 (tie) | identical tokens seen |
| Depth scan | 8L vs 16L | **all 8 task structures saturated** | identical data and steps |
| 8-digit addition | 8L vs 16L | 0.8900 vs 0.8967 | exact match, 300 test |
| SigLIP2 glyph probe | 4-way text location | 0.210 | chance = 0.250 |
| InternViT-448 glyph probe | 4-way text location | 0.209 | chance = 0.250 |
| TrOCR glyph probe | 4-way text location | **0.750** | chance = 0.250 |
| Point localisation metric | exact vs +-1 cell | 0.109 vs **0.504** | identical predictions |
| Box detection metric | exact vs IoU>0.3 | 0.058 vs 0.158 | identical predictions |
| Memory JM-S vs control | SICK-R Spearman | -0.0168 | approx 3x seed SD |
| Memory JM-R carried vs static | F1 | -0.0785 | 5/5 runs negative |
| TTC oracle gap | oracle vs single-pass | 0.711 vs 0.350 | coverage 0.751 |
| Planning closed loop | fine-tuned top-1 vs beam success | 0.475 vs **0.000** | imitation-trained scorer |

### The decisive experiment: 8-digit addition

Loop papers cite multi-digit arithmetic as the depth task. We made it as
favourable as possible:

- operands are 8 digits, at least one carry is guaranteed per example
- the answer appears nowhere in the input
- scoring is exact match, so one wrong digit scores zero
- both depths warm-start from the same 16-layer causal checkpoint
- 20000 training / 300 test, 8 epochs, identical hyperparameters

| Depth | Best exact match | Train time |
|---|---|---|
| 8L | 0.8900 | 810 s |
| 16L | 0.8967 | 1459 s |

16L gains 0.0067 for 1.8x the time. The gain is inside seed noise, and the
task is saturated at 8 layers.

### Multi-task conflict and the fix

Ablation on the same data and schedule:

| Setup | OCR mean | Vision mean | CIFAR |
|---|---|---|---|
| Single tower, mixed tasks | 0.650 | 0.950 | 0.958 |
| Dual tower + question gate | **0.887** | **0.913** | **0.954** |

The gate is a sigmoid over a question embedding; both raw streams stay in the
sequence as residuals. Removing either the gate or the residual path
reproduces the single-tower collapse.

Supporting diagnostic: training OCR alone from the collapsed checkpoint
recovers only 0.34 on text location, and training semantics alone wipes OCR
to 0.000 - the conflict is representation overwrite, not gradient noise.

### Hardware

**Ascend 910B3, 64 GiB HBM.**

### Infrastructure

| Item | Value | Method |
|---|---|---|
| Context verified | **16384** tokens | window loss flat from 1024 to 16384 |
| Safe training length | 4096 | 8192 OOMs (explicit attention, 64 GiB) |
| KV cache equivalence | max diff 1e-6 | cached vs full forward, same weights |
| KV cache speedup (350M) | 1.0x | latency-bound: T=1 154 ms vs T=64 197 ms |
| 350M LM throughput | ~0.5 step/s | batch 24x2, len 192, AICore 7-15% |
| VLM epoch time | 150-280 s | 9k samples, frozen towers + top-4 |

The KV cache result is a correctness win and a long-context requirement, not a
speed win at this size. Attention is only about 20% of decode time at 350M, so
removing it barely moves wall-clock.

## Research map

| Member | Status | Where |
|---|---|---|
| Embedding | v0.1 released | `SPEC.md`, `PLAN.md` |
| Memory | Closed, both mechanisms failed | `docs/memory-report.en.md`, `docs/memory-preregistration.md` |
| TTC | Closed, all channels failed | `docs/ttc-spec.md` |
| Planning | Closed, closed-loop fails | `docs/planning-report.md` |
| Loop | Closed, 8 task structures saturate | `docs/loop-spec.md` |
| **Vis** | **Stage 1-3 complete** | `docs/vlm-dual-tower-spec.md`, `docs/jaume-vis-training-spec.md` |
| Temporal | Spec only | `docs/temporal-spec.md` |
| **Actor** | **Prototype, positive results** | `docs/jaume-actor-prototype.md`, `prototype/actor/` |

## Papers

| Paper | Format |
|---|---|
| Jaume-Vis: three negative results + one protocol | [HTML](docs/paper-vis/paper.html) · [LaTeX](docs/paper-vis/paper.tex) |
| Why a finite-capacity neural memory does not help | [PDF](docs/paper/Jaume-Memory-Technical-Report.pdf) · [LaTeX](docs/paper/paper.tex) |
| Depth saturation predicts when looping cannot help | [HTML](docs/paper-loop/paper.html) · [LaTeX](docs/paper-loop/paper.tex) |

## Checkpoints

| Path | Notes |
|---|---|
| `350m-mlm/` | Stage 1 MLM base (EN/ZH) |
| `350m-mlm-v2/` | Multilingual continued MLM |
| `350m-stage2/`, `350m-stage2-ml/` | Contrastive, STS-B 0.6111 @512D |
| `350m-stage3/`, `350m-stage3-ml/` | Hard negatives, STS-B 0.6223 / 0.6108 @128D |
| `350m-decision-v2/` | Decision heads: choice 99.75%, noul AUROC 0.837 |
| `350m-causal-mlm/final/` | Causal twin, used by all Vis experiments |

STS scores come from a historical pipeline with train/eval overlap; treat them
as pipeline checks, not held-out results. The 700M model has not been trained.

## Layout

```
docs/            research records, one file per direction
  paper-vis/     unified paper (HTML + LaTeX)
  paper-loop/    loop-only draft
  paper/         memory paper
code/repo/       package: jaume/modeling, scripts, tests
tools/           remote orchestration
350m-*/          checkpoints
```