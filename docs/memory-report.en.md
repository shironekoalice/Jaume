# Jaume-Memory Technical Report: Negative Results and Methodology for Finite-Capacity Neural Memory

> Version 1.0 ｜ Compiled: 2026-10-22
> Testbed: Jaume-Embedding 350M (291.9M parameters), single RTX 4060 Ti 16GB
> Status: **Both JM-S and JM-R fail under their pre-registered criteria in the current implementation. Conclusions are frozen; no criterion will be adjusted to search for a positive result.**
> This document consolidates 27 process documents (`docs/memory-*.md`). The originals are retained as the traceable evidence chain.

---

## 0. Abstract

We test two neural memory mechanisms on a 350M bidirectional encoder:

- **JM-S**: replace conventional pooling with a set of learnable latent slots, aiming for fixed-length vectors that better carry long inputs;
- **JM-R**: carry slot state across input chunks, aiming for cross-chunk memory.

Under pre-registered criteria, **both fail**. Key measurements:

| Mechanism | Comparison | Result | Verdict |
|---|---|---|---|
| JM-S | B3 (memory slots) − B2mlp (param-matched pooling), SICK-R | **−0.0168** (~3× seed SD) | ❌ Fail |
| JM-R (block-level) | carried − static, frozen backbone | **−0.0785** (5/5 negative) | ❌ State unsuitable as block representation |
| JM-R (synthetic cross-chunk) | carried − reset | +0.331 (3/3 same sign, CI excludes 0) | ⚠️ Mechanism does work on synthetic task |
| JM-R (real text) | carried − shuffled, SciFact | 2/3 same sign, CI contains 0 | ❌ Below preregistered bar |
| JM-R (long document) | state norm and inter-block cosine | norm 37.2 → 205.3; cosine 0.99 | ❌ Structural collapse |

**The main contribution is not "what we built" but three reusable methodological results and four mechanism-level diagnostic findings** (§5, §6).

---

## 1. Research Question and Preregistration

### 1.1 Question

> Can cross-input memory on a finite-capacity neural state produce a measurable gain on evidence-localization / cross-chunk integration tasks?

### 1.2 Preregistered criterion (frozen in advance, never adjusted)

```
Memory is effective <=> carried > reset  AND  carried > shuffled
                       AND paired differences share sign across 3 seeds,
                       AND paper-clustered bootstrap 95% CI excludes 0
```

The nine-run protocol additionally fixed S1–S4, stating explicitly: **no extra seeds, no split changes, no metric changes in response to interim results.**

### 1.3 Control arms

| Arm | State | Purpose |
|---|---|---|
| `static` / `reset` | no state passed | no-memory baseline (identical code path in this implementation, §5.2) |
| `carried` | true preceding state | the memory gain under test |
| `shuffled` | another example's state | rules out a generic-bias artifact |
| `concat` | both chunks encoded at once | reference for task learnability (information-availability ceiling) |
| `prev_only` | previous chunk only | one-step memory |

**Decision logic**: `carried` must beat both `reset` (state contributes) and `shuffled` (the contribution comes from correct content, not arbitrary bias). Beating only one is insufficient.

---

## 2. JM-S: Latent Memory Slot Pooling

### 2.1 Setup (fairness)

| Item | Value |
|---|---|
| Base | `350m-mlm` (same base for all arms) |
| Training data | `all_nli.jsonl`, 66,799 entailment pairs, 1 epoch, 521 steps |
| Optimizer | AdamW, lr 3e-4, warmup 20, wd 0.01 |
| Batch | 32 × 4 (effective 128) |
| max_length | 128 |
| Seeds | 1 / 2 / 3 |
| MoE | retained, `load_balance_weight=0.01` constant, not ablated |

### 2.2 Main result (SICK-R, independent holdout, 3 seeds)

| ID | Arm | SICK-R Spearman | Params |
|---|---|---|---|
| B0 | Mean Pooling | **0.6089 ± 0.0039** | 291.87M |
| B1 | CLS Pooling | 0.5903 ± 0.0037 | 291.87M |
| B2 | Attention Pooling | 0.6078 ± 0.0067 | 291.87M |
| B2mlp | Attention + MLP (param-matched) | **0.6091 ± 0.0045** | 292.92M |
| B3 | **Latent Memory (JM-S)** | **0.5923 ± 0.0035** | 292.92M |

**B3 − B2mlp = −0.0168**, pooled seed SD 0.0057 → ~3× SD, **opposite to the success criterion**.

SciFact retrieval agrees: B0 R@10 0.0737 / B2mlp 0.0678 / B3 0.0619.
**Two independent holdouts agree; not a single-dataset artifact.**

### 2.3 Root-cause diagnosis: slots do not differentiate

| Metric | B3 (8 slots) | Reading |
|---|---|---|
| Pairwise cosine of learned slots | −0.0301 | well separated at init |
| JS divergence of slot attention | **0.0437** | near 0 → all slots read the same tokens |
| Pairwise cosine of slot states | **0.8716** | outputs nearly identical |

Slots initialize well but **converge to the same point** after training. Eight slots are effectively one.

### 2.4 Repair attempt: Slot Attention (round 2; the root-cause hypothesis is falsified)

Following Locatello et al., NeurIPS 2020 ([arXiv:2006.15055](https://arxiv.org/abs/2006.15055)), six repair arms:

| ID | Change | SICK-R | Slot cosine | Slot attention JS |
|---|---|---|---|---|
| B3 | v1 baseline | 0.5923 | 0.872 | 0.044 |
| C1 | slot-dimension competitive softmax | 0.5736 | 0.825 | 0.089 |
| C2 | C1 + 3 iterations + GRU | 0.5694 | **0.977** | 0.106 |
| C3 | C2 + diversity loss | **0.5639** | **0.259** | 0.120 |
| C4 | C2 + 16 slots | 0.5669 | 0.985 | — |
| C5 | C2 + slot attention pooling | 0.5635 | 0.976 | — |
| C6 | C2 without GRU | **0.5954** | 0.909 | — |
| — | B2mlp (param-matched control) | **0.6091** | — | — |

**Key conclusion: the competition mechanism works, yet performance gets worse.**

- C1 does change slot behavior (JS 0.044→0.089, cosine 0.872→0.825) — **the diagnosed symptom is fixed**;
- C3's diversity loss drives slot cosine down to **0.259** (the "healthiest" structure) yet scores **second worst**;
- **Slot differentiation is uncorrelated, even negatively correlated, with task performance.**

This falsifies the "slots failed because they did not differentiate" hypothesis. **Even with ideal slot structure, JM-S still trails parameter-matched Attention Pooling.**

### 2.5 JM-S verdict

| Claim | Verdict |
|---|---|
| Memory slots add independent value over param-matched pooling | ❌ **Not supported** (−0.0168, both holdouts agree) |
| Learned pooling works | ⚠️ True, but **that gain is already captured by Attention Pooling**, not by slots |
| Slot differentiation is the bottleneck | ❌ **Falsified** (C3 best-differentiated, second worst score) |

**Limitation**: on short sentences (SICK-R sentences mostly ≤128 tokens), attention pooling already covers all tokens, so slots hold no information advantage. The original hypothesis in `memory-plan.md` §1.1 — slots for "lossy compression of long inputs" — is **neither confirmed nor refuted** on short-sentence tasks.

---

## 3. JM-R: Cross-Input Memory (Round 1, Probe Stage)

### 3.1 Four of six probes are invalid

We tested each probe with a lexical-overlap ceiling (`chunk_overlap_ceiling`):

| Probe | Lexical ceiling | Best model score | Verdict |
|---|---|---|---|
| **conflict** | **1.00** | 1.00 | ❌ Zero information: 100% without reading memory |
| **isolation** | **0.63** | 0.43 | ❌ Zero information: lexical beats all models |
| **order** | **0.53** | 0.60 | ❌ Nearly invalid |
| far_retrieval | 0.30 | 0.53 | ⚠️ Partially valid |
| distractor | 0.23 | 0.63 | ⚠️ Partially valid |
| **aggregation** | **0.00** | 0.87 | ✅ **The only fully clean probe** |

**The conflict probe went through four revisions**, each refuted by a control:

| Version | Design | Result | Failure |
|---|---|---|---|
| v1 | differing numbers | 0.00 or 1.00 | near-identical candidates; ordering driven by tokenizer artifact |
| v2 | labels Current/Outdated | B0 = 1.000 | stateless model scores perfectly → pure keyword match |
| v3 | only last digit differs | all 0.000 | candidate embeddings nearly collinear |
| v4 | natural sentences | chunk-ceiling 1.00 | candidates lexically overlap the block |

**The single most important lesson**: without the `chunk_overlap_ceiling` baseline, `B6r`'s **1.000** on conflict would have been reported as a major success.

### 3.2 Main result: no gain, mostly negative

`carried − reset` (>0 means memory helps):

| Probe | B5 | B5r | B6 | B6r |
|---|---|---|---|---|
| far_retrieval | 0.000 | 0.000 | −0.100 | 0.000 |
| aggregation | 0.000 | 0.000 | −0.133 | 0.000 |
| conflict | 0.000 | 0.000 | 0.000 | 0.000 |
| distractor | 0.000 | 0.000 | 0.000 | +0.067 |
| isolation | −0.067 | 0.000 | 0.000 | 0.000 |
| order | −0.067 | 0.000 | 0.000 | 0.000 |

**24 valid cells: 19 zero, 4 negative, 1 positive** (+0.067, n=30 → ~2 examples, and that probe's lexical ceiling is 0.23).

### 3.3 The only clean probe shows memory is harmful

aggregation probe (lexical ceiling 0.00, no shortcut possible):

```
B5: independent 0.867 → carried 0.000
B6: independent 0.867 → carried 0.100
```

**Turning on state collapses 0.867 to 0.000.** Catastrophic breakage, not insufficient gain.

### 3.4 Methodological defects found by the P0 audit

| # | Problem | Impact |
|---|---|---|
| 1 | On non-recurrent arms, `independent`/`reset`/`carried` **execute the same code** | 72 cells are code identities, no evidence |
| 2 | `independent` (batched) vs `carried` (serial) differ in **batching** | differences confound batching effects |
| 3 | B5 uses a different effective batch (128 vs 96) | any cross-arm comparison involving B5 needs re-running |

**The only clean comparison is `carried − reset` on recurrent arms.**

---

## 4. JM-R: Cross-Input Memory (Round 2, Building a Clean Testbed)

### 4.1 Synthetic task: the mechanism does work (the only positive result)

**P1 task validity** (lexical ceiling 0.055–0.080 < chance 0.156; oracle 1.000):

| Arm | carried | concat | reset | shuffled |
|---|---|---|---|---|
| mean (3 seeds) | **0.492** | 0.461 | 0.161 | 0.162 |
| SD | 0.012 | 0.006 | 0.003 | 0.018 |

Chance 0.167. **carried − reset = +0.331**, 3/3 same sign, bootstrap CI excludes 0.

**Two critical qualifications:**

1. **Fair `concat` (both chunks encoded at once) = 0.491, indistinguishable from carried 0.492** (paired differences +0.002/+0.010/−0.009). The earlier claim "state transfer beats direct concatenation" is void — it used an unfair baseline (two embeddings summed, missing the second forward pass).
2. **`shuffled` 0.162 ≈ `reset` 0.161**: a wrong state equals no state, so the gain comes from correct content, not generic bias.

**Accurate statement**: on a synthetic cross-chunk task, the correct preceding state supplies useful information for subsequent prediction, **but the gain is equivalent to concatenating the text**.

### 4.2 Real text (SciFact): below the preregistered bar

Three-stage task (abstract first half → second half → claim 3-way classification), document-component splits, 228 test examples, Macro-F1:

| Condition | Mean | SD |
|---|---|---|
| **carried** | **0.3151** | 0.0408 |
| concat | 0.2953 | 0.0259 |
| shuffled | 0.2866 | 0.0193 |
| reset | 0.2553 | 0.0227 |

Constant-predictor baseline 0.196.

**carried > reset: 3/3 same sign ✅**
**carried > shuffled: 2/3 same sign ❌** (seed 1: −0.032)
**Accuracy CI: 2 of 3 seeds contain 0**

### 4.3 Long documents (Qasper): state collapse

12 chunks, evidence distance ≥8 chunks (93% of test):

```
loss 2.4304 → 2.4735      (ln(12) = 2.479, i.e. stuck at chance)
val carried acc@1 0.0667 → 0.0333   (chance 0.083)
inter-block cosine 0.9904, block-rep SD 0.0036, similarity rises monotonically with position
```

**State norm 37.2 → 205.3** (5.5× within 12 chunks).

### 4.4 Nine runs (preregistered, 3 splits × 3 seeds)

`carried − static` is negative on all three splits, **9/9 negative**:

| Split | carried − static | carried − shuffled |
|---|---|---|
| 0 | −0.0224 | +0.0026 |
| 1 | −0.0289 | −0.0053 |
| 2 | −0.0116 | −0.0076 |

Cross-split: carried − static = **−0.0210**, 95% CI [−0.0414, +0.0170] (contains 0).

**But the nine runs cannot adjudicate the mechanism**: loss is stuck at ln(12), every condition sits at chance (0.143–0.228), and **the trained model scores below zero-training BM25 (0.281–0.321)**.

---

## 5. Methodological Findings (the Reusable Core Contribution)

These come from direct measurement and are **independent of whether the memory mechanism works**.

### 5.1 Full fine-tuning destroys the pretrained representation

Full fine-tuning of the 350M model on 514 examples:

| lr | loss | test F1 | **separation sep** |
|---|---|---|---|
| untrained | — | — | **+0.021** |
| 1e-5 | 10.17 | 0.2096 | +0.0044 |
| 3e-5 | 19.34 | 0.1511 | +0.0032 |
| 1e-4 | 23.56 | 0.1598 | −0.0000 |

**Untrained sep = +0.021; after training at any learning rate it drops to ≤ +0.0044. Gradients are healthy (norm 14.3), so this is not a gradient problem.**

By contrast: frozen backbone + a 524.8K-parameter head reaches **0.215 in 170 seconds**, beating the best full fine-tuning result (0.21) that took 25 minutes.

**Implication**: fine-tuning a large model on tiny data is over-parameterized fine-tuning that destroys existing representation quality. **All subsequent JM-R experiments moved to a frozen backbone** — the key step that made results clean.

### 5.2 `reset` and `concat` are the same code

The P0 audit instrumented the real load path: on non-recurrent arms, `independent` / `reset` / `carried` take **the same branch**.

**Consequence**: 72 comparison cells are code identities. Treating those zeros as proof that "controls are correctly implemented" is **invalid reasoning** — they reflect code structure only.

### 5.3 Filtering leakage: BM25 baseline underestimated 2.2×

The original data construction filtered samples by BM25 performance **before splitting** (dropping items where BM25's top-1 was an evidence block), turning the test set into "the set of samples where BM25 failed":

| | Previously reported | Unbiased (no filter) |
|---|---|---|
| Retained samples | 592 | **881** |
| test size | 124 | **171** |
| **BM25 F1** | **0.140** | **0.302** |

**The test set was selected against baseline performance.** The filter needs no labels but does need baseline behavior, making it a subtle selection bias.

### 5.4 Missing projection in evaluation: 0.656 on train measured as 0.16

`eval_plan1.step2_repr` returned `encode()`'s **pooled vector (pre-projection)**, whereas the true embedding is `pooled → projection → L2 normalize`.

Result: a model with 0.656 training-set argmax accuracy measured at 0.16 (**below chance**) under that evaluator; all four arms became indistinguishable.

Localization: first ruled out state computation (batched vs serial max diff 2e-06), then batch size (8–128 all 0.656), and finally found the missing `projection`.

**Lesson: when training performance far exceeds evaluation, suspect the evaluator before the model.**

### 5.5 Small-sample positives must be re-checked at scale

At n=300, `carried` (0.170) beat both `reset` (0.120) and `concat` (0.113) — apparently the **first positive result in the study**. At n=1500 the gap **collapsed from +0.050 to +0.002**.

> Stopping at the small-sample result would have produced an "JM-R shows its first gain" claim that is wrong.

### 5.6 A near-miss measurement error

A "spread > 0.02" criterion for gate behavior output `gate value MATTERS`, but at n=122 two or three lucky examples trigger that threshold. Switching to "monotonicity + within chance ±2SE" reversed the conclusion to **gate value is not the bottleneck**.

### 5.7 Bad criteria invite wrong directions

The first version of experiment A used the "spread > 0.02" criterion (§5.6); an even earlier version treated "5 runs agree in direction" as significance. **Directional agreement ≠ statistical significance**, and those 5 runs came from 3 splits (split0 contributed 3).

---

## 6. Mechanism Diagnostics (Four-Layer Pathology of the State Pathway)

This is **the most valuable part of the study**: not "no effect", but the specific stages where failure occurs.

### 6.1 The output path never reads state

Code fact (`jaume/modeling/memory.py`):

```python
output_t = pool_slots(new_slots)     # depends on new_slots only
new_slots = state_{t-1} + read(hidden_t)     # pure residual, 1 iteration
```

Substituting:

```
output_t = mean(state_{t-1}) + mean(read(hidden_t))
```

- The first term is a **task-independent additive bias**;
- The second depends only on the current chunk and **contains no preceding content**.

**`pooled_A` and `state_A_mean` are bit-identical (max |diff| = 0.000000)** — the "memory state" is invisible at the output.

Linear probes confirm no extra information in slot structure either (mean 0.385 / flat 0.334 / best single slot 0.415; inter-slot cosine 0.935). **The hypothesis "mean pooling discards information" is falsified.**

### 6.2 Additive state collapse

Cross-attention read-out is discriminative, but the update erases it:

| Stage | Inter-block cosine |
|---|---|
| cross-attention **read** `read_i` | **0.7651** ✅ discriminative |
| **state after update** `state_i` | **0.9901** ❌ differences erased |

Relative change decays monotonically with chunk index:

| Chunk | 1 | 3 | 5 | 7 | 9 | 11 |
|---|---|---|---|---|---|---|
| \|Δ\|/\|s_prev\| | 0.3176 | 0.2219 | 0.1581 | 0.1261 | 0.1024 | **0.0979** |

**Collapse happens in the state update, not the read-out.** The gradient path is fine (backbone grad norm 1298.36), so this is not an optimization issue.

### 6.3 The gate degenerates into a constant

| State | g mean | **g SD** |
|---|---|---|
| initialization (random inputs) | 0.5031 | **0.1316** |
| after 1 epoch training | 0.5193 | **0.0063** |

**SD shrinks 21×.** Meanwhile the weights changed by **141%** — the gate was trained extensively, into a solution that barely depends on input.

**This is genuinely "learned to ignore input", not "never trained".**

### 6.4 But forcing the gate value does not help

Frozen checkpoint, gate forced to fixed values:

| g | 0.10 | 0.30 | 0.50 | 0.70 | 0.90 | 0.99 |
|---|---|---|---|---|---|---|
| acc@1 | 0.0820 | 0.1148 | 0.0656 | 0.0984 | 0.0902 | 0.0738 |

No monotonic trend; **all six values fall within chance ±2SE**.

**Conclusion**: the problem is not what value the gate takes, but that **no matter how old and new state are mixed, the state holds no information useful for the task**.

### 6.5 Counterfactual sensitivity: 6× information loss

Fixing history and swapping the current chunk with chunk 1:

```
|Δ read|  = 0.09595
|Δ state| = 0.01550      ← 6.2× attenuation
g2        = 0.5192 → 0.5196   (essentially unchanged)
```

Chunk content changed; the state barely moved; the gate did not react.

### 6.6 Gating and collapse are two independent problems

| Target | Result |
|---|---|
| state norm no longer diverges (R1a/R1b normalization) | ✅ Solved: 205.3 → 64.0 |
| evidence vs non-evidence blocks distinguishable | ❌ Not solved: cosine still 0.99 |

**Normalization fixes scale, not direction** — as expected: "normalization constrains magnitude only, not direction".

---

## 7. Established Facts

**Directly measured, independent of interpretation:**

| # | Fact | Value |
|---|---|---|
| 1 | The 350M encoder has block-level signal with zero training | cos(gold) − cos(other) = +0.021; mean gold rank 3.42 (chance 6.0) |
| 2 | Full fine-tuning destroys that signal | all three lrs drop to ≤ +0.0044 |
| 3 | Frozen backbone + 525K head reaches 0.30 F1 in 170 s | direct measurement |
| 4 | State is worse than the chunk's own vector as a block representation | 5/5, −0.0785 |
| 5 | State is order-sensitive (wrong state is worse) | carried > shuffled, +0.032, 5/5 |
| 6 | **Encoder semantic representation is worse than BM25** | static 0.28–0.31 vs BM25 0.322 (zero training) |
| 7 | Slot differentiation is unrelated, even negatively related, to performance | C3 with cosine 0.259 scores second worst |
| 8 | Additive state collapse | norm 37.2→205.3; inter-block cosine 0.99 |
| 9 | Gate degenerates to a constant under training | SD 0.132→0.006, weights changed 141% |

Item 6 deserves emphasis: **on this task, lexical retrieval beats 350M semantic encoding.**

---

## 8. Verdict and Rationale for Stopping

### 8.1 Verdict for the current implementation

| Claim | Verdict | Evidence |
|---|---|---|
| JM-S: memory slots have independent value | ❌ **Not supported** | −0.0168, two holdouts, repair hypothesis falsified |
| JM-R: state is suitable as a block representation | ❌ **Not supported** | −0.0785, 5/5, confound-free under frozen backbone |
| JM-R: state carries relevant information | ✅ **Supported** | carried > shuffled +0.032, 5/5 |
| JM-R: that information beats direct concatenation | ❌ **Not supported** | carried 0.492 vs fair concat 0.491 |
| The cross-input memory route is refuted | ⚠️ **Cannot claim this** | only one state pathway tested (8 slots + mean pooling) |

**Precise statement**:

> Under the current implementation (8-slot state + slot mean pooling + gating) and the current training objective,
> cross-input state **produces no measurable information-transfer gain**;
> and the state is unsuitable as a block representation.

### 8.2 Three reasons to stop this round

1. Enough alternative explanations are excluded: task has no shortcut (lexical ceiling 0.055–0.080), answer is derivable (oracle 1.000), the model can learn the task (concat 0.476), training is sufficient (8 epochs, loss 0.44), evaluation is bug-free (re-measured after the P0 audit);
2. Pathology is localized to the structural level (output ignores state, additive collapse, gate degeneration) — three independent lines of evidence agree, not fixable by tuning;
3. Longer sequences **worsen** rather than mitigate: norm diverges monotonically (37→205) and collapse deepens (relative change 0.32→0.098).

### 8.3 What we explicitly will not do

Per `memory-final-report.md` §8:

- ❌ No further architecture expansion (no extra slots, no read-out layers)
- ❌ No further hyperparameter search
- ❌ No switching tasks to hunt for a positive result
- ❌ No adjusting criteria in light of results

**Rationale**: with no incremental gain shown for the state channel, added complexity would only make attribution harder.

---

## 9. The Only Retained Direction

`memory-final-report.md` §7 retains one direction: **incremental value for document-level prediction**.

The question that must be answered:

> Under identical training and evaluation conditions, does adding cross-chunk state make **document-level** prediction better than a model without state transfer?

Minimal configuration:

| Item | Requirement |
|---|---|
| Controls | `static` / `carried` (final state for document-level prediction) / `shuffled` |
| Data | train/val/test isolated by document/paper |
| Upfront | fixed primary metric, decision rule, compute budget |

**If `carried` is not significantly better than `static`, the direction ends; no further task-switching to find a positive result.**

Prerequisites (low cost, must come first):

1. Is there an existing document-level task with reliable labels?
2. Can existing data be reused?
3. **Does state actually participate in the final prediction** (avoiding another "output path never uses state")?

**If data, training pipeline, and evaluation must all be rebuilt, defer.**

### 9.1 A question this report does not cover

Every experiment here tests "**can state carry information**". None tests "**can a model judge what to remember**".

The latter is the meta-memory layer (salience / importance / type). It does not require a working state channel — an existing representation plus a prediction head suffices.
**It is the only direction this report does not refute.**

---

## 10. Limitations (Stated Plainly)

1. **Small test sets**: 168–179 per split; SciFact 228; probes 30;
2. **Only 3 independent splits**, with 3 seeds on split0;
3. **Only one task type tested** (block-level evidence localization);
4. **Only one base checkpoint** (`350m-mlm`); other pretraining stages untested;
5. **State-channel mechanism not fully probed** (partially covered in §6.1);
6. **BM25 was not trained as the model was** — not a strictly equal-condition comparison;
7. **Training length 128 tokens**, far below the configured 8192; long-sequence capability was never truly tested;
8. **Multilinguality untested**.

---

## 11. Reproduction

```bash
# JM-S (with slot diagnostics and Slot Attention repair arms)
python scripts/train_memory.py --config configs/jaume-350m.json --pooling latent_memory \
    --num-slots 8 --init-from 350m-mlm --train-data data/all_nli.jsonl \
    --output output/mem/B3-latent-s1 --seed 1 --epochs 1 \
    --batch-size 32 --grad-accum 4 --lr 3e-4 --max-length 128
python scripts/evaluate_holdout.py --model output/mem/B3-latent-s1/final --tasks sickr
python scripts/diagnose_slots.py   --model output/mem/B3-latent-s1/final

# JM-R (synthetic task, four conditions)
python scripts/make_crossblock_v3.py --n 4000 --cands 6 --out data/synth_v3
python scripts/g3_memory.py --data data/synth_v3 \
  --arms "B5=D:/Jaume/output/mem/B5-recurrent-s1/final" --epochs 1

# Diagnostic chain
python scripts/probe_state_shapes.py
python scripts/probe_sanity.py
python scripts/probe_state_facts.py --n 1500 --arm B5
python scripts/diagnose_composition_learning.py
python scripts/sweep_lr.py --lrs 1e-5 3e-5 1e-4 --epochs 1
python scripts/run_state_on_frozen.py --split 0 --seed 0
python scripts/diagnose_gate.py --mode additive --blocks 12 --papers 4

# Nine runs (preregistered)
python scripts/make_comp_clean_splits.py
for s in 0 1 2; do for k in 0 1 2; do python scripts/run_nine.py --split $s --seed $k; done; done
python scripts/aggregate_nine.py
```

---

## 12. Process Document Index

| Document | Content |
|---|---|
| `memory-plan.md` | original plan (JM-S/JM-R definitions, B0–B6, E1–E9) |
| `memory-preregistration.md` | nine-run preregistration (frozen) |
| `memory-results.md` / `memory-results-round2.md` | JM-S results, two rounds |
| `memory-results-crossinput.md` | JM-R probe results + probe-validity checks |
| `memory-p0-audit.md` | evaluator audit (3 defects) |
| `memory-p1-synthetic-task.md` | synthetic task validity |
| `memory-plan1.md` | aligning training objective with evaluation (the only positive result) |
| `memory-p15-g3.md` | decisive experiment (carried = reset = shuffled) |
| `memory-p2-round1.md` / `memory-p2-summary.md` | P2 iterations and failure analysis |
| `memory-scifact-spec.md` | real-text transfer on SciFact |
| `memory-qasper-collapse.md` | state collapse on long documents |
| `memory-task-necessity-audit.md` | task necessity audit (BM25 > model) |
| `memory-r1-results.md` | state-update repair (additive/norm/bounded) |
| `memory-gate-diagnosis.md` | gate behavior diagnostics (D4–D8) |
| `memory-state-probe.md` | state information probe (output does not read state) |
| `memory-experiment-a.md` | output-path repair experiment |
| `memory-breakthrough-diagnosis.md` | diagnostic chain for fine-tuning destroying representation |
| `memory-nine-run-report.md` / `memory-archive-index.md` | nine-run results and archive |
| `memory-final-report.md` | JM-R closing report (conclusions frozen) |
