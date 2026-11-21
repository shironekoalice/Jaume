# P0 评测脚本审计报告

> 日期：2026-10-17
> 对象：`scripts/eval_memory_crossinput.py`、`scripts/summarize_crossinput.py`
> 触发：用户指出 `carried ≠ reset` 只能说明"两种运行方式结果不同"，
> 不能证明状态携带了有用信息；并要求先审计统计单位与评分逻辑。
> **本审计不需要重新训练。**

## 1. 结论摘要

审计发现 **3 个实质问题**，其中 1 个使原报告的一部分结论失效：

| # | 问题 | 影响 |
|---|---|---|
| 1 | 非循环臂上 `independent`/`reset`/`carried` **执行同一段代码** | 报告里 25 个恒为 0 的格子**不含证据**；先前"健全性验证"的表述属过度解读（已更正） |
| 2 | `independent`（3 文本批量）与 `carried`（逐块串行）**批处理方式不同** | `carried − independent` 的差异**混杂批处理差异**，不能单独归因于记忆 |
| 3 | 报告未声明 B5 与其他三臂的有效批量不同（128 vs 96） | 涉及 B5 的横向比较**待复跑确认**（已在主报告 §7 补充） |

**唯一干净的对照**是循环臂上的 `carried − reset`：两者都逐块串行，唯一差别是状态是否传递。

## 2. 审计方法

用 `JaumeEmbedding.from_pretrained`（与真实评测完全相同的加载路径）
加载三种代表性臂，对 `_doc_vector` 打桩，记录每个策略实际调用的编码次数与文本数：

| 臂 | pooling | recurrent |
|---|---|---|
| B0-mean-s1 | mean | False |
| B3-latent-s1 | latent_memory | False |
| B6-recurrent-gate-s1 | recurrent_memory_gate | True |

## 3. 实测结果

```
=== B0-mean-s1  pooling=mean recurrent=False ===
  independent    batch_encode_calls=1 texts=3
  prev_only      batch_encode_calls=1 texts=1
  reset          batch_encode_calls=1 texts=3     ← 与 independent 同一分支
  carried        batch_encode_calls=1 texts=3     ← 与 independent 同一分支
  concat         batch_encode_calls=1 texts=1

=== B3-latent-s1  pooling=latent_memory recurrent=False ===
  independent    batch_encode_calls=1 texts=3
  reset          batch_encode_calls=1 texts=3     ← 同上
  carried        batch_encode_calls=1 texts=3     ← 同上

=== B6-recurrent-gate-s1  pooling=recurrent_memory_gate recurrent=True ===
  independent    batch_encode_calls=1 texts=3
  reset          batch_encode_calls=0 texts=0     ← 逐块串行，传 None
  carried        batch_encode_calls=0 texts=0     ← 逐块串行，传 state
```

### 3.1 问题 1 的代码位置

`_doc_vector` 中，非循环臂的三个分支是等价的：

```python
if strategy == "independent":
    embs = _encode(model, chunks)
    return normalize(embs.mean(dim=0, keepdim=True))

if strategy == "reset":
    embs = _encode(model, chunks) if not recurrent else None
    if embs is not None:
        return normalize(embs.mean(dim=0, keepdim=True))   # 与 independent 相同

# strategy == "carried"
if not recurrent:
    embs = _encode(model, chunks)
    return normalize(embs.mean(dim=0, keepdim=True))       # 与 independent 相同
```

**后果**：非循环臂的 4 个臂 × 6 个探针 × 3 个策略 = **72 个格子是代码恒等式**，
其中 24 个进入 `carried − reset` 表、24 个进入 `carried − independent` 表。
把这些 0 当作"对照实现正确的证明"是**错误的**——它们只反映代码结构。

### 3.2 问题 2 的实质

`independent` 与 `carried` 在**循环臂上**的差异不止"是否传状态"：

- `independent`：3 个块**一次批量**编码，再平均；
- `carried`：3 个块**逐块串行**编码，传状态。

批量 vs 串行会改变 padding、批归一化统计（若有）与逐块池化的组合方式。
因此 `carried − independent` 的差值**不能归因于记忆**。

**正确的隔离方式**是 `carried − reset`（循环臂专有），因为两者批处理完全一致。

## 4. 对既有结论的影响

| 原报告结论 | 审计后判定 |
|---|---|
| "B0 上 independent=reset=carried，证明对照实现正确" | ❌ **过度解读**，是代码恒等式（已更正） |
| "`carried − reset`：43 零 / 4 负 / 1 正" | ✅ 仍然成立，但其中 24 个 0 来自非循环臂，**循环臂上只有 19 个有效格子** |
| "`carried − independent`：21 负 / 25 零 / 2 正" | ⚠️ **差异混杂批处理效应**，只能作为弱证据 |
| "B5 在 aggregation 上从 0.867 崩到 0.000" | ⚠️ 该对比同时混淆了批处理差异 |
| "状态路径参与了计算" | ✅ 成立（循环臂上 reset/carried 确实走不同分支） |
| "状态携带了有用信息" | ❌ **从未被证明**，报告不应暗示 |

**循环臂上 `carried − reset` 的有效格子重算**：
4 个循环臂 × 6 个探针 = 24 格，扣除…… 实际 B5/B5r/B6/B6r 均为循环臂，
所以 `carried − reset` 表的 24 个格子**全部有效**（不是 48）。
重算：**19 零 / 4 负 / 1 正**。

## 5. 评分逻辑的其余部分（未发现问题）

- **统计单位**：每个探针 n=30 条，准确率 = 命中数 / n。分母是**条数**，不是候选数。
- **候选生成**：多数探针 8 个候选（1 正 + 7 干扰），随机基线 ≈ 0.125；
  实测随机基线 0.067，因为部分探针候选更多。**随机基线已逐探针报告**。
- **`carried` 与 `reset` 的输入完全一致**：同一 `chunks`、同一 `query`、同一 `candidates`，
  唯一差别是 `encode(..., state)` 的 `state` 参数。✅
- **评分一致**：两者都走 `run_probe` 的同一 `memory_only` 分支，同样的 `argmax`。✅
- **`lexical_leakage` 与 `chunk_overlap_ceiling`**：均为纯 Python 词面统计，与模型无关。✅

**唯一未解决的是 §3.2 的批处理差异**，需要改写 `independent` 使其也逐块串行（不传状态）
才能真正隔离。这是 P1 的前置工作。

## 6. 建议的修正（供 P1 使用）

1. `independent` 改为**逐块串行、不传状态**，与 `carried` 的批处理对齐；
2. 非循环臂上不再报告 `reset`/`carried`（它们是同一分支），
   或明确标注为"代码恒等，非证据"；
3. 所有涉及 B5 的横向比较统一批量后复跑；
4. 报告表格中区分"有效格子"与"恒等式格子"。

## 7. 复现

```bash
python scripts/_audit_paths.py        # 策略分支审计
```
