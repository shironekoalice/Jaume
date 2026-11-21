# Qasper 长文档实验：发现状态坍缩

> 日期：2026-10-18
> 脚本：`make_qasper_blocks.py`、`check_qasper_data.py`、`qasper_batch.py`、`train_qasper.py`
> 数据：`data/qasper_blocks/`
> **结论：发现跨块状态的累加式坍缩，这是长文档场景下的结构性缺陷。**

## 1. 数据构造（全部检查通过）

Qasper 官方数据（S3 下载）：train 888 篇 / 2593 问，dev 368 篇 / 1143 问。

按 128 token 分块，按**论文连通分量**划分：

| 检查 | 结果 |
|---|---|
| Q1 论文级重叠 | **0 / 0 / 0** ✅ |
| Q2 答案不在问题中 | **0**（剔除 6 条） ✅ |
| Q3 证据块支撑答案 | 94.3% ✅ |
| Q4 词面基线 @1 | 0.0565（随机 0.0389）✅ |
| Q5 距离分层 | d≥8 有 196 条 ✅ |

距离分层（test）：d=0:2 / d=1-3:41 / d=4-7:44 / **d≥8:196**。

**这是 SciFact 完全没有的维度**：93% 的测试样本证据距离 ≥8 块。

## 2. 工程约束（实测，改变了配置）

| 配置 | 每步 | epoch | 峰值显存 |
|---|---|---|---|
| 逐块 batch-1 | — | — | GPU 利用率 **仅 4%** |
| maxlen=256, 20 块, bps=6 | **107s** | **3.4 小时** ❌ | OOM 风险 |
| maxlen=128, 12 块, bps=4 | **3.9s** | **7 分钟** ✅ | 11.5GB |

采用最后一种。批处理实现经 `verify_qasper_batch.py` 验证与逐块参考
**完全一致**（max\|diff\| = 0.000000）。

## 3. 核心发现：状态坍缩

2 epoch 训练后：

```
loss 2.4304 → 2.4735      （ln(12) = 2.479，即停在随机水平）
val carried acc@1 0.0667 → 0.0333   （随机 0.083）
```

**模型完全没学到东西。** 诊断单个样本：

```
blocks: 12  gold idx: 9
scores: [0.810, 0.868, 0.873, 0.890, 0.896,
         0.900, 0.903, 0.904, 0.903, 0.907, 0.909, 0.907]
argmax: 10          (正确答案是 9)
block-pair mean cosine: 0.9904
block rep std across blocks: 0.0036
```

### 3.1 根因分析

**块表示几乎完全相同（余弦 0.99，标准差 0.0036），相似度只随位置单调上升。**

状态更新公式（事实，读代码得到）：

```
state_i = state_{i-1} + read(block_i)      ← 在 slot_pass 内部累加
rep_i   = pool(state_i)
```

展开后是一个累积和：

```
state_i = state_0 + read(b_1) + read(b_2) + ... + read(b_i)
```

**这是一个候选解释**：累积项随 i 增大而主导方向，
导致不同块的 `state_i` 余弦趋近 1，输出只反映"读了几块"。

**但这是假设，未经干预验证。** 其他同样可能的解释：

| 候选解释 | 如何区分 |
|---|---|
| 累积项主导（加法稀释） | R1b：改为有界更新后应恢复 |
| 状态范数无界增长 | R1a：只归一化即可检验 |
| 交叉注意力读出在长输入上退化 | 需单独测 read(block) 与块数的关系 |
| 训练信号在多块下不可达 | 需对照 concat（无状态）是否也能学 |

**R1a / R1b 的目的就是分离这些解释。**

### 3.2 块数阈值：未验证

| 任务 | 块数 | 观测 |
|---|---|---|
| 合成 v3 | 2 | 正常（carried 远超 reset） |
| SciFact | 2 | 正常 |
| **Qasper** | **12** | 坍缩 |

**中间的 3–7 块从未测过。**
因此"坍缩从几块开始"**是未知的**，不能声称"只在 ≥8 块出现"。

## 4. 这属于哪一类问题

| 层次 | 判断 |
|---|---|
| 任务设计 | ✅ 干净（泄漏检查全过，词面基线接近随机） |
| 训练流程 | ✅ loss 起点 = ln(块数)，说明目标形式正确 |
| **状态通路** | ⚠️ **观测到多块下高度相似 + 训练失败，原因待定** |
| 记忆路线本身 | ⚠️ 不能据此否定 |

**这不是"记忆机制无效"，而是"当前的累加更新方式不适用于多块长序列"。**

## 5. 修复方向（未实施，需评审决定）

坍缩的原因是**纯加法**，没有归一化或遗忘。可选：

### 方案 R1：归一化状态更新
```python
state_i = normalize(state_{i-1} + read(block_i))
```
保持累加但防止范数无界增长。改动最小。

### 5.1 一个已被排除的猜测

初稿写"改用门控（R2）"作为修复方向。**经核查，门控本来就已启用**：

```
Q-smoke 训练配置：pooling = recurrent_memory_gate   (use_gate = True)
```

但坍缩依然发生。原因在代码路径：

```python
new_slots = slot_pass(hidden, init=memory_state)   # 内部是加法残差：slots = slots + read
carry     = gate(memory_state, new_slots)          # 门控只作用于 carry
return pool_slots(new_slots), carry                # ← 输出用的是 new_slots
```

**门控后的 `carry` 只保存给下一步，不进入输出。**
坍缩发生在 `slot_pass` 内部的累加上，与 gate 无关。

**这与之前"实验 A"诊断出的问题同源**：输出路径绕过 `carry`。
当时在**两块**任务上测不出差异（`carried` 与 `reset` 差别巨大，因为两块时累加尚未稀释），
但**在多块长序列上，同样的输出路径直接导致坍缩**。

**因此 R2 作废，R1（归一化累加）与"输出改用 carry"需重新评估。**

### 5.2 直接测试：改输出路径能否救回

在已训练的 Q-smoke 上切换 `memory_output`，测量块间坍缩：

| 输出路径 | 块间平均余弦 | 块表示标准差 |
|---|---|---|
| `new_slots`（当前） | 0.9847 | 0.0049 |
| `carry`（输出改用更新后状态） | 0.9806 | 0.0052 |

**两者都严重坍缩，差异可忽略。**

**结论：坍缩不来自输出路径的选择，而来自 `slot_pass` 内部的累加本身。**
改输出路径救不了（这也解释了为什么之前的"实验 A"在两块任务上无收益）。

**真正的问题在 `LatentMemoryPooling.slot_pass`：**

```python
init_slots = memory_state          # 上一块的状态
read = cross_attend(init_slots)    # 读当前块
slots  = init_slots + read         # ← 纯加法，无归一化、无遗忘
```

**R1（归一化累加）成为唯一有依据的方向。**

### 方案 R3：每块都从块内容重新初始化（滑动窗口）
不累积，只用最近 k 块的状态。退化风险是丢失远距离信息。

**建议先验证 R2**：检查 B6 的 gate 是否在长序列上真正发挥作用，
因为它在数学上最贴合"防止坍缩"的需求。

## 6. 必须在修复后重测的项目

修复状态坍缩后，按原定标准重新评估：

| 标准 | 状态 |
|---|---|
| S1 任务可学（concat 超随机） | 待测 |
| S2 记忆有效（carried > reset, shuffled） | 待测 |
| S3 长程收益（carried−reset 随 d 增大） | 待测 |
| S4 评价可靠 | ✅ 已满足（论文级划分、证据位置保留） |

## 7. 复现

```bash
python scripts/make_qasper_blocks.py --out data/qasper_blocks
python scripts/check_qasper_data.py --data data/qasper_blocks
python scripts/verify_qasper_batch.py
python scripts/train_qasper.py --seed 0 --epochs 2 --max-blocks 12 \
  --max-length 128 --papers-per-step 4 --out output/mem/Q-smoke
```
