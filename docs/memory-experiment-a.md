# 实验 A：输出路径修复（进行中）

> 日期：2026-10-17
> 前置：[memory-state-probe.md](memory-state-probe.md)
> 状态：**训练流程的可学习性尚未确立，A 的判定暂不可得出**

## 1. 为什么必须先确认可学习性

按评审要求，A 必须**重新训练**（不能拿旧 checkpoint 做一次前向就当作公平对照）。
但重训的前提是：这条训练路径本身能学会任务。
否则 A 与 baseline 的差别会被"都没学会"掩盖，得到无意义的相等结果。

**本文件记录的就是这个前提的验证过程，以及发现的三个自身缺陷。**

## 2. 实验 A 的定义

只改一处：输出从 `mean(new_slots)` 改为 `mean(carry)`。

```
现状： n_t = slot_pass(h_t, s_{t-1})
       s_t = gate(s_{t-1}, n_t)        [B6]
       y_t = mean(n_t)                 ← 状态只作为初始化，不进入输出

A  ：  y_t = mean(s_t)                 ← 更新后的状态直接进入输出
```

代码位置：`jaume/modeling/memory.py` 的 `RecurrentMemoryPooling.forward`，
由配置项 `memory_output ∈ {new_slots, carry}` 控制，**默认值保持原行为**。

### 2.1 改动是惰性的（已验证）

`scripts/verify_output_path.py`：

| 检查 | 结果 |
|---|---|
| 默认模式下输出 == 原公式 | max\|diff\| = **0.0** |
| `carry` 模式下输出 == `pool_slots(carry)` | max\|diff\| = **0.0** |
| 新增参数量 | **0** |
| 旧 checkpoint 可加载 | ✅（state_dict 键不变） |

### 2.2 一个关键的预先发现

| 臂 | `use_gate` | A 是否改变输出 |
|---|---|---|
| **B5**（`recurrent_memory`） | False | ❌ **max\|diff\| = 0.0 —— A 完全无效** |
| **B6**（`recurrent_memory_gate`） | True | ✅ max\|diff\| = 0.783 |

因为 B5 里 `carry = new_slots`，所以 `mean(carry) ≡ mean(new_slots)`。

**结论：实验 A 只能在 B6 上做。在 B5 上做等于没做。**
这省掉了一整轮无效训练。

## 3. 已发现并修复的三个缺陷

### 3.1 训练数据格式不匹配（已修）

`PairDataset` 只认 `{query, positive}`，而 v3 数据是 `{chunks, query, candidates, gold}`。
直接喂进去会**静默产出 0 条样本**且不报错。

### 3.2 `train_memory.py` 参数块被破坏（已修）

我的一次编辑误删换行，导致 `SyntaxError`，三次训练瞬间失败。

### 3.3 训练时漏掉 `projection`（已修，影响最大）

```python
out, _, _ = base.encode(...)          # 返回的是“投影前”的 pooled 向量
loss = lossf(out, t_emb)              # ❌ 用未归一化的 512 维向量算对比损失
```

**症状**：loss 恒为 **3.4659 → 3.4658**，5 轮 600 秒**完全不动**。
而 `ln(32) = 3.466` —— 正好是 batch 内均匀随机的交叉熵。

修复：`out = normalize(base.projection(out))`，与推理路径一致。

修复后 loss 立刻开始下降：**3.4466 → 3.1010**（2 epoch）。

**这个 bug 曾让我一度以为"recurrent 路径的梯度是断的"——是我自己的实现错误。**
在没有排除实现错误之前，不能把这个现象当作模型的性质。

## 4. 当前状态（截至本文件）

### 4.1 可学习性验证：8 epoch（决定性）

修复 bug 后用 `carry` 模式训练 8 epoch（4000 对，bs 32）：

| epoch | loss |
|---|---|
| 0 | 3.4466 |
| 1 | 3.1010 |
| 2 | 2.6814 |
| 3 | 2.1515 |
| 4 | 1.1964 |
| 5 | 0.9752 |
| 6 | **0.4351** |
| 7 | 0.4857 |

**训练 loss 从 3.45 降到 0.44 —— 训练集几乎完美拟合。**

**测试准确率：0.179（随机 0.167）。**

### 4.2 诊断：训练目标与评测任务不一致

这是本轮最重要的发现，它解释了 `carried` 为何一直无效：

| 阶段 | 序列 | 答案依赖什么 |
|---|---|---|
| **训练** | 块 B（指令）→ 状态 → **块 A** → 输出 | 块 B 是**当前输入**，信息直接可见 |
| **评测（carried）** | 块 A → 状态 → 输出 | 块 B 已过去，**只能从状态取回** |

**训练时模型从不需要"从状态里取回块 B"**，因为块 B 就是当前步骤的输入。
于是状态承担的角色仅是"给下一步一个初始化偏置"，
模型学到的是「块 A + 当前步骤信息 → 答案」，而测试时并没有那个当前步骤信息。

**因此测试集停留在随机，且 loss 下降说明学到的是训练分布的 shortcut。**

### 4.3 这对实验 A 意味着什么

**A 无法在当前的训练流程下被判定。**
因为 baseline 与 A 都停在随机水平，四条件比较将得到"全都相等"的无信息结果。

必须先修正训练目标，让状态**必须**承载跨块信息，例如：

- 训练时只在**第二步**给答案标签，且第二步的输入**不含**块 B 的任何文本；
- 或加一个辅助损失：用第二步的状态预测块 B 的指令（强制信息写入状态）。

**只有训练目标真正奖励"从状态取回信息"时，A 与 baseline 的对比才有意义。**

## 5. 判定规则（事先固定，届时直接套用）

对每个 tag，用**跨种子均值**：

```
记忆有效 <=> carried > reset 且 carried > shuffled
```

同时报告：

- 每个种子的数值（至少 3 个配对种子）
- 均值、标准差、最小/最大
- `carried − reset` 与 `carried − shuffled` 的差值
- 与 `concat` 的差距（用于判断"信息是否可用但接口不通"）

## 6. 结果解读表（预先写定）

| 观察 | 结论 |
|---|---|
| A 中 `carried` 明显 > `reset` 且 > `shuffled` | 当前状态通路**能**支持跨块传递，值得继续 |
| A 与 baseline 相同，但 `concat` 明显更好 | 修复无效；下一步查门控梯度、状态保留、池化 |
| A 有效 | 支持"输出接口是瓶颈" |
| 全部失败 | **只能说这些实现与训练设置尚未建立有效传递**，不能证明跨输入神经记忆路线不可行 |

## 7. 未采纳 B 候选的理由

评审提出：若 A 无效，可试"保留槽结构的轻量可训练读出层"。

但槽结构探针已给出结论（见 state-probe §5.1）：

| 读出 | R² |
|---|---|
| mean | 0.385 |
| flat（8 槽） | 0.334 |
| 最佳单槽 | 0.415 |

槽间余弦 **0.935**，`std(mean)/std(slots) = 0.97`。

**信息并不在槽结构中，均值池化几乎无损。**
因此"读出层能救回被池化丢掉的信息"这一假设缺乏依据，B 未实施。

## 8. 复现

```bash
python scripts/verify_output_path.py                     # 确认改动惰性
python scripts/train_experiment_a.py --mode carry     --seed 0 --epochs N --out ...
python scripts/train_experiment_a.py --mode new_slots --seed 0 --epochs N --out ...
python scripts/eval_experiment_a_fast.py --tags ... --seeds 0 1 2
```
