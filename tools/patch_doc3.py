import io
p = "docs/planning-report.md"
src = io.open(p, encoding="utf-8").read()

old_start = "### 4.6 三标签对照实验（进行中）"
old_end = "### 4.7 结果"
i = src.index(old_start)
j = src.index(old_end)
k = src.index("\n---\n", j)

new = """### 4.6 BCE 下的三标签对照 —— 三个 head 全部塌缩（910B3，2026-11-07）

一次嵌入、三套标签、同一套 head 结构、同一批测试任务（`scripts/plan_head_study.py`）：

| 标签 | 正例率 | train acc | **全负基线** | beam success | beam progress |
|---|---|---|---|---|---|
| optimal-next | 0.018 | 0.982 | **0.982** | **0.000** | 0.438 |
| imitation | 0.025 | 0.975 | **0.975** | **0.020** | 0.445 |
| in-optimal | 0.138 | 0.872 | 0.862 | （见 JSON） | — |

**train acc 等于全负基线 = head 根本没有学任何东西。**
训练 loss 也几乎不动（optimal-next: 0.0456 → 0.0430，起点约 −log(0.018) ≈ 0.078）。

这暴露了第三个、也是最根本的设计错误。

### 4.7 第三个设计错误：损失函数把"选择"变成了"打分"

原来的损失是**对每个 (state, action) 对独立做二元交叉熵**：

```
loss = BCE(head(concat(s, a)),  y[state, action])
```

在这个形式下，"永远预测负例"就是一个接近最优的解 —— 正例率 1.8% 时，全负的 loss 只有 0.078，
比模型要跨过的门槛低得多。更糟的是，**BCE 允许模型完全不看状态**：
只要学一个关于 *action 本身* 的先验（哪些动作在正例里出现得更频繁）就能降低 loss。
诊断里看到的正是这个退化 —— head 的 top-3 永远是 `place itemX at slot3`。

但我们要学的根本不是"这个动作好不好"，而是**"在这个状态下该选哪个动作"**。
那是一个**状态内部**的排序问题，正确的损失是**状态内 softmax 交叉熵**：

```
loss = CE( head(concat(s, a_1..a_|A|)),  target_action )
```

在这个损失下，**只用动作先验是 provably 无用的** —— 固定先验的 loss 恒为 log|A|，
唯一能降低它的办法就是使用状态信息。

于是设计对照实验（`scripts/plan_ce_study.py`）：

| 维度 | 取值 | 检验什么 |
|---|---|---|
| 损失 | BCE（对） vs **CE（状态内 softmax）** | 是否必须用状态 |
| 标签 | optimal-next / imitation | 哪种监督信号更可学 |
| 编码 | **split**（state 与 action 分开编码） vs **joint**（`state \\|\\| action` 一起编码） | mean-pooling 是否稀释了对应关系 |

CE 版的 `imitation` 是**能力上限参照**：如果它连 greedy 都学不到，
就说明 350M 基座没把状态暴露给浅层 head —— 那是关于**基座容量**的结论，与标签无关。

### 4.8 CE 版实验（进行中）

"""

src = src[:i] + new + src[k:]
io.open(p, "w", encoding="utf-8", newline="\n").write(src)
print("ok", len(src))
