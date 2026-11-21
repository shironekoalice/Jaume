import io
p = "docs/planning-report.md"
src = io.open(p, encoding="utf-8").read()

old = """### 4.8 CE 版实验（进行中）

"""
new = """### 4.8 CE 版结果：损失函数对了，但表征是瓶颈（910B3 + 本机 XPU）

| 编码 | 标签 | train top-1 | beam success (w=1/3) | beam progress |
|---|---|---|---|---|
| split | imitation | 0.181 | 0.000 / 0.000 | 0.000 |
| split | optimal-next | 0.213 | 0.000 / 0.000 | 0.000 |
| split | in-optimal（BCE） | 0.872（二分类 acc，= 全负基线） | 0.000 | 0.416 |

对照：同一批任务上手写 greedy 的 **top-1 命中率是 1.000**，成功率 0.810。

**怎么读这组数**：

1. **CE 确实比 BCE 好**：CE 的 loss 从 log|A|≈3.18 降到 2.59（= 学到了 1.75 倍于随机的提升），
   而 BCE 的 loss 几乎不动（0.0456→0.0430），三个 head 全部塌缩到全负基线。
2. **但离 greedy 还差得远**：top-1 从 0.042（随机）到 0.21，学到了 5 倍提升，
   但 greedy 是 **1.000**。
3. **score-mode（logit vs log-prob 累加）没有区别** —— 排除了"累加尺度"这个解释。
4. **连"模仿 greedy"都做不到** —— 这是最关键的一条：
   imitation 标签是最容易学的（唯一正例、锐利），如果连它都学不到 greedy 的水平，
   那么问题**不在标签，也不在损失函数**。

**结论：瓶颈在表征。** 冻结的 350M 编码器 + mean pooling + 浅层 head，
拿不到"在给定状态下该选哪个动作"所需的信息。

### 4.9 决定性实验：能力在不在基座里？（`scripts/plan_finetune.py`）

冻结表征学不出来，只有两种可能，方向完全相反：

| | 含义 | 如果成立 |
|---|---|---|
| **(a)** 能力**在**基座里，只是冻结的浅层读出取不出来 | 内部能力建设路线**成立** | 微调后 head 应该能追上 greedy |
| **(b)** 能力**不在**基座里 | 这个方向在 350M 上走不通 | 微调后仍然学不会，得换基座/换任务 |

做法：**解冻整个编码器**，用同样的 segment softmax CE 端到端训练
（joint 编码，让编码器直接看到"状态 ‖ 动作"的配对）。
如果微调后 `imitation` 的 top-1 能从 0.21 提到接近 1.0，
那就证明了 (a) —— 这对"用小模型做规划"是一条**明确的正面结论 + 产品路径**。

"""
assert old in src
src = src.replace(old, new)
io.open(p, "w", encoding="utf-8", newline="\n").write(src)
print("ok")
