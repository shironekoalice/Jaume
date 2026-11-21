# Jaume-Memory 实验结果报告（JM-S 阶段）

> 执行日期：2026-10-16
> 试验台：Jaume-Embedding 350M（291.9M 参数），RTX 4060 Ti 16GB
> 计划：[docs/memory-plan.md](memory-plan.md)；判定标准同 §2 / §5 / §9

## 1. 执行摘要

**JM-S（可学习记忆槽）在本轮实验中没有通过成功标准。** 在独立 holdout（SICK-R）上，Latent Memory Pooling 稳定低于参数量匹配的 Attention Pooling 对照臂，差距超过种子噪声。

| 编号 | 臂 | SICK-R Spearman (mean±std, 3 seeds) | 参数量 | 峰值显存 |
|---|---|---|---|---|
| B0 | Mean Pooling | **0.6089 ± 0.0039** | 291.87M | 8.03GB |
| B1 | CLS Pooling | 0.5903 ± 0.0037 | 291.87M | 8.03GB |
| B2 | Attention Pooling | 0.6078 ± 0.0067 | 291.87M | 8.03GB |
| B2mlp | Attention+MLP（参数匹配） | **0.6091 ± 0.0045** | 292.92M | 8.05GB |
| B3 | **Latent Memory (JM-S)** | **0.5923 ± 0.0035** | 292.92M | 8.04GB |

按 §9 预注册标准判定：

- **JM-S 相对参数匹配对照（B2mlp）：−0.0168**，种子合并标准差 0.0057 → 差距约为 3 倍标准差，**方向与成功标准相反**；
- SciFact 检索 holdout 同向（B0 R@10 0.0737 / B2mlp 0.0678 / B3 0.0619）；
- 两个独立 holdout 一致，不是单数据集偶然。

结论：**记忆槽没有产生独立价值**。按计划 §4 的论述，B3 连 B2 都未超过，因此连"学习式池化有效"都无法由记忆槽证明——现有 350M 已采用的 Attention Pooling 已经拿到了这份收益。

## 2. 实验配置（公平性，计划 §3）

| 项 | 取值 |
|---|---|
| 基座 init | `350m-mlm`（Stage 1 MLM，**所有臂同一基座**） |
| 训练数据 | `all_nli.jsonl`，66,799 蕴含对，1 epoch，521 步 |
| 优化器 | AdamW，lr 3e-4，warmup 20，weight decay 0.01 |
| batch / accum | 32 × 4（有效 batch 128） |
| max_length | 128 |
| 种子 | 1 / 2 / 3 |
| MoE | 保留，`load_balance_weight=0.01` 恒定，未消融 |
| 共用基座张量 | 309 个；记忆臂新增 7 个张量（+1.05M 参数） |

成本（均值）：约 90 pairs/s，8.0GB 峰值显存，单臂约 12 分钟。JM-S 相对 B0 额外参数 1.05M（+0.36%），推理延迟差异在测量噪声内。

## 3. 为什么失败：根因诊断

槽诊断脚本（`scripts/diagnose_slots.py`）显示这不是实现 bug，而是槽没有分化：

| 指标 | B3 (8 槽) | 解读 |
|---|---|---|
| 学习到的槽两两余弦 | −0.0301 | 初始化层面**分化良好** |
| 槽注意力分布 JS 散度 | **0.0437** | 接近 0 → **各槽读的是同样的 token** |
| 槽状态两两余弦 | **0.8716** | 输出几乎重合为同一向量 |
| 池化/槽状态范数比 | 0.942 | 均值池化未发生严重抵消 |

即：槽在参数空间被初始化得很好，但训练后**收敛到同一处**。8 个槽实际等价于 1 个槽，额外容量没有被使用，反而引入了额外的注意力噪声与优化难度。

两个修复方向均已实测，**未能挽回**：

| 消融 | 改动 | SICK-R | 槽状态余弦 | 判定 |
|---|---|---|---|---|
| E1 | 槽数 8 → 4 | 0.5938 | — | 容量不是瓶颈，减槽也无改善 |
| E4 | 加入槽间自注意力 + FFN | 0.5832 | 0.9568（更差） | 强制交互反而加剧同质化 |

E4 让槽状态余弦从 0.872 升到 0.957——槽间通信使它们进一步趋同，与"让不同槽提取不同语义"的假设相反。

## 3.1 第二轮：Slot Attention 修复（已执行，结论更强）

依 Slot Attention（Locatello et al., NeurIPS 2020, [arXiv:2006.15055](https://arxiv.org/abs/2006.15055)）实现 6 个修复臂，全部跑完（seed 1，与 B3-latent-s1 配对）：

| 编号 | 改动 | SICK-R | 槽状态余弦 | 槽注意力 JS |
|---|---|---|---|---|
| B3 | v1 基线（softmax 在 token 维） | 0.5923 | 0.872 | 0.044 |
| C1 | **槽维竞争 softmax**（论文核心机制） | 0.5736 | 0.825 | 0.089 |
| C2 | C1 + 3 轮迭代 + GRU | 0.5694 | **0.977** | 0.106 |
| C3 | C2 + 多样性损失 | **0.5639** | **0.259** | 0.120 |
| C4 | C2 + 16 槽 | 0.5669 | 0.985 | — |
| C5 | C2 + 槽注意力池化 | 0.5635 | 0.976 | — |
| C6 | C2 去掉 GRU | **0.5954** | 0.909 | — |
| — | B2mlp（参数匹配对照） | **0.6091** | — | — |

**关键结论：竞争机制确实生效了，但性能反而更差。**

- C1 真正改变了槽行为：JS 0.044→0.089，余弦 0.872→0.825，说明槽开始读取不同 token——**修复了诊断出的症状**；
- C3 的多样性损失把槽余弦压到 **0.259**（槽高度分化），是所有臂中最"健康"的槽结构，**却是第二差的分数**（0.5639）；
- 即：**槽分化程度与任务性能无关，甚至负相关**。C6（去掉 GRU、单轮精化）回到 0.5954，与 B3 基线持平，仍未超过 B2mlp。

这否定了"槽没分化导致失败"这一修复假设本身：即使把槽结构调到理想状态，JM-S 依然落后于参数匹配的 Attention Pooling。结合 §1 的结论，可以较有把握地判定：**在短句 STS/检索任务上，额外的槽容量没有可用信息可装**——plan §1.1 假设的收益场景是"有损压缩长输入"，与本轮任务不匹配。

（另：C2 的 3 轮迭代在未训练时就使槽余弦升到 0.998，迭代本身是收敛/同质化的加速器；C6 去掉 GRU 反而最好，说明 GRU 式精化在此任务上是负担。）

## 4. 与已发布 checkpoint 的对照

| 模型 | SICK-R |
|---|---|
| `350m-stage3`（已发布，历史管线） | 0.6293 |
| 本轮 B0-B3（MLM 基座 + 1 epoch NLI） | 0.5903 – 0.6091 |

差距来源：已发布 checkpoint 经过 Stage 2→3 完整流程（含硬负例挖掘与重训），本轮仅 1 epoch NLI 对比学习。**这不影响臂间结论**——所有臂共享同一训练预算，比较的是相对差异。

## 5. 评测隔离执行情况

- SICK-R（9,927 对，mteb/sickr-sts test）与 SciFact（339 query / 5,183 corpus，mteb/scifact test）**均为独立 holdout**，训练数据为 all_nli，无重叠；
- 所有臂的 checkpoint 选择未使用 holdout（固定 1 epoch，不做早停/择优），holdout 仅用于最终报告；
- bootstrap 1000 次报 95% CI；
- dev 集（STS-B / STS22）本轮未参与任何决策。

## 6. 结论与建议

1. **JM-S 在此试验台上判定不通过**，建议不进入初代定版；
2. **"槽未分化"的修复假设已被证伪**（§3.1）：Slot Attention 的竞争机制与多样性损失都成功改变了槽结构（余弦 0.872→0.259），但都使分数更低。继续在短句相似度任务上调池化结构，预期收益很低；
3. 若仍要验证记忆槽的价值，应换任务而非换结构：plan §1.1 的假设是"有损压缩长输入"，需要在**长文档压缩/跨块检索**上评测（plan §5.C 的六项探针与 JM-R 场景），当前 SICK-R/SciFact 短句任务无法证伪该假设，也无法证实；
4. **JM-R 已开展并判定不通过**：B5/B6 四个臂已训练完成并做了五项对照探针评测，
   详见 [memory-results-crossinput.md](memory-results-crossinput.md)。结论是跨输入记忆状态
   不仅无收益，在 21/24 个对照格子上为负；同时发现六个探针中有四个本身
   测量的是词面匹配而非记忆（其中 conflict 探针的词面上限为 1.00）。
   **该报告同时给出了"为什么负面结果不能证明路线失败"的机制分析。**

## 7. 产物

- 训练脚本 `scripts/train_memory.py`、评测 `scripts/evaluate_holdout.py`、跨输入探针 `scripts/eval_memory_crossinput.py`、跨输入汇总 `scripts/summarize_crossinput.py`、槽诊断 `scripts/diagnose_slots.py`、汇总 `scripts/report_memory.py`、上传 `scripts/upload_jaume.py` / `stage_upload.py` / `ms_upload_round3.py` / `ms_delete.py`；
- **18 个训练产物**在 `output/mem/`（B0/B2/B2mlp/B3 各 3 seed + E1/E4 + B5/B5r/B6/B6r），各含 `run_config.json`（配置+种子）、`train_summary.json`（步数/吞吐/显存）、`holdout_results.json`，记忆臂另含 `slot_diagnostics.json`、`memory_crossinput.json`；
- **已清理**（结论已否定且已成文，本地与远程均删）：`B1-cls`（基线被 B0 超越）与 `C1–C6`（第二轮槽位分化修复臂，假设已被证伪）共 10.0GB；
- 已上传 ModelScope `LightNeko/Jaume`（共 460 条目）：全部实验记录（config/结果/日志/TB/探针 JSON）+ 已发布 350m checkpoint 7 份 + 精选记忆臂权重 8 份（B0/B2/B2mlp/B3×3seed/E1/E4）；
- holdout 数据：`data/sickr-test.jsonl`、`data/scifact_*.jsonl`。

## 8. 实现注意（复现必读）

槽维竞争 softmax 绕过了 PyTorch 融合 SDPA，第一版实现把 K/V 投影放在迭代循环内，导致 14.8 pairs/s（比基线慢 6 倍）。`_kv()` 缓存后恢复到 47–49 pairs/s（基线 92 pairs/s，瓶颈是槽注意力的非融合路径）。若复现 C1–C6，请确认使用带 KV 缓存的版本。
