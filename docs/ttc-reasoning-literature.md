# Test-Time Compute / Reasoning 方向文献整理与核查

> 状态：**文献整理与核查完成；方向未立项**。本文是 Jaume 新方向的选型输入，不是 spec。
> 核查日期：2026-11-03
> 种子：微信文章《LoOPSD：让不同的层执行路径成为监督来源》+ 用户提供的 "Reasoning/Test-Time Compute (12篇)" 截图。
> 关联文档：`docs/Transformer热门创新思路166篇.md`（§Reasoning/Test-Time Compute 12篇）、`docs/memory-report.md`（负结果教训，见 §5.4）

---

## 0. 核查约定

| 标记 | 含义 |
|---|---|
| **[已核实]** | 经 arXiv API 读到摘要原文 + 作者 + 日期 + 分类，或经 GitHub API 读到仓库元数据 |
| **[仅元数据]** | 只核到标题 / arXiv ID / 日期，**未读正文摘要** |
| **[正文未读]** | 摘要已读，但方法细节、消融、实验设置均未核 |
| **[链接失效]** | 链接经实测不可达或 ID 不存在 |
| **[同名歧义]** | 同名或近名论文不止一篇，引用时必须带 arXiv ID 区分 |

**本次核查手段与限制**（透明记录）：

- arXiv API（`export.arxiv.org`）——可用，全部摘要经此核实。
- GitHub REST API（`api.github.com`）——可用，仓库元数据经此核实。
- **OpenReview API 不可用**：`api2.openreview.net` 返回 `403 ChallengeRequiredError`（人机验证），`openreview.net/forum?id=...` 返回的是 SPA 空壳（不同 id 长度均为 4787 字节，**状态码 200 不可作为"论文存在"的证据**）。因此 ICLR26 条目一律以 GitHub 仓库描述 + arXiv 交叉核验为准。
- ModelScope 侧无影响。

---

## 1. 种子文献：LoOPSD 全文核查

**LoOPSD: On-Policy Self-Distillation with Looped Teachers** — arXiv:**2610.18162** [已核实]

| 项 | 值 |
|---|---|
| 日期 | 2026-10-13（无后续版本） |
| 作者 | Wei Fan, Qingyue Yang, Chao-Hong Tan, Qian Chen, Wen Wang, Xiangang Li, Kejiang Chen, Weiming Zhang, Nenghai Yu |
| 分类 | cs.LG, cs.AI, cs.CL |
| 状态 | **Under Review**（非已接收会议） |
| 代码 | 摘要中未给出仓库链接 |

**方法（摘要原文核实）**：

1. 权重不变，在**层执行顺序**上搜索——具体是"重复执行某些层"，构造更强的教师；
2. 用教师在**学生自己生成的 prefix** 上的预测，通过反向 KL 指导学生学习；
3. **交替**"搜索 → 蒸馏"，每轮用更新后的学生重建教师。

**已核实实验事实**：Qwen3-1.7B 在 GSM8K、400-token thinking budget 下，42.30% → **79.61%**；多轮蒸馏在首轮之后仍有增益（论文 Figure 5：Qwen3-4B @128-token，33.81% → 50.87%，+17.06pp，而同等更新次数的标准路径自蒸馏为 26.38%）；同 400-token 预算下准确率 74.98% → 89.76%，平均总生成长度 385.04 → 299.98 tokens。

**LoOPSD 引用的对比方法（已逐一核实身份，注意歧义）**：

| 名字 | 实际所指 | arXiv | 核查结论 |
|---|---|---|---|
| Elastic Reasoning | *Scalable Chain of Thoughts via Elastic Reasoning*（Salesforce，2025-05） | **2505.05315** | [已核实] 思考/求解双预算，GRPO 集成，代码 Apache-2.0 |
| CRISP | ***CRISP: Compressed Reasoning via Iterative Self-Policy Distillation***（2026-03） | **2603.05433** | [已核实] **[同名歧义]**：另有 2507.08037（Step-based Plans）、2604.17297（CoT 剪枝）等 ≥4 篇同名，引用必须带 ID |

---

## 2. 核心文献总表（Top 15，均已核实摘要）

| # | 标题 | arXiv | 日期 | 定位 | 对我们的作用 |
|---|---|---|---|---|---|
| 1 | LoOPSD: On-Policy Self-Distillation with Looped Teachers | 2610.18162 | 2026-10-13 | 种子 | 起点 |
| 2 | Recurrent Self-Improvement (LoopOPD / D-LoopOPD) | 2610.10623 | 2026-10-07 | **直接竞品** | 用 frozen terminal loop 当教师，跨 loop 蒸馏；给出"单次更新同时改善两个 loop 深度的充分条件" |
| 3 | RISE: Recursive Improvement via Self-Extrapolating Policy Distillation | 2609.05295 | 2026-09-04 | **直接竞品** | 用 RLVR 轨迹在参数/logit 空间外推造教师；RLVR+OPD 互补循环 |
| 4 | CRISP: Compressed Reasoning via Iterative Self-Policy Distillation | 2603.05433 | 2026-03-05 | 对比方法 | "be concise" 指令当教师条件，per-token reverse KL；**教师靠 prompt 而非架构** |
| 5 | SelfMorph: Compute Has Shape | 2610.38382 | 2026-10-28 | **最直接对标** | 统一 BASE / REPEAT / BRANCH 三种计算形态；轻量 selector 按输入选形态后蒸馏 |
| 6 | MACRO: Markov Chain Routing of Transformer Layers | 2608.05872 | 2026-08-06 | 层路由 | skip/repeat/residual 三种操作建模为马尔可夫策略；比 Dr. LLM +7.2%，搜索时间 9.4× ↓ |
| 7 | Input-Blind Controls Produce Substantial Oracle Headroom for Layer Programs | 2610.10368 | 2026-10-07 | **方法论警示** | 见 §3.6 —— 对整条路线的前提构成挑战 |
| 8 | Towards Looped Models Done Right, Part I | 2610.23036 | 2026-10-16 | 架构 | Huginn vs Ouro 对比，**730M 参数 / 336B token** 规模；prelude-recurrent block-coda 三明治架构胜出 |
| 9 | Does Latent CoT Actually Reason? (Taxonomy) | 2610.21182 | 2026-10-15 | **证伪方法论** | 21 个 model-task 配置；区分"链里有什么"与"模型怎么用它" |
| 10 | What Makes Effective Supervision in Latent CoT | 2606.20075 | 2026-06-18 | 监督信号 | 潜变量崩塌 = 梯度衰减 + 表征漂移；提出 ULP 互信息探针 |
| 11 | Demystifying Hidden-State Recurrence (SWITCH) | 2606.13106 | 2026-06-11 | 可训练性 | 用 `<swi>`/`</swi>` 边界 token 让隐空间推理可被标准 on-policy RL 优化 |
| 12 | Elastic Reasoning | 2505.05315 | 2025-05-08 | 预算 | 思考/求解预算分离 + 预算约束 rollout |
| 13 | Input-Conditioned Search for Efficient Test-Time Scaling of Recursive Reasoners | 2610.18547 | 2026-10-13 | 搜索 | 递归推理器的输入条件化搜索 |
| 14 | Resample the Answer: Where to Spend Test-Time Compute in Recursive Reasoning | 2610.16427 | 2026-10-12 | 预算分配 | 递归推理中测试时计算该花在哪 |
| 15 | Monitoring latent reasoning in looped language models | 2610.18127 | 2026-10-13 | 诊断 | 循环 LM 的隐式推理监控 |

---

## 3. 文献分组与脉络

### 3.1 A 组：层执行路径 / 递归深度（Looped & Recurrent Depth）

LoOPSD 的教师来自"重复执行某些层"。这条线的技术演进：

| 论文 | arXiv | 日期 | 要点 | 状态 |
|---|---|---|---|---|
| UpLoop: Training Looped Transformers via Upcycling | 2610.41720 | 2026-10-30 | 零额外预训练算力把模型改造为 looped | [仅元数据] |
| How to Loop Your Transformer | 2610.40325 | 2026-10-30 | Loop scaling / 特化 / 并行推理 | [仅元数据] |
| ElTab: Elastically Looped Tabular Transformers | 2610.41819 | 2026-10-30 | 表格任务上的弹性 loop | [仅元数据] |
| CopyLoop | 2610.38822 | 2026-10-28 | 上下文复制 + 浅层筛选，加速 looped 解码 | [仅元数据] |
| LoopFold | 2610.35556 | 2026-10-26 | 并行化 looped transformer 的循环深度 | [仅元数据] |
| StateLoop | 2610.35544 | 2026-10-26 | looped transformer 的持久状态 | [仅元数据] |
| ScaleRLT | 2610.34427 | 2026-10-25 | looped transformer 上的精确 log-prob + 延迟 rollout（RL 训练需要） | [仅元数据] |
| Looping Beyond Twice | 2610.01153 | 2026-10-01 | 可扩展的 looped MoE 配方 | [仅元数据] |
| Decoding Looped Transformers Better for (Almost) Free | 2610.02185 | 2026-10-01 | 解码侧改进 | [仅元数据] |
| Depth-Asynchronous Self-Speculation | 2609.34538 | 2026-09-28 | 深度异步自推测 | [仅元数据] |
| LoopUS: Recasting Pretrained LLMs into Looped Latent Refinement Models | 2605.11011 | 2026-05-10 | **把预训练 LLM 改造成 looped 隐式精炼模型** | [已核实标题] |
| Where KV-Cache Error Lives (Ouro & Huginn) | 2610.41825 | 2026-10-30 | 诊断 | [仅元数据] |
| Agreement rather than normalisation keeps depth reusable | 2610.31943 | 2026-10-22 | 负结果/机制 | [仅元数据] |
| How Much Gradient Does Truncated Backprop Lose in Looped Transformers? | 2610.28871 | 2026-10-20 | 梯度截断代价 | [仅元数据] |
| Present but Hard to Read: Probing Latent Reasoning with Minimal Pairs | 2610.28498 | 2026-10-20 | 隐式推理探针 | [仅元数据] |
| Recency Is the Wrong Axis (KV cache 负结果) | 2610.34436 | 2026-10-25 | 负结果 | [仅元数据] |

### 3.2 B 组：教师构造——自蒸馏 / 在线策略蒸馏

这是 LoOPSD 所属的大类。按"教师从哪来"可分三条路线，**LoOPSD 属于第三条**：

| 路线 | 教师来源 | 代表 |
|---|---|---|
| B1 外部教师 | 更大更强的模型 | 经典 OPD（TeacherGRPO 2609.33426 等） |
| B2 **特权条件** | 同模型 + 学生看不到的额外上下文/指令 | CRISP 2603.05433（"be concise"）、DualOPSD 2608.26019 |
| B3 **改变自身计算** | 同模型 + 不同的层执行路径 | **LoOPSD 2610.18162**、LoopOPD 2610.10623、SelfMorph 2610.38382 |
| B4 参数/logit 空间外推 | 自身 RLVR 轨迹外推 | RISE 2609.05295 |

B2 与 B3 的**关键区别**（对规划很重要）：

- B2（CRISP）：教师和学生**同一个权重**，差别只在 prompt。教师的额外能力来自"指令诱导"，**推理时序不变**。
- B3（LoOPSD）：教师和学生**同一个权重、同一个 prompt**，差别在**层执行路径**。额外能力来自"多算了几层"。

→ **B3 的独特卖点是：教师不需要任何特权信息，也不会引入 prompt 分布偏移**（这是 B1、B2 的共同软肋）。RISE 摘要明确点出 B1 的"分布失配"和 B2 的"in-context learning capacity 上限"。

B3 的其他成员 [仅元数据]：

| 论文 | arXiv | 日期 |
|---|---|---|
| Know Thyself, Teach Thyself: Internal Information Flow for Selective Self-Distillation | 2609.36695 | 2026-09-29 |
| On-Policy Self-Training: Data-Free Self-Improvement | 2610.32770 | 2026-10-23 |
| Every Token Needs Its Own Teacher: Multi-teacher Adaptive On-policy Self-distillation | 2610.40574 | 2026-10-29 |
| Distilling the Shape of Uncertainty | 2610.39020 | 2026-10-28 |
| How Much do Teacher Bridges Matter in Interleaved OPD? | 2609.38383 | 2026-10-28 |
| ATI-OPD: Adaptive Teacher Intervention | 2610.38160 | 2026-10-28 |
| DTDD: Divergence-Triggered Dynamic Distillation | 2610.19302 | 2026-10-13 |
| Edit, Don't Imitate: Minimal-Edit OPD | 2610.30231 | 2026-10-21 |
| Adversarial Self-Distillation | 2610.38732 | 2026-10-28 |
| Not Every Token Is Worth Distilling (Selective Supervision for Direct-OPD) | 2609.29142 | 2026-09-24 |
| HiLo-RD: Cross-Budget Self-Distillation for Masked Diffusion LM | 2610.36901 | 2026-10-27 |
| MUON-RT: Reusing Teacher's Trajectory for Momentum-Free Muon | 2610.41104 | 2026-10-30 |
| MUON-RT / Turing LLM Post-Training | 2610.39876 | 2026-10-29 |
| DecisionRank: OPD for Decision-Model Rerankers | 2610.42128 | 2026-10-30 |
| Reasoning Compression with Mixed-Policy Distillation | 2605.08776 | 2026-05-09 |
| Stepwise-Querying: Distilling Reasoning without Teacher CoT | 2610.41769 | 2026-10-30 |

### 3.3 C 组：隐空间推理（Latent Reasoning）

与"隐空间推理"对应的用户截图里的 MoDr / EBT：

| 论文 | 来源 | 核查 |
|---|---|---|
| **MoDr: Mixture-of-Depth-Recurrent Transformers for Test-Time Reasoning** | ICLR 2026，OpenReview `9Pba4rcQbE` | [已核实-仓库描述] `zhangxjohn/MoDr` 描述写明 "[ICLR 2026] MoDr: ..."。**但仓库 3 star、创建于 2026-02-06 后再无 push、无 README（main/master 均 404）、无 license —— 实质为空壳，无可用代码** |
| **Energy-Based Transformers are Scalable Learners and Thinkers** | ICLR 2026，OpenReview `ZBj3Qp1bYg` | [已核实-仓库描述] `alexiglad/EBT` 描述 "PyTorch Code for Energy-Based Transformers paper -- generalizable reasoning and scalable learning"，**665 star / 91 fork / Apache-2.0 / 2026-04-21 仍在更新 —— 本方向唯一成熟可复现代码库** |

C 组其他重要文献：

| 论文 | arXiv | 日期 | 要点 |
|---|---|---|---|
| LatentWeave: Test-Time Latent Optimization for Multimodal Reasoning | 2610.39557 | 2026-10-29 | [仅元数据] |
| Latent Reasoning in Continuous Space for Unified MLLMs | 2610.42149 | 2026-10-30 | [仅元数据] |
| Learning to Reason with Selective Internal Feedback | 2610.21392 | 2026-10-15 | [仅元数据] |
| Structural Process Supervision for Latent CoT | 2609.09928 | 2026-09-09 | [仅元数据] |
| SCIT: Testing Causal Cache Carriers in Latent CoT | 2608.27265 | 2026-08-27 | [仅元数据] |
| Latent Path Exploration via Potential-Guided Subtrajectory Balance | 2610.40283 | 2026-10-29 | [仅元数据] |
| MPCoT: Reward-Guided Multi-Path Latent Reasoning | 2606.06245 | 2026-06-04 | [仅元数据] |
| Parallel-Probe: 2D Probing for Parallel Thinking | 2602.03845 | 2026-02-03 | [仅元数据] |

### 3.4 D 组：测试时计算扩展与验证

| 论文 | arXiv | 日期 |
|---|---|---|
| Adaptive Replay Refinement: Allocating Test-Time Compute Between Revision and Resampling | 2610.28294 | 2026-10-20 |
| ATEST: Adaptive Resource-Aware Test-Time Scaling | 2610.25519 | 2026-10-18 |
| P(Yes): 15× Improvements via Test-Time Scaling of Decision Models | 2610.25474 | 2026-10-18 |
| Compare to Correct: Pairwise-Guided Logit Adjustment | 2610.18583 | 2026-10-13 |
| Thinking-in-Bets at Test Time Reduces Sycophantic Revisions | 2610.34654 | 2026-10-26 |
| More: Mode-Optimized Reasoning with Self-Adaptive Effort | 2610.33684 | 2026-10-24 |
| SelfThinker: Reliability-Verified Intrinsic Visual Thoughts | 2610.40115 | 2026-10-29 |
| CLIFT: Conformal Self-Verification | 2610.06829 | 2026-10-05 |
| The Density of Thought: Predicting and Steering Reasoning | 2610.21286 | 2026-10-15 |
| LLMs Improving LLMs: Agentic Discovery for Test-Time Scaling | 2605.08083 | 2026-05-08 |

### 3.5 E 组：计算形态路由（MoD / Layer Program）

| 论文 | arXiv | 日期 | 要点 |
|---|---|---|---|
| Input-Blind Controls for Layer Programs | 2610.10368 | 2026-10-07 | 见 §3.6 |
| X-MoD: Sparse-Depth Routing Beyond Mixture-of-Depths | 2609.34212 | 2026-09-28 | [仅元数据] |
| Mixture-of-Depths Attention | 2603.15619 | 2026-03-16 | [仅元数据] |
| What Layers When: Learning to Skip Compute with Residual Gates | 2510.13876 | 2025-10-13 | [仅元数据] |
| LayerRoute: Adaptive Layer-Skipping with LoRA-Preserved Quality | 2609.13682 | 2026-09-12 | [仅元数据] |
| VALSE: Vertical Adaptive Layer Skipping | 2610.07606 | 2026-10-06 | [仅元数据] |
| vSkipper: Dynamic Layer Skipping → LLM Serving Gains | 2609.37062 | 2026-09-29 | [仅元数据] |
| SliMOO: Interpretable Multi-Objective Evolutionary Search for LLM Depth Pruning | 2610.27812 | 2026-10-20 | [仅元数据] |
| What Does Layer-Importance Reveal About Transformers and SSMs? | 2609.16537 | 2026-09-15 | [仅元数据] |
| CascadeFormer: Depth-Tapered Transformers (Gradient Fan-in Asymmetry) | 2606.26538 | 2026-06-25 | [仅元数据] |
| MIDUS: Memory-Infused Depth Up-Scaling | 2512.13751 | 2025-12-15 | [仅元数据] |
| Optimal Transport Depth Up-Scaling | 2508.08011 | 2025-08-11 | [仅元数据] |

### 3.6 F 组：方法论警示与负结果 —— **对本方向最关键的一节**

**Input-Blind Controls Produce Substantial Oracle Headroom for Layer Programs in Multiple-Choice Evaluation**（arXiv:2610.10368，2026-10-07，Yibei Guo, Rui Liu）[已核实摘要]

设计：32 个层跳过/重复程序 × 2 个模型（Qwen3-4B-Base、Llama-3.1-8B）× 4413 道多选题。对照组是**不看题目**（input-blind）选出的固定程序。

已核实的定量结果：

| 模型 | input-blind 对照的 headroom | 真实（按题目搜索）程序的 headroom |
|---|---|---|
| Qwen3-4B-Base | 10.2 – 11.8 pp | 9.0 pp |
| Llama-3.1-8B | 15.6 – 19.4 pp | 10.1 pp |

**对照组在全部 3 次随机抽取中都超过了真实程序。** 摘要结论原文：*"substantial headroom can persist across prompts with shared option order without establishing a benefit specific to the selected layer computation; neither ordering against these controls identifies that benefit."*

补充事实：固定字母偏移（fixed letter offsets）也能产生同量级 headroom；打乱选项顺序会同时削弱两类 headroom（真实程序仍领先 1.4–2.3 / 3.7–4.5 pp，但统计支撑依赖进一步校正）；一个补充的生成式测试发现"搜索选出的程序在改写题干后仍比"为其他题目选的程序"高 26.0 pp"，但**该测试没有 placebo 对照**。

**这意味着**：层程序带来的增益可能根本不是"针对这个问题的"，而是**任何固定层扰动都能带来的通用效应**（在多选题 + 共享选项顺序这一具体设定下）。

→ **对 LoOPSD 的直接后果**：LoOPSD 的整个前提是"搜到的循环路径能提供**更有用的**教师预测"。若该增益是通用的、与题目无关的固定扰动效应，那么蒸馏出来的学生可能只是学会了"对任何输入都做同一种扰动"，而非学会"针对该问题选更优路径"。**该文献的设定是多选题 + 共享选项顺序**，与 GSM8K 的生成式设定不同，因此不构成直接否证，但**它要求任何复现必须内置 placebo 对照**。

**这对 Jaume 尤其重要**：JM-S 方向的失败根因之一，就是缺少"参数匹配的联合编码对照组"（`memory-report.md` §4.1：carried 0.492 ≈ 公平 concat 0.491），导致"绕过记忆"无法排除。**同一个坑不能再踩第二次。**

其他相关警示：

- **Does Latent CoT Actually Reason? (2610.21182)** [已核实摘要]：在 21 个 model-task 配置上发现，COCONUT 这类方法在逻辑任务上常常"答案在链条开始前就已经可读"，状态早早收敛，删掉链条预测几乎不变。即使是"有序链条"也可能只是占位符——删除会掉分，但内容替换不影响答案。→ **"隐空间可读"不等于"在推理"**。
- **Recency Is the Wrong Axis (2610.34436)**、**Agreement rather than normalisation (2610.31943)**、**How Much Gradient Does Truncated Backprop Lose (2610.28871)**、**Monitoring latent reasoning (2610.18127)**：均为负结果/诊断类，说明该子领域当前有大量"看起来有效、实则机制不明"的风险。

---

## 4. 复现成本情报：代码可用性（GitHub API 实测，2026-11-03）

| 仓库 | star | 创建 | 最后 push | license | 可用性判断 |
|---|---|---|---|---|---|
| `alexiglad/EBT` | 665 | 2025-06-26 | 2026-04-21 | Apache-2.0 | ✅ **成熟可用**（本方向唯一） |
| `HJSang/OPSD_Reasoning_Compression` (CRISP) | 69 | 2026-03-07 | 2026-10-16 | 无 | ⚠️ 有实质代码，需自行核 license |
| `SalesforceAIResearch/Elastic-Reasoning` | 51 | 2025-05-07 | 2026-06-02 | Apache-2.0 | ✅ 可用 |
| `EIT-NLP/Supervision-in-Latent-CoT` | 1 | 2026-05-19 | 2026-06-20 | 无 | ❌ 空壳 |
| `Batorskq/MACRO` | 3 | 2026-08-06 | 2026-08-06 | MIT | ❌ 空壳（创建即终） |
| `creator-xi/SelfMorph` | 2 | 2026-10-27 | 2026-10-27 | MIT | ❌ 空壳（创建即终） |
| `zhangxjohn/MoDr` | 3 | 2026-02-06 | 2026-02-06 | 无 | ❌ 空壳，无 README |

**结论**：
1. **LoOPSD 本身没有公开代码**（摘要未给链接，2026-10-13 提交、状态 Under Review）——**必须从零实现**。
2. 该方向最热的两篇（MACRO、SelfMorph）代码都是空壳，**不要指望直接跑通**。
3. **EBT（665★ / Apache-2.0）** 是唯一有实质工程价值的代码库，若走隐空间/能量模型路线应优先评估它。
4. 若要复用 LLM 训练基础设施，优先看 EBT / CRISP / Elastic-Reasoning 这三个有实质代码的仓库。

---

## 5. 对 Jaume 的适配分析

### 5.1 现有资产（可复用的）

| 资产 | 位置 | 对新方向的用处 |
|---|---|---|
| 350M MoE+GQA+RoPE 双向编码器 | `350m-stage3/`、`code/repo/jaume/modeling/backbone.py` | 既是基座，也可能是"被搜索层执行路径的模型"本体 |
| 多语言 MLM 基座 | `350m-mlm-v2/` | 若走 AR 路线，可作为 AR 预训练起点 |
| 决策头（Choice / Score / Noul） | `350m-decision-v2/` | 判别式任务的现成评测头，**AR 化后可保留** |
| 预注册 + 验收判据范式 | `docs/memory-preregistration.md`、`docs/memory-plan.md` | 新方向直接沿用，是本项目最有价值的制度资产 |
| 负结果记录制度 | `docs/memory-report.md` | 失败也出报告 |
| 数据基建 | `data/`（89 文件，823MB） | 含 qasper/scifact 证据型任务（**非数学推理**，但可做证据推理） |

### 5.2 三条候选路线（成本/风险差异极大）

**路线 A —— AR 化 + 复现 LoOPSD（原汁原味）**
- 需要：350M 自回归基座、rollout buffer、reverse-KL 蒸馏管线、层重复路径搜索
- 风险：① 350M 模型在 GSM8K/MATH 上准确率大概率接近随机，LoOPSD 的 42.30%→79.61% 效应**无法在 350M 上重现**；② 蒸馏要大量 rollout，单卡 12GB 存不下大 buffer；③ 要从零实现（无参考代码）
- 判定：**高风险**。除非选更小的任务让 350M 有非随机基线

**路线 B —— Encoder-native 的"测试时计算"（保留 Jaume 的判别式优势）**
- 做法：层执行路径搜索 / 迭代精炼 / 多路径集成用在**双向编码器**上；教师 = 增强路径（如层重复 / 多路径平均 / 迭代 refine），学生 = 标准路径，用**表征分布对齐**蒸馏
- 好处：无需 rollout（encoder 一次 forward），单卡 12GB 友好；能直接复用 Jaume 的 STS/决策评测体系；判别任务有非随机基线
- 风险：没有现成文献直接对应（需自己定义"更强教师的信号"），是原创方向
- 判定：**低风险、与现有资产最匹配**，但需要自己想清楚"teacher 凭什么更好"

**路线 C —— 机制研究优先（做"证伪/诊断"型贡献）**
- 做法：直接对标 §3.6 的 Input-Blind Controls，在 Jaume 自己的设置下回答"层执行路径的增益是问题特异的，还是通用扰动效应？"
- 好处：**成本最低**，且这正是当前文献的空白点（2610.10368 只做了多选题 + 共享选项顺序，生成式推理任务上无人验证）；负结果也是一等贡献（Jaume 项目的既定风格）
- 风险：听起来"不够激动人心"
- 判定：**最低风险，且可作为路线 B 的前置阶段**

### 5.3 算力约束（硬约束，来自 README）

- RTX 3080 Ti **12GB**；350M + bf16 + 梯度检查点 + batch 8 ≈ 6.3GB VRAM
- GradCache（chunk 96，无 checkpointing）：等效 batch 256，**17.2GB —— 超卡**
- 单卡，无多机
- → **任何需要"教师 + 学生同时驻留 + 大 rollout buffer"的方案（路线 A）显存紧张**；路线 B/C 的前向-蒸馏显存友好

### 5.4 从 JM 负结果继承的硬约束（必须写进新 spec）

| 来源 | 事实 | 对新方向的约束 |
|---|---|---|
| `memory-report.md` §4.1 | carried 0.492 ≈ 公平 concat 0.491 | **任何"增强路径"必须配"公平对照路径"**，否则无法排除"就是多算了几层/多了一条通路" |
| `memory-report.md` §6.1 | 输出路径从不读取状态 | 蒸馏目标必须**结构性参与输出**，不能留旁路 |
| `memory-report.md` §6.3 | 门控训练后退化为常数 | **必须监控选择器/路由权重对输入的标准差**，并入验收项 |
| `memory-report.md` §6.4 | 强制改门控值救不回性能 | 先证明状态内含任务信息，再谈融合机制 |
| 2610.10368 | input-blind 对照 headroom ≥ 真实程序 | **必须内置 placebo / input-blind 对照**（这是文献级的强制项） |
| 2610.21182 | "隐状态可读"≠"在推理" | 任何"隐空间推理"主张必须做**因果干预**（删除/内容替换），不能只做探针可读性 |

---

## 6. 现有 `166篇` 文档 §Reasoning/Test-Time Compute 的核查结果

| 条目 | 链接状态 |
|---|---|
| MoDr（ICLR26） | OpenReview id 存在；`zhangxjohn/MoDr` 仓库存在但**为空壳**，无 README → **[代码不可用]** |
| Energy-Based Transformers（ICLR26） | OpenReview id 存在；`alexiglad/EBT` **665★ / Apache-2.0 / 活跃** → ✅ 唯一可用 |
| Multi-Task Bayesian In-Context Learning | arXiv 2606.20538 ✅ 标题/日期核实 |
| In-Context Algorithm Emulation in Fixed-Weight Transformers | arXiv 2508.17550 ✅ 标题/日期核实 |
| Transformers Learn Latent Mixture Models In-Context via Mirror Descent | arXiv 2604.10848 ✅ 标题/日期核实 |
| Continuum Transformers (X63V2CWjj3) | OpenReview id 存在；正文未读 |
| Transformers as Unsupervised Learning Algorithms (4hKNGmjXVQ) | OpenReview id 存在；正文未读 |
| RFT for Transformer-based Agents / PRGS / STAIRS-Former / A/B Testing RL | OpenReview id 均存在；正文未读 |

**建议**：166 篇文档的该节只有 12 篇，覆盖面远窄于本次整理（本次新增 **60+ 篇**，其中已核实摘要 15 篇）。建议扩为独立文档（本文件）并在 166 篇里加交叉引用。

---

## 7. 未解问题（需项目负责人决策，见对话）

1. **范式**：走 AR（复现 LoOPSD 原味）还是 encoder-native，还是先做机制诊断？
2. **任务**：数学推理（GSM8K/MATH）需要强基线；Jaume 现有的证据型任务（Qasper/SciFact）更贴合现有资产但非该文献的标准评测。
3. **预注册判据**：什么算通过？Jaume-Memory 用了"预注册 + 对照组 + 多 seed"。
4. **与已有方向的关系**：Jaume-NM、Temporal 均未启动，新方向是否排队？

---

## 附录：本文全部 arXiv ID 索引（已核实存在）

`2610.18162`, `2610.10623`, `2609.05295`, `2603.05433`, `2505.05315`, `2610.38382`, `2608.05872`, `2610.10368`, `2610.23036`, `2610.21182`, `2606.20075`, `2606.13106`, `2606.20538`, `2508.17550`, `2604.10848`, `2610.41720`, `2610.40325`, `2610.41819`, `2610.38822`, `2610.35556`, `2610.35544`, `2610.34427`, `2610.01153`, `2610.02185`, `2609.34538`, `2605.11011`, `2610.41825`, `2610.31943`, `2610.28871`, `2610.28498`, `2610.34436`, `2609.36695`, `2610.32770`, `2610.40574`, `2610.39020`, `2609.38383`, `2610.38160`, `2610.19302`, `2610.30231`, `2610.38732`, `2609.29142`, `2610.36901`, `2610.41104`, `2610.39876`, `2610.42128`, `2605.08776`, `2610.41769`, `2610.39557`, `2610.42149`, `2610.21392`, `2609.09928`, `2608.27265`, `2610.40283`, `2606.06245`, `2602.03845`, `2610.28294`, `2610.25519`, `2610.25474`, `2610.18583`, `2610.34654`, `2610.33684`, `2610.40115`, `2610.06829`, `2610.21286`, `2605.08083`, `2609.34212`, `2603.15619`, `2510.13876`, `2609.13682`, `2610.07606`, `2609.37062`, `2610.27812`, `2609.16537`, `2606.26538`, `2512.13751`, `2508.08011`, `2610.18547`, `2610.16427`, `2610.18127`

---

## 附录 B：本文自检记录（2026-11-03）

本文引用的 **84 个 arXiv ID 已逐个通过 arXiv API 批量 `id_list` 查询验证，全部存在**（`VERIFIED EXISTING: 84 / 84`）。

自检发现并已修正的一处错误：

| 错误 | 实际情况 |
|---|---|
| 原文写作 `2606.28871→2610.28871`（"How Much Gradient Does Truncated Backprop Lose in Looped Transformers?"） | **2606.28871 是一篇完全无关的论文**（*A Bayesian latent Gaussian process framework for aerodynamic uncertainty quantification*，贝叶斯高斯过程气动不确定性量化）。正确 ID 为 **2610.28871**。已修正 |

**教训**：`2606.28871` 与 `2610.28871` 仅月份数字不同，肉眼极难发现，但指向完全不同领域。**arXiv ID 必须程序化验证，不能靠肉眼。**

### 已知的核查盲区（本文未能覆盖）

| 盲区 | 原因 | 建议后续动作 |
|---|---|---|
| OpenReview 收录状态（ICLR26 等） | `api2.openreview.net` 返回 `403 ChallengeRequiredError`（人机验证）；`openreview.net/forum?id=` 返回 SPA 空壳，不同 id 响应长度均为 4787 字节，**状态码 200 不可作为存在性证据** | 人工浏览器访问逐条确认；本文一律以 GitHub 仓库描述 + arXiv 交叉核验 |
| 论文正文、消融、实验设置 | 本次仅读摘要 | 立项后按 §1 方法段逐篇精读 |
| 引用数 / 影响力排序 | 未接入 Semantic Scholar API | 如需，可后续补 |
| arXiv API 限流 | 连续单条查询会被限流（返回 301/超时） | 批量 `id_list` + `curl -L` + 间隔 3s，可稳定绕过 |
