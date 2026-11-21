# metamem（元记忆写入选择）：暂停归档

> 状态：**已暂停，未产出任何有效结果**
> 日期：2026-10-24
> 归属：**不属于 Jaume-Memory（JM-S/JM-R），也不属于拟建的 Neuro-Memory 系列**

---

## 0. 一句话结论

代码与预注册全部就绪，但**编码阶段被中止，实验从未跑出真实结果**。
`output/metamem/_mock_pipeline_check/` 里的数字来自随机向量，仅用于验证流水线，**不得引用**。

---

## 1. 这个实验是什么（以及不是什么）

**研究问题**：在看不到未来查询的前提下，冻结表征 + 小判断头能否判断"哪些文本块值得写入有限容量记忆"。

**它与 Jaume-Memory 的区别**（这是归档时最重要的界定）：

| 维度 | Jaume-Memory（JM-S/JM-R） | metamem |
|---|---|---|
| 问题 | 状态能否承载跨块信息 | 能否判断该记什么 |
| 改架构 | 是（槽位、门控、状态传递） | 否（冻结骨干 + 分类头） |
| 训练对象 | 350M 骨干 | 仅一个线性/MLP 头 |
| 记忆位置 | 模型内部 | 外部（判断的是写入策略） |

**它与 Neuro-Memory 的区别**：Neuro-Memory 指类脑海马体方向（Hippoformer / mm-TEM 式结构化关系记忆），
研究**结构性关系如何被神经表示承载**；metamem 完全不涉及类脑结构或关系记忆，
只做"文本块 → 是否重要"的分类。

**结论**：metamem 是**外部记忆写入策略**问题，独立于上述两个系列。若日后要归入某个系列，
需要新开一个"记忆管理/写入策略"方向，而不是并入 Neuro-Memory。

---

## 2. 已完成的工作（可复用）

| 项 | 位置 | 状态 |
|---|---|---|
| 预注册 spec（判定标准冻结） | `docs/memory-metamem-spec.md` | ✅ 完成 |
| 数据构造脚本 | `code/repo/scripts/make_metamem_data.py` | ✅ 完成并在真实数据上跑通 |
| 评测脚本 | `code/repo/scripts/eval_metamem.py` | ✅ 完成，mock 流水线验证通过 |
| 论文级数据集 | `data/metamem/papers.jsonl` | ✅ 已生成 |

**数据集实测规格**：581 篇论文（train 378 / val 87 / test 116），16,229 个块，
1,489 个问题，正块率 0.082（1,323 个证据块），论文级隔离（零重叠）。

**脚本已修复的环境问题**（这些修复是净收益，与实验成败无关）：
- `load_tokenizer()` 兼容老版 transformers 保存的 `extra_special_tokens`（list → dict）；
  `encode.py` 改用同一入口，所有 checkpoint 通用。
- M1/M2 判定改为**方向性**（只认神经 > 词面），修正了原实现"显著为负也算 pass"的逻辑错误。
  **这是判定文本的修正，须记录在案**：初版写"3/3 同号"，未规定方向；
  实测暴露后改为"3/3 全正 且 CI 下界 > 0"。修改发生在任何真实结果之前。

---

## 3. 未完成的工作

**唯一未完成项：块嵌入编码**（16,229 块 × 350m-mlm）。

- 在核显（Intel Meteor Lake）上跑了约 20 分钟未完成，GPU compute 引擎满载但吞吐太低。
- 中止后 `output/metamem/block_emb.npz` 从未生成，因此后续所有臂（H1/H2/H3、B1/B2、U0）均未运行。

**为什么移到服务器**：编码是一次性的特征提取，服务器上预计数分钟完成；
核显适合轻量验证，不适合这种批量推理。

---

## 4. 迁移到服务器：所需全部信息

**代码已在 `jaume/device.py` 中做好设备抽象**，服务器上 `resolve_device()` 自动走 cuda，**参数不用改**。

```bash
# 1) 数据（若仓库完整 clone 则已存在；否则重新生成）
python scripts/make_metamem_data.py

# 2) 主实验（首次运行会编码并缓存，之后复用 block_emb.npz）
python scripts/eval_metamem.py --seeds 1 2 3 --bootstrap 10000

# 3) 产物
#    output/metamem/block_emb.npz   嵌入缓存（可复用于其他写入策略实验）
#    output/metamem/results.json    全部臂指标 + M1/M2/G1 判定
```

**依赖**：numpy、scipy、scikit-learn、torch、transformers、tensorboard（若走训练脚本）。

**预计耗时（服务器）**：编码 3–10 分钟；三个神经臂 × 3 种子（纯 CPU 头训练，
特征已缓存）+ 两个 sklearn 基线 + bootstrap 10,000 次，约 10–20 分钟。

**判定标准不要改**：`docs/memory-metamem-spec.md` §5 已冻结。若迁移途中需要改动，
必须在跑 test 之前改，并记录改动与理由。

---

## 5. 恢复此项工作时的第一步

```bash
python scripts/eval_metamem.py --seeds 1 2 3 --bootstrap 10000
```

跑完先看三件事：
1. `B2_trivial_lr` 的 AUROC 是否 ≥ 神经臂（若成立，G1 触发，结论为"平凡特征已足够"）；
2. `B1_lexical_lr` 与神经臂的差值方向（报告事实 #6 预测词面更强）；
3. U0（BM25 + 查询）的 U1/U2 作为"有查询上限"的参照。

---

## 6. 相关文档

- 预注册与判定标准：[memory-metamem-spec.md](memory-metamem-spec.md)
- 元记忆方向的来源：`memory-report.md` §9.1（唯一未被否证的方向）
- 冻结边界：`memory-final-report.md` §8
