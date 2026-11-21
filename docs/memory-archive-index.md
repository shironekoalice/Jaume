# JM-R 实验归档清单

> 归档日期：2026-10-20
> 远程：ModelScope `LightNeko/Jaume`
> **本归档不覆盖任何既有结果；第一轮（九次运行）与第二轮（dual）并列保存。**

---

## 1. 第二轮：dual 通道实验（本轮新增）

### 1.1 运行与划分的对应关系（关键）

**5 次运行来自 3 个划分，其中 split0 占 3 次。**

| 运行文件 | 划分 | 种子 | static | carried | dual | dual−static |
|---|---|---|---|---|---|---|
| `split0_seed0.json` | split0 | 0 | 0.2992 | 0.2281 | **0.3177** | **+0.0185** |
| `split0_seed1.json` | split0 | 1 | 0.2943 | 0.2242 | 0.2768 | **−0.0175** |
| `split0_seed2.json` | split0 | 2 | 0.3109 | 0.2086 | 0.2856 | **−0.0253** |
| `split1_seed0.json` | split1 | 0 | 0.2933 | 0.2402 | 0.2952 | +0.0019 |
| `split2_seed0.json` | split2 | 0 | 0.2798 | 0.2232 | 0.2679 | −0.0119 |

**汇总**：

| 比较 | 均值 | 范围 | 符号一致性 |
|---|---|---|---|
| dual − static | **−0.0069** | −0.0253 ~ +0.0185 | 正 2/5 |
| carried − static | **−0.0707** | −0.1023 ~ −0.0531 | **负 5/5** |
| dual − carried | **+0.0638** | +0.0446 ~ +0.0897 | **正 5/5** |

**统计单位说明**：独立划分只有 **3 个**；split0 的 3 次是同一划分的不同种子，
**不可当作独立样本**。

### 1.2 脚本

| 文件 | 作用 |
|---|---|
| `run_concat_state.py` | dual 实验（static / carried / dual 三条件） |
| `aggregate_concat.py` | 跨划分汇总 |
| `run_state_on_frozen.py` | 第一版冻结实验（state 替代块向量） |
| `run_frozen_head.py` | 冻结骨干 + 轻量头可行性验证 |
| `diagnose_composition_learning.py` | 诊断训练失败原因 |
| `sweep_lr.py` | 学习率扫描（发现全量微调破坏信号） |

### 1.3 结果文件

```
output/comp_concat/split{0,1,2}_seed{0..2}.json
output/comp_concat/aggregate.json
output/comp_state_frozen/split{0,1,2}_seed{0..2}.json
output/comp_head/split0_seed0.json
output/comp_multi/split0_seed0.json
output/lr_sweep.json
```

### 1.4 数据

```
data/qasper_comp_clean/split{0,1,2}/{train,val,test}.jsonl
data/qasper_comp_clean/meta.json
```

- 881 条跨块组合样本
- 三个固定的论文级划分，论文重叠 0/0/0
- **已移除 BM25 筛选**（原筛选造成测试集选择偏差）

### 1.5 报告

| 文件 | 内容 |
|---|---|
| `docs/memory-final-report.md` | **本轮最终收尾报告** |
| `docs/memory-breakthrough-diagnosis.md` | 诊断链与冻结实验结果 |

---

## 2. 第一轮：九次运行（保留，未覆盖）

| 项 | 内容 |
|---|---|
| 协议 | `docs/memory-preregistration.md`（事先冻结） |
| 报告 | `docs/memory-nine-run-report.md` |
| 脚本 | `run_nine.py`、`aggregate_nine.py`、`check_a_leakage.py`、`check_b_fairness.py`、`make_comp_clean_splits.py` |
| 结果 | `output/comp_nine/split{0,1,2}_seed{0,1,2}.json` + `aggregate.json` |
| 结论 | 未达预注册标准；且训练未成功，**不能归因于机制** |

**第一轮与第二轮的关系**：第一轮用全量微调（后发现会破坏表示），
第二轮改用冻结骨干重测。两轮结论**并列保存**，不互相覆盖。

---

## 3. 更早的实验（均已归档）

| 阶段 | 报告 | 主要产物 |
|---|---|---|
| 计划 | `memory-plan.md` | — |
| JM-S 结果 | `memory-results.md`、`memory-results-round2.md` | `output/mem/{B0,B2,B2mlp,B3,E1,E4}` |
| 跨输入探针 | `memory-results-crossinput.md` | `memory_crossinput.json` |
| P0 审计 | `memory-p0-audit.md` | `_audit_paths.py` |
| 合成任务 | `memory-p1-synthetic-task.md`、`memory-plan1.md` | `data/synth_v3/`、`output/mem/P1b-*` |
| SciFact | `memory-scifact-spec.md` | `data/scifact_blocks/`、`output/mem/SF-*` |
| Qasper | `memory-qasper-spec.md`、`memory-qasper-collapse.md` | `data/qasper_blocks/` |
| 任务必要性审计 | `memory-task-necessity-audit.md` | `audit_*.py` |
| Gate 诊断 | `memory-gate-diagnosis.md` | `diagnose_gate.py` |
| R1 实验 | `memory-r1-results.md` | `output/mem/R1-*` |

---

## 4. 关键 checkpoints

| 路径 | 说明 |
|---|---|
| `output/comp_state_frozen/` | 冻结实验的中间产物（特征已缓存，无权重） |
| `output/comp_concat/` | dual 实验（无权重，仅结果） |
| `output/mem/B0-mean-s1` … `B3-latent-s3` | JM-S 核心权重（3 seeds） |
| `output/mem/B5/B5r/B6/B6r` | JM-R 四臂权重 |
| `output/mem/P1b-s{0,1,2}` | 合成任务方案 1 权重（唯一有正结果的实验） |

**注**：dual 与冻结实验未保存模型权重（骨干冻结，只训练轻量头，
头参数在结果 JSON 中未序列化）。如需复现，用脚本重跑即可（每次约 10 分钟）。

---

## 5. 复现命令

```bash
# 数据
python scripts/make_comp_clean_splits.py

# 诊断链
python scripts/diagnose_composition_learning.py
python scripts/sweep_lr.py --lrs 1e-5 3e-5 1e-4 --epochs 1

# 冻结实验（state 替代）
python scripts/run_state_on_frozen.py --split 0 --seed 0

# dual 实验
python scripts/run_concat_state.py --split 0 --seed 0
python scripts/aggregate_concat.py
```

---

## 6. 归档检查

| 项 | 状态 |
|---|---|
| 第一轮结果保留未覆盖 | ✅ |
| 5 次运行与划分的对应关系已记录 | ✅ §1.1 |
| 统计单位限制已说明 | ✅ 独立划分仅 3 个 |
| 结论限定表述 | ✅ final-report §0–1 |
| 解释假说与已验证事实分开记录 | ✅ final-report §2–3 |
| 剩余限制如实记录 | ✅ final-report §5 |
| 明令不做的事 | ✅ final-report §8 |
