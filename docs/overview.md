# Jaume 系列总览

轻量级、面向 Agent 的模型系列。每个成员是一个独立模型、解决一个方向的问题；成员之间不共享必选骨干，同仓托管、分文件归档。

> English version: [README.md](../README.md)
>
> 训练进度与权重发布在 ModelScope：[LightNeko/Jaume](https://modelscope.cn/models/LightNeko/Jaume)（本仓库）。代码镜像在 GitHub：[modemneko/Jaume](https://github.com/modemneko/Jaume)。

## 系列成员

| 成员 | 方向 | 状态 | 文档 |
|---|---|---|---|
| **Jaume-Embedding** | 文本语义表示 + 结构化决策（Choice / Score / Noul） | v0.1 已发布（350M 全流程）；新一代 MLM 基座训练中 | [SPEC.md](../SPEC.md) · [PLAN.md](../PLAN.md) |
| **Jaume-Memory** | 记忆槽池化（JM-S）与跨输入记忆（JM-R） | **已完成评估，两个机制均未通过预注册判据** | [技术报告](memory-report.md) · [实验计划](memory-plan.md) |
| **Neuro-Memory**（规划中） | 类脑海马体方向：结构化关系记忆、结构性编码（参考 Hippoformer / mm-TEM） | 方向定义完成，实验未启动 | [方向定义与首轮方案](neuro-memory-spec.md) |
| **Jaume-Temporal** | 时序骨架估计与控制（工作名称） | 任务定义阶段 | [temporal-spec.md](temporal-spec.md) |

**不属于上述任何系列的独立实验**：

| 实验 | 方向 | 状态 | 文档 |
|---|---|---|---|
| metamem | 外部记忆写入策略：无查询条件下判断"哪些信息值得记住"（冻结骨干 + 分类头，不改架构） | **已暂停，未产出有效结果**（编码阶段中止，待迁至服务器） | [暂停归档](memory-metamem-parked.md) · [预注册](memory-metamem-spec.md) |

> 归并原则：Jaume-Memory 研究**模型内部**的神经状态通道；Neuro-Memory 研究**类脑结构**如何承载关系记忆；
> metamem 研究**外部**的写入选择策略。三者问题不同，不应合并归档。


## Jaume-Embedding v0.1（当前唯一已发布成员）

架构：Bidirectional Transformer Encoder（GQA + RoPE + RMSNorm + MoE），Pooling → Projection → L2 Normalize 输出单位向量；同一 encoder 表征上挂 Choice / Score / Noul 决策头，一次编码多路决策。

### 已发布 checkpoint（350M 系）

| checkpoint | 说明 |
|---|---|
| `350m-mlm/` | Stage 1 MLM 基座（英/中），loss 11.9 → 4.06 |
| `350m-mlm-v2/` | 多语言继续 MLM（+德/西/土/阿/波/法 wiki） |
| `350m-stage2/` | 对比训练（自 `350m-mlm`）；STS-B 0.6111 @512D |
| `350m-stage2-ml/` | 多语言对比训练（自 `350m-mlm-v2`） |
| `350m-stage3/` | 硬负例重训；**STS-B 0.6223 @512D / 0.6108 @128D**，STS22 0.3554 |
| `350m-stage3-ml/` | 多语言硬负例重训；STS-B 0.6214 / 0.6107 @128D，STS22 0.3581 |
| `350m-decision-v2/` | 决策头（自 `350m-stage3`）：Choice 99.75%（ECE 0.0017），Noul AUROC 0.837，Score Pearson 0.29（弱） |

> 模型卡中的 STS 分数来自历史训练管线，其评测集与训练流程存在重叠风险；独立 holdout 复核结果以 `memory-plan.md` 的评测协议产出为准。

700M 尚未训练；当前所有实验先在 350M 规模上完成。

### 快速开始

```bash
# Stage 0 骨干验证（10 项检查）
python scripts/smoke_test.py --config configs/jaume-smoke.json

# 训练
python scripts/train_embedding.py \
    --config configs/jaume-350m.json \
    --train-data data/all_nli.jsonl \
    --output output/350m-nli \
    --epochs 1 --batch-size 16 --grad-accum 4 --grad-ckpt \
    --lr 1e-4 --use-matryoshka

# 评测
python scripts/evaluate.py --model output/350m-nli/final \
    --data data/stsbenchmark_sts.jsonl --baseline
```

### Python API

```python
from jaume import JaumeEmbedding

model = JaumeEmbedding.from_pretrained("output/350m-nli/final")
emb = model.encode(["HakusAgent 是一个 AI Agent。"])          # [1, 512]
scores = model.similarity(["如何训练模型？"], ["训练方法介绍"])  # cosine
emb128 = model.encode(text, dimensions=128)                   # Matryoshka 截断
```

## 仓库结构

```text
├── README.md            模型卡（Jaume-Embedding）
├── docs/overview.md     系列总览（本文件）
├── SPEC.md              Jaume-Embedding 架构规范
├── PLAN.md              Jaume-Embedding 训练计划与记录
├── docs/                成员级规划文档（memory / temporal）
├── configs/             模型配置
├── jaume/               Python 包（modeling / losses / data / inference）
├── jaume_tokenizer/     Loftey 151646 词表 BPE
├── scripts/             训练 / 评测 / 挖掘 / 同步脚本
├── tests/               单元测试
├── 350m-*/              已发布 checkpoint
├── data/                训练与评测数据
├── logs/                训练日志与 TensorBoard
└── code/                代码快照（随 checkpoint 同步打包）
```

## 硬件实测（RTX 3080 Ti 12GB）

- 350M + bf16 + gradient checkpointing + batch 8：约 6.3GB 显存，12 pairs/s（max_length 128）
- GradCache（chunk 96，关 checkpointing）：有效 batch 256，52 pairs/s，17.2GB（22GB 卡）
- 单卡对比学习需梯度累积或 GradCache 凑有效 batch；大规模训练需多卡
