# Jaume Training Plan (v0.1)

## Stage 0 — Backbone Validation ✅
`scripts/smoke_test.py`：10 项检查全部通过（Forward / Padding Mask / Backward / GQA / MoE Routing / Gradient Checkpointing / bf16 / Checkpoint Save-Load / Multi-Question Decision / Param Count）。

## Stage 1 — Encoder Pretraining ✅（2026-09-27，MLM）
- `scripts/train_mlm.py`：80/10/10 掩码，[MASK] 扩词表（151646→151647），**只对掩码位置投影 logits**（151k 词表瓶颈优化，3000→4400 tokens/s）
- 语料：wikitext-103（40 万行）+ 中文维基（40 万段），实际训练 30 万句
- 1 epoch（7503s），loss 11.9 → **4.06**；checkpoint 每 1000 步自动保存

## 全流程结果（MLM 初始化 → Stage 2 → Stage 3，GC 无 ckpt chunk 96 = 52 pairs/s）

| 模型 | STS-B 512D | STS-B 128D | STS22 512D |
|---|---|---|---|
| 旧 stage2（随机初始化） | 0.5451 | 0.5327 | 0.3892 |
| 旧 stage3-gc256（旧最佳） | 0.6100 | 0.5957 | 0.4391 |
| **新 stage2（MLM 初始化）** | 0.6111 | — | 0.3433 |
| **新 stage3（硬负例，新最佳）** | **0.6223** | **0.6108** | 0.3554 |

结论：MLM 预训练在英语 STS 上 +0.077（相对旧起点），一个 epoch 对比学习即超越旧全流程；**代价是多语言退化**（STS22 0.4391→0.3554），根因是 MLM 语料只有英/中。修复方向：多语言语料补齐 / 多语言对比数据。

## Stage 2 — Sentence Embedding ✅
- MNRL/InfoNCE + 硬负例列 + MatryoshkaLoss；`--init-from` 从 checkpoint 配置加载（词表权威化修复）
- GC 有效 batch 256；实测 chunk 96 关梯度检查点 17.2GB / 52 pairs/s（比 ckpt 32 pairs/s 快 62%）

## Stage 3 — Hard Negative ✅（第二轮）
- 新编码器挖掘：70% 覆盖（46,535/66,799），硬负例 mean 0.678
- Stage 3b：882s 重训，STS-B 0.6223

## Stage 4 — Matryoshka ✅
- 截断衰减极小：0.6223 → 128D 0.6108（只掉 0.012）

## Stage 5 — Decision ✅（v2，2026-09-27）
- `scripts/train_decision.py`：MLP-512 trunk 头（GELU+LN）+ 解冻顶层 4 层（backbone lr 1e-5 / heads 1e-3）+ Noul 数据 2 万→6 万 + Score 加 STS22 多源
- v1 → v2 对比：

| 头 | v1（冻结+线性） | v2（MLP+解冻4层） |
|---|---|---|
| Choice | 98.0% / ECE 0.070→0.018 | **99.75% / ECE 0.0017**（T=1.0，天生校准） |
| Noul | AUROC 0.608 | **AUROC 0.837**，温度标定后 ECE 0.015（T=3.16） |
| Score | Pearson 0.253 | Pearson 0.286（仍弱） |

Score 头待办：排序损失（regression-to-rank）或更大输入上下文；0-5 分 MOS 类任务对 4.6K→4M 参数头之外还依赖标注质量。

## Stage 6 — Calibration ✅（接入 v2）
- 温度标定：Choice 天生校准（T≈1.0）；Noul 过自信（T=3.16）标定后 ECE 0.015

## 已知问题 / 记录
1. 评测必须 hold-out（STS-B 自训自测会过拟合，0.3442 实证）
2. **GradCache**：两段式梯度等价（chunk=B 逐位一致，测试验证），MoE aux 按 token 占比加权、硬负例 pair→H 行映射；配合无 ckpt 实测有效 batch 256（6.8GB @ chunk 64）
3. **吞吐选优**：350M @ 22GB：GC chunk 96 + 无 grad-ckpt = 52 pairs/s（17.2GB），优于 ckpt 32 pairs/s——显存够时优先关 ckpt
4. 多语言退化：MLM 语料（英/中）导致 STS22 0.4391→0.3554；v0.5 需多语言语料/数据补齐
5. 硬负例覆盖率 70%（新编码器更强，带内竞争更平缓）
6. pkill/pgrep -f 会匹配自身命令行——用 `pattern[c]` 方括号技巧或按 PID
7. 断电恢复：MLM 每 1000 步存档有效；训练前检查 `/dev/nvidia*`（无 GPU 模式启动时缺失）

## 产物
- ModelScope 私有仓库 [LightNeko/Jaume-Embedding](https://modelscope.cn/models/LightNeko/Jaume-Embedding)：350m-mlm / 350m-stage2 / 350m-stage3 / 350m-decision-v2 + logs/（训练日志 + TB events）+ PLAN/SPEC
- 仓库单元测试 17/17；smoke 10/10

## CPU（无 GPU）环境测试记录
- cgroup 内存限额 2GB 且与 agent 进程共享（基线约 0.9GB），350M 权重（1.2GB）无法在 CPU 加载；21M 模型训练也超限（词表嵌入占 92% 参数）
- 新增 `configs/jaume-cpu-tiny.json`（hidden 64, 2 层, 9.9M 参数）：CPU 训练 46s 跑通（loss 1.77→1.18），推理 API / decide() / Matryoshka 截断全部可用，峰值 RSS 963MB
- 需要 `OMP_NUM_THREADS` 限制 + `MALLOC_ARENA_MAX=2` + `num_workers 0`，否则内存超限被 OOM-kill
- 单元测试 `tests/test_model.py` 通过（RoPE 相对位置性/范数保持、Matryoshka 截断一致性、pooling、MNRL、config 往返）
