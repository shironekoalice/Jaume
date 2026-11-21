# Jaume-Memory 第二轮：Slot Attention 修复（结果）

> 2026-10-16 ｜ 承接 [memory-results.md](memory-results.md) 的 JM-S 负面结果

## 结论

**"槽未分化"这一根因假设已被证伪。** 依 Slot Attention（Locatello et al., NeurIPS 2020, [arXiv:2006.15055](https://arxiv.org/abs/2006.15055)）实现的竞争机制与多样性损失都**成功改变了槽结构**，但**性能反而更低**。

| 编号 | 改动 | SICK-R | 槽状态余弦 | 槽注意力 JS |
|---|---|---|---|---|
| B2mlp | 参数匹配对照（Attention+MLP） | **0.6091** | — | — |
| B3 | v1 基线（softmax 在 token 维） | 0.5923 ± 0.0035 (3 seed) | 0.872 | 0.044 |
| C1 | **槽维竞争 softmax** | 0.5736 | 0.825 | 0.089 |
| C2 | C1 + 3 轮迭代 + GRU | 0.5694 | **0.977** | 0.106 |
| C3 | C2 + 多样性损失 | **0.5639** | **0.259** | 0.120 |
| C4 | C2 + 16 槽 | 0.5669 | 0.985 | — |
| C5 | C2 + 槽注意力池化 | 0.5635 | 0.976 | — |
| C6 | C2 去掉 GRU | **0.5954** | 0.909 | — |

（C1–C6 均为 seed 1，与 B3-latent-s1 的 0.5894 配对比较。）

## 证据链

1. **竞争机制确实生效**：C1 把槽注意力 JS 从 0.044 提到 0.089、槽余弦从 0.872 降到 0.825——槽开始读取不同 token，诊断出的症状被修复了；
2. **但分数更低**（0.5923→0.5736）；
3. **多样性损失把分化做到极致**：C3 槽余弦 0.259（槽高度正交），是所有臂中槽结构最"健康"的，却是倒数第二的分数（0.5639）；
4. **分化程度与性能无关，甚至负相关**：最不塌缩的 C3/C1 分数最低，最塌缩的 C2/C4（余弦 0.98）也没好过基线；
5. **迭代是塌缩加速器**：C2 在未训练时槽余弦就已达 0.998，3 轮精化让槽更快趋同；
6. **去掉 GRU 反而最好**（C6 0.5954），说明论文的 GRU 式精化在此任务上是负担。

## 解释

短句 STS/检索任务（SICK-R、SciFact）中，句子通常只有十几到几十个 token，**没有可供多槽分配的信息量**。Slot Attention 的竞争机制是为视觉场景（多物体、需要把像素分配给不同物体）设计的；在单句编码上，强行让槽竞争只会**切碎本应完整表示的语义**，而均值池化聚合多个近似副本反而起到类似集成的稳定作用。

因此 plan §1.1 的原始假设——记忆槽用于"有损压缩长输入"——**在短句任务上既无法证实也无法证伪**。要检验该假设必须换任务（长文档压缩、跨块检索、plan §5.C 的六项探针），而不是继续调池化结构。

## 实现注意（复现必读）

槽维竞争 softmax 绕过了 PyTorch 融合 SDPA。第一版把 K/V 投影放在迭代循环内，实测 **14.8 pairs/s**（比基线慢 6 倍，单臂 75 分钟）；在 `_kv()` 中缓存投影后恢复到 **47–49 pairs/s**（单臂约 23 分钟）。C1–C6 的结果均出自缓存后的版本。

## 复现

```bash
# C1 槽维竞争
python scripts/train_memory.py --config configs/jaume-350m.json --pooling latent_memory \
    --num-slots 8 --slot-competition --init-from 350m-mlm \
    --train-data data/all_nli.jsonl --output output/mem/C1-competition-s1 \
    --seed 1 --epochs 1 --batch-size 32 --grad-accum 4 --lr 3e-4 --max-length 128

# C3 = C1 + --slot-iterations 3 --slot-div-weight 0.1
# C6 = C1 + --slot-iterations 3 --no-slot-gru
python scripts/evaluate_holdout.py --model output/mem/C1-competition-s1/final --tasks sickr
python scripts/diagnose_slots.py   --model output/mem/C1-competition-s1/final
```
