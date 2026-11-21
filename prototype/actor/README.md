# Jaume-Actor（原型）

小模型「聪明决策」原型：浏览器控制 + 游戏。全部实验可在本机复现（Intel XPU，
无下载、无外部 API）。

完整研究记录见 [`docs/jaume-actor-prototype.md`](../../docs/jaume-actor-prototype.md)。

## 一句话

**决策必须是「一次前向 + 打分 + argmax」，不能自回归解码；而且小模型应该做感知，
规划应该交给确定性求解器。**

## 结果

所有数字带 seed 数与**最差 seed**（方法论教训见报告 §3.6）。

| 架构 | 任务 | 成功率 | **最差 seed** | 延迟 | 参数 |
|---|---|---|---|---|---|
| **感知检测器 + BFS 规划** | 迷宫闭环 | **1.000** | **1.000** | 13.6 ms | 17.0M |
| 同上，检测器降频 1/16 | 迷宫闭环 | 1.000 | — | **0.85 ms (1175 Hz)** | 17.0M |
| 纯像素策略网络（CE 值函数） | 迷宫闭环 | 0.890 | — | 2.09 ms | 7.2M |
| 候选打分 actor | 合成 UI（decoy twin） | 1.0000 | **1.0000** | 40 ms | 0.26M |
| 候选打分 actor | **真浏览器 grounding** | **1.000** | — | — | 0.26M |
| 候选打分 actor | **真浏览器闭环控制** | **1.000** | — | — | 0.26M |

三条可迁移的结论：

1. **一次前向出决策**（仓库实测 350M 解码 154 ms/token，实时环用不了）
2. **规划交给求解器**（学出来的策略 0.890 vs 感知+BFS 1.000）
3. **能绕过视觉的输入通路会诱发捷径解**（坐标通道 / 过大初始 logit 尺度都会塌缩，
   最差 seed 从 1.0000 掉到 0.0213）

两个负结果：

- **空间 softmax 定位**：0.0607 vs 全局分类 1.0000 —— 目标是离散格子时，
  亚格精度没有价值，head 的形式要跟任务形式走。
- **帧堆叠 / 多步 lookahead**（游戏侧）：都比纯单帧策略更差（见报告 §3.4）。

## 基座选型（真实 ScreenSpot）

| 基座 | 参数 | AUROC (held-out) | top-1 | 地板 | 每 crop |
|---|---|---|---|---|---|
| **SigLIP2-base** | **93M** | 0.8991 | **0.8500** | 0.1667 | **44 ms** |
| Qwen3-VL-2B | 2B | 0.9472 | 0.8056 | 0.1667 | 481 ms |
| Qwen2-VL-2B | 2B | 0.9046 | 0.6667 | 0.1667 | 580 ms |

**指令条件化门控（真实干扰项，seeclick，held-out）**：

| head | 参数 | top-1 | 地板 |
|---|---|---|---|
| 线性探针（指令仅拼接） | 1.5K | 0.1846 | 0.1667 |
| **门控头（指令调制视觉）** | 2.4M | **0.6483** | 0.1667 |
| **门控头但去掉指令（对照）** | 2.4M | 0.1827 | 0.1667 |

> 随机干扰项会让「无指令对照组」也拿 1.0000 —— 指令根本没被用上。
> **任何 grounding 基准都必须有「去掉指令的对照组」。**

## 文件

| 文件 | 作用 |
|---|---|
| `envs.py` | 合成 UI 任务（两版）与迷宫环境 |
| `envs_hard.py` | 难版 UI（带 decoy twin）+ heuristic/oracle 对照 |
| `model.py` | `UIActor`（候选打分）、`MazeActor`、`SpatialCNN` |
| `model_jaume.py` | SigLIP2 + Jaume-350M 版本（NPU 用） |
| `train_ui.py` / `train_ui_hard.py` | UI 训练 + 消融 |
| `train_ui_ckpt.py` | 只训 full 变体并存权重（供真浏览器评测） |
| `train_maze_ce.py` | 迷宫 CE 值函数策略 |
| `train_maze_stack.py` | 帧堆叠 / 多步 lookahead 消融 |
| `train_maze_split.py` | 局部方向 vs 全局路径的分离 |
| `train_maze_value.py` | 值函数（MSE）版本 |
| `probe_localise.py` | 感知探针（agent/goal 定位） |
| `analyse_errors.py` | 错误分类（E1 撞墙 / E2 选错 / E3 振荡） |
| `screenspot_data.py` | **ScreenSpot 载入 + 候选集构造** |
| `fetch_models.py` | 从 ModelScope 拉基座权重（本机 HuggingFace 不可达） |
| `backbone_registry.py` | 基座注册表（SigLIP2 / VLM 的零样本与探针接口） |
| `backbone_probe.py` | **线性探针（held-out）—— 基座选型的依据** |
| `backbone_selection.py` | 零样本候选打分评测 |
| `zeroshot_vlm.py` | 零样本单次前向（读字母 logit）+ 坐标回归 |
| `inference_layouts.py` | **推理布局的延迟/精度权衡（L1 全提示 vs L2 只截图）** |
| `head_comparison.py` | **head 对照：线性探针 vs 门控 vs 无指令对照** |
| `real_distractor.py` | **真实干扰项上的 head 对照（决定性实验）** |
| `train_conditional_scorer.py` | 在 seeclick 上训练条件化打分头 |
| `perception_solver.py` | 感知 + BFS 规划器（第一版，含 bug） |
| **`final_detector.py`** | **最终检测器（去掉坐标通道 + 投影 + 双 trunk），3 seed 稳定 1.000** |
| `detector_ablation.py` | 检测器损失函数消融（C1–C5 + 稳定性扫描） |
| `trunk_attribution.py` | **塌缩归因：坐标通道 / 投影 / 死分支（A–E 变体）** |
| `head_init.py` | **head 初始 logit 尺度对塌缩的影响（0.1 / 1 / 10）** |
| `spatial_softmax.py` | 空间 softmax 定位 vs 全局分类（**负结果**：0.0607 vs 1.0000） |
| `hybrid_arch.py` | 检测器降频 + 规划器高频的成本收益曲线 |
| `eval_generalisation.py` | 分布外泛化（D0–D5） |
| `eval_browser_ui.py` | 真浏览器 grounding（Playwright） |
| `browser_loop.py` | **真浏览器闭环控制**（点完页面会变，再选下一个） |
| `ui_multiseed.py` | UI 多 seed 稳定性（3 seeds × 5 变体） |
| `diagnose_browser.py` | 真浏览器差距诊断 |
| `train_jaume_actor.py` | NPU 训练入口（SigLIP2 + Jaume-350M） |
| `run_npu.sh` | 昇腾 910B3 启动脚本（check / ui / maze） |

## 复现

```bash
cd prototype/actor

# 最终检测器（3 seed 稳定 1.000）+ 闭环
python final_detector.py --device xpu --epochs 20 --mazes 1500 --seeds 3

# 混合架构成本曲线（检测器降频）
python hybrid_final.py --device xpu --ckpt final_detector.pt

# 迷宫策略（上界参照）+ 分布外泛化
python train_maze_ce.py --device xpu --epochs 40 --mazes 1500 --frames 60 --width 48 --depth 4
python eval_generalisation.py --device xpu --epochs 40 --mazes 1500 --frames 60

# UI（含 decoy twin）+ 多 seed
python train_ui_ckpt.py --device xpu --epochs 25 --train 12000 --test 2000
python ui_multiseed.py --device xpu --epochs 15 --train 6000 --test 1500 --seeds 3

# 真浏览器
python eval_browser_ui.py --device xpu --ckpt ui_hard_model.pt
python browser_loop.py    --device xpu --ckpt ui_hard_model.pt

# 基座选型（真实 ScreenSpot）
python fetch_models.py
python backbone_probe.py --rows 200 --k 6 --epochs 8
python zeroshot_vlm.py     --rows 60 --k 6 --backbones qwen3vl_2b
python inference_layouts.py --rows 50 --k 6 --backbone qwen3vl_2b
python backbone_summary.py

# 决定性实验：指令条件化（需要 seeclick 训练数据）
python real_distractor.py --rows 400 --k 6 --epochs 12

# 诊断（为什么塌缩）
python probe_localise.py      --device xpu
python analyse_errors.py      --device xpu
python trunk_attribution.py   --device xpu --seeds 3
python head_init.py           --device xpu
```

## 昇腾 910B3

```bash
# 把 run_npu.sh 送到机器上执行
bash run_npu.sh check
bash run_npu.sh ui
```

远端路径由环境变量指定：`JAUME_REMOTE_ROOT`（默认 `/mnt/workspace`）、`JAUME_CANN_HOME`、
`JAUME_REPO_PKG`。远程编排是本地工具，不在公开仓库里。
