# 换机器交接说明（Jaume-Actor）

> 生成时间：2026-11-20。本机预计只剩约 1 小时。
> 目标：换一台机器后，**不需要重新推导任何东西**，直接接着跑。

---

## 0. 一分钟版

```bash
git clone https://modelscope.cn/models/LightNeko/Jaume.git
cd Jaume

# 1. 取基座权重（ModelScope，本机 HuggingFace 不可达）
cd prototype/actor && python fetch_models.py && cd ../..

# 2. 取数据（ScreenSpot 评测集 + seeclick 训练集）
python - <<'PY'
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download("rootsautomation/ScreenSpot", repo_type="dataset",
                  local_dir="prototype/actor/data/screenspot")
snapshot_download("moondream/seeclick", repo_type="dataset",
                  allow_patterns=["data/train-00000-of-00255.parquet"],
                  local_dir="prototype/actor/data/seeclick")
PY

# 3. 复现全部结论（约 30 分钟，见 §3 的时间表）
cd prototype/actor && python check_consistency.py
```

**必须先满足的环境条件**（§1 有完整列表）。

### 仓库里有什么（换机器时别漏）

| 路径 | 内容 | 体积 | 是否进 git |
|---|---|---|---|
| `code/repo/` | **jaume 包本体**：`jaume/modeling/`（backbone、MoE、attention、kv_cache、decision）、`jaume/omni/`、`scripts/`、`tests/`、`configs/` | ~24 MB | 是 |
| `code/jaume/` | 另一份包副本（内容与 `code/repo/jaume/` 相同） | — | 部分 |
| `code/jaume-repo.tar.gz` | 打包快照 | 4 MB | 是 |
| `prototype/actor/` | **本次新增**：30 个脚本 + 30 个 json 结果 | ~1 MB | 是（不含权重/数据） |
| `docs/` | 研究记录（本方向 3 份 + 历史 40+ 份） | ~1 MB | 是 |
| `prototype/actor/models/` | 基座权重（SigLIP2 / Qwen3-VL / InternVL2） | **13 GB** | **否**，重新拉 |
| `prototype/actor/data/` | ScreenSpot 3 分片 + seeclick 1 分片 | **1.1 GB** | **否**，重新拉 |

> `code/` 里 `code/repo/` 是权威副本，`code/jaume/` 是同一内容的镜像。
> actor 的 NPU 脚本通过 `sys.path.insert` 指向 `code/repo`（见
> `train_jaume_actor.py` 顶部的 `REPO` 常量）。

---

## 1. 环境依赖（本机修复记录，新机器照做）

| 包 | 问题 | 处理 |
|---|---|---|
| `torchvision` | 0.20.1 与 `torch` 2.14.1 ABI 不匹配，报 `operator torchvision::nms does not exist` | `pip install --no-deps torchvision==0.29.2` |
| `timm` | 缺失 | `pip install timm` |
| HuggingFace | 本机不可达 | **所有权重走 ModelScope**（`fetch_models.py`） |
| `numpy` | 2.x 移除了 `np.trapz` | AUROC 改用秩公式（已在代码里） |
| `transformers` | 4.57 与 Florence-2 的 remote code 不兼容（cache 布局变了） | Florence-2 不纳入候选 |
| SigLIP2 checkpoint | 旧 config 格式：`get_image_features` 不存在，`model(**pixel_values)` 要求 `input_ids` | 手动跑 `vision_model` / `text_model`（代码里已封装） |
| playwright | 系统 Chrome 用默认 headless 参数会立刻退出 | 必须加 `--headless=new` |

**已验证可用**：torch 2.14.1+xpu、transformers 4.57.6、Pillow 13、numpy 2.4.6、
modelscope 1.34.0、playwright + 系统 Chrome。

---

## 2. 代码在哪、做什么

全部在 `prototype/actor/`，30 个脚本。按用途分四组：

### 2.1 架构原型（游戏 + 合成 UI）

| 文件 | 作用 |
|---|---|
| `envs.py` / `envs_hard.py` | 合成 UI 任务（两版，含 decoy twin）与迷宫环境 |
| `model.py` | `UIActor`（候选打分）、`MazeActor`、`SpatialCNN` |
| `final_detector.py` | **最终检测器（3 seed 稳定 1.000）** |
| `hybrid_final.py` | 检测器降频 + 规划高频的成本曲线 |
| `train_maze_ce.py` / `train_maze_stack.py` / `train_maze_split.py` | 迷宫策略与消融 |
| `probe_localise.py` / `analyse_errors.py` / `trunk_attribution.py` / `head_init.py` | 诊断塌缩原因 |
| `eval_generalisation.py` | 分布外泛化（D0–D5） |
| `spatial_softmax.py` | 空间 softmax 对照（负结果） |
| `eval_browser_ui.py` / `browser_loop.py` | **真浏览器 grounding 与闭环控制** |

### 2.2 基座选型与条件化打分头（真实 GUI 数据）

| 文件 | 作用 |
|---|---|
| `fetch_models.py` | 从 ModelScope 拉基座权重 |
| `screenspot_data.py` | ScreenSpot 载入 + 候选集构造 |
| `backbone_registry.py` | 基座注册表（SigLIP2 / VLM 的零样本与探针接口） |
| `backbone_probe.py` | **线性探针（held-out）—— 选型依据** |
| `backbone_selection.py` / `backbone_summary.py` | 零样本评测与汇总 |
| `zeroshot_vlm.py` / `inference_layouts.py` | 零样本单次前向与推理布局的延迟/精度权衡 |
| `head_comparison.py` / `real_distractor.py` | **head 对照（决定性实验）** |
| `train_conditional_scorer.py` | 条件化打分头训练（研究版，含特征预计算） |
| **`train_gated_head.py`** | **门控头训练 + 只存 head（部署版）** |
| **`eval_browser_gated.py`** | **门控头在真实 DOM 上的评测（带无指令对照）** |

### 2.3 一致性校验

`check_consistency.py` —— **38 项检查**，验证两份报告里每个数字都有 json 支撑。
改任何数字后必须重跑。

---

## 3. 复现全部结论的时间表

| 实验 | 命令 | 时间（本机 XPU） | 产出 |
|---|---|---|---|
| 最终检测器（3 seed） | `python final_detector.py --epochs 20 --mazes 1500 --seeds 3` | ~25 min | `final_detector_result.json` |
| 混合架构曲线 | `python hybrid_final.py` | ~3 min | `hybrid_result.json` |
| 真浏览器 grounding | `python eval_browser_ui.py --ckpt ui_hard_model.pt` | ~1 min | `browser_ui_result.json` |
| 真浏览器闭环 | `python browser_loop.py --ckpt ui_hard_model.pt` | ~1 min | `browser_loop_result.json` |
| UI 多 seed | `python ui_multiseed.py --epochs 15 --train 6000 --seeds 3` | ~60 min | `ui_multiseed_result.json` |
| 泛化 D0–D5 | `python eval_generalisation.py --epochs 40 --mazes 1500` | ~35 min | `maze_generalisation.json` |
| 基座探针（3 个） | `python backbone_probe.py --rows 200 --k 6` × 3 | ~25 min | `probe_*.json` |
| 零样本 VLM | `python zeroshot_vlm.py --rows 60` | ~10 min | `zs_qwen3vl.json` |
| 推理布局 | `python inference_layouts.py --rows 50` | ~20 min | `inference_layouts.json` |
| **真实干扰项对照** | `python real_distractor.py --rows 400 --k 6 --epochs 12` | ~25 min | `real_distractor_result.json` |
| 校验 | `python check_consistency.py` | <1 min | — |

**合计约 4 小时**（含数据下载）。有 NPU 的话视觉编码部分会快很多。

---

## 4. 已确认的结论（换机器不需要重新验证）

### 4.1 架构（`docs/jaume-actor-prototype.md`）

| 结论 | 数字 |
|---|---|
| 感知 + 符号规划器 > 学出来的策略 | **1.000**（3/3 seed，最差 seed 也是 1.000） vs 0.890 |
| 真浏览器 grounding | **1.000**（20/20） |
| 真浏览器闭环控制 | **1.000**（6/6，含两步任务） |
| 混合架构（检测降频 1/16） | 1.000，**0.85 ms/tick（1175 Hz）** |
| 空间 softmax 定位（负结果） | 0.0607 vs 全局分类 1.0000 |
| 帧堆叠 / lookahead（负结果） | 都比纯单帧策略更差 |
| 塌缩的机制 | **塌缩解永远是「不看图」**：坐标通道 → 最差 seed 0.0213；head 初始 logit ×10 → 0.0192 |

### 4.2 基座选型（`docs/jaume-actor-backbone-selection.md`）

| 基座 | 参数 | AUROC | top-1 | 地板 | 每 crop |
|---|---|---|---|---|---|
| **SigLIP2-base** | **93M** | 0.8991 | **0.8500** | 0.1667 | **44 ms** |
| Qwen3-VL-2B | 2B | 0.9472 | 0.8056 | 0.1667 | 481 ms |
| Qwen2-VL-2B | 2B | 0.9046 | 0.6667 | 0.1667 | 580 ms |

**选 SigLIP2-base。**

### 4.3 指令条件化（决定性实验）

| head | 参数 | held-out top-1 | 地板 |
|---|---|---|---|
| 线性探针（指令仅拼接） | 1.5K | 0.1846 | 0.1667 |
| **门控头（指令调制视觉）** | 2.4M | **0.6483** | 0.1667 |
| **门控头但去掉指令（对照）** | 2.4M | 0.1827 | 0.1667 |

### 4.4 推理布局（Qwen3-VL-2B）

| 布局 | 准确率 | 延迟 | 每次前向图数 |
|---|---|---|---|
| 截图 + K 候选裁剪，读字母 logit | 0.480 | **8998 ms** | 7 |
| 只截图 + 坐标回归 | 0.140 | 12329 ms | 1 |

---

## 5. 下一步（按优先级，都写清楚了判据）

| 优先级 | 事项 | 判据 | 需要算力 |
|---|---|---|---|
| **P0** | **阶段 2 提速**：现在 862 ms/crop（K=8 → 7 秒）。批量化 / 更小的 OCR 模型 / 只读 top-3 候选 | 决策延迟 < 200 ms | **大**（要 NPU 批量推理） |
| **P0** | **阶段 1 自给**：不用 DOM 给框，用检测器从像素出候选（真 DOM 里没有 DOM 的场景） | 检测器 recall > 0.9 | 中 |
| **P1** | 换更小的 OCR 专用模型（UI 字体域训练，不是 TrOCR 印刷体） | OCR 精确匹配 > 0.95 且 < 50 ms/crop | **大** |
| **P1** | 门控增益在 **ScreenSpot** 上复现 | 门控 vs 线性探针，真实干扰项，held-out | 小 |
| **P1** | 候选集放大到 **K=20**（真实 DOM 规模） | top-1 在 K=20 时的表现 | 小 |
| **P1** | 训练数据扩到多个 seeclick 分片 + OS-Atlas | 门控增益是否随数据量继续涨 | **大** |
| **P2** | **用 NPU（910B3）训大门控头** | 当前 2.4M 头；扩到 20M+ 看 top-1 是否继续涨 | **大** |
| **P2** | 用仓库自有的 Jaume-350M + SigLIP2 配方做对照 | 与纯 SigLIP2 头比 held-out top-1 | **大** |

**P0 的两项和 P1 的两项是需要 NPU 的部分**
（仓库已有 `run_npu.sh` 与远程编排，`tools/npu-ssh.ps1` 走 SSH 隧道；
主机、端口与密钥由环境变量提供，仓库里不保存）。

> **为什么需要 NPU**：现在阶段 2 的 862 ms/crop 是**逐个 generate**，
> 在 XPU 上没法批量。NPU 上批量读 K 张小图 + 一个更小的 OCR 头，
> 是把决策延迟从 7 秒压到 100 ms 量级的唯一现实路径。
> 这也是本项目第一次真正需要大算力 —— 前面所有实验都是探针级别的开销。

### 5.1 ⚠️ 真 DOM 的关键发现（换机器后不要重犯）

**这一节是本轮最重要的产出，它推翻了一个中间结论。**

把训练好的门控头接到**真实 DOM 候选集**上，top-1 = **0.0000**，
而且**无指令对照组也是 0.0000**。两个都是 0，说明问题在特征不在头。

最小复现：把 "Action 0"…"Delete" 六个按钮渲染成图，裁剪后与对应指令算
SigLIP2 的余弦相似度 —— **6 个只对 1 个**。

> **SigLIP2 读不出按钮上的文字。它是图文对齐模型，不是 OCR 模型。**
> 而真实 DOM 里，元素的身份就是它的标签。

真 DOM 上的四条路线（同一批页面、同样的候选集）：

| 方案 | top-1 | 地板 | 延迟/决策 |
|---|---|---|---|
| SigLIP2 + 训练过的门控头 | 0.0000 | 0.125 | 9.6 ms（只有头） |
| Qwen3-VL-2B，K 张裁剪塞进一个 prompt | 0.2857 | 0.125 | **16530 ms** |
| TrOCR-small + 字符串匹配 | 0.1818 | 0.056 | 1659 ms/页 |
| **Qwen3-VL-2B，一张裁剪一次 + 匹配** | **1.0000** | 0.125 | 862 ms/crop |

**能用的结构是两阶段**：

```
阶段 1  定位   候选框（DOM 给，或训练一个检测器）
阶段 2  确认   每张裁剪单独读一次文字 → 与指令做字符串匹配
```

**并且「一次前向塞 K 张图」被证伪**：同一个模型、同一个候选集，
一次 8 张图 = 0.2857（无指令对照 0.1429 = 地板）+ 16530 ms；
8 次单图 = 1.0000 + 862 ms/crop。
VLM 在多图 prompt 里会丢失逐图对应关系。

**骨干选型的适用边界**：

| 候选集靠什么区分 | 用什么 | 证据 |
|---|---|---|
| **外观**（图标、颜色、空间关系） | SigLIP2 更好 | top-1 0.850 vs 0.806，参数少 22 倍，每 crop 快 11 倍 |
| **文字**（真实 DOM 的按钮标签） | 必须有 OCR 能力 | OCR 精确匹配 0.875，任务 top-1 1.0000 |
| 两者都要 | **双通路**：SigLIP2 管语义 + OCR 管字形 | 仓库 `docs/vlm-dual-tower-spec.md` 的配方 |

### 5.2 本轮踩过的三个评测坑（换机器后直接绕过）

1. **探针在训练集上评估** → 「AUROC 0.99」是拟合度。加 70/30 的**截图级**划分
   （不是样本级，同一张截图的多个样本必须同侧）才有意义。
2. **随机框当干扰项** → 「无指令对照组」也拿 1.0000，指令压根没用上。
   **任何 grounding 基准都必须有「去掉指令的对照组」。**
3. **VLM 的 `generate` 回显整个聊天提示** → OCR 精确匹配率读成 0.0000，
   而任务 top-1 是 1.0000（因为回显的提示里含指令文字，"包含关系匹配"就成功了）。
   **必须先在 assistant 标记处截断再测。**

---

## 6. 换机器的三种方式

### 方式 A：换本机（最简单）
把仓库目录整个拷过去（**不含** `prototype/actor/models/` 13GB 和 `data/` 1.1GB，
这两个重新拉更快），按 §1 修环境，按 §3 复现。

### 方式 B：换 NPU 服务器（推荐）
仓库里已有远程编排：

```bash
# 本机（隧道开好时）
powershell -File tools/npu-ssh.ps1 -ScriptFile prototype/actor/run_npu.sh -RunCmd "check"
powershell -File tools/npu-ssh.ps1 -ScriptFile prototype/actor/run_npu.sh -RunCmd "ui"
```

远端布局：`/mnt/workspace/Jaume/prototype/actor`，
输出 `/mnt/workspace/output/jaume-actor`，日志 `/mnt/workspace/log/jaume-actor`。

`run_npu.sh` 的模式（**已按 §4.2 的选型结论改过**，默认跑选中的基座）：

| 模式 | 做什么 | 需要算力 |
|---|---|---|
| `check` | 环境自检（torch_npu、设备、HBM、jaume 包路径、权重/数据是否就位） | 无 |
| `probe` | 线性探针（选型的最小复现） | 小 |
| **`ui`** | **训练门控头 → 在真实 DOM 上评测**（P0 全流程） | **中** |
| `maze` | 迷宫控制训练 | 中 |

> `run_npu.sh` 现在默认跑 **SigLIP2 + 门控头**，不再是 Jaume-350M 版本 ——
> 因为选型数据说 93M 的塔在 top-1 和延迟上都更优。
> `model_jaume.py`（Jaume-350M + SigLIP2 双塔）保留作对照，
> 对应仓库 `docs/vlm-dual-tower-spec.md` 的配方。

### 方式 C：混合
本机跑推理/评测（XPU 够快），NPU 跑训练。

---

## 7. 一句话总结当前状态

**架构已经定了，数据已经通了，选型已经用真实数据做完，
剩下的是把「构造基准上的 0.6483」推进到「真实 DOM 上的可部署数字」。**
所有结论都有 json 支撑并由 `check_consistency.py`（38 项）自动校验。

三个已确认的、可直接继承的判断：

1. **动作空间**：候选打分，不做坐标回归；一次前向 + argmax，不解码文本。
2. **职责划分**：小模型做感知，规划交给确定性求解器（1.000 vs 0.890）。
3. **基座**：SigLIP2-base（93M），不是 2B 的 VLM，不是仓库自己的 350M ——
   在真实 ScreenSpot 的 held-out 探针上 top-1 更高（0.850 vs 0.806），
   每 crop 快 11 倍，且参数量少 22 倍。

以及一条不能忘的验证纪律：**任何 grounding 基准都必须带「去掉指令的对照组」**，
否则「模型学会了」和「模型没看指令」会给出同样的满分（这个坑本轮踩过一次）。
