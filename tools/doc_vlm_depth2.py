import io
p = r"docs/vlm-spec.md"
s = io.open(p, encoding="utf-8").read()
anchor = "## 5. 资源"
assert anchor in s
add = """## 4b. 深度扫描结果：视觉计数同样在 8L 饱和（2026-11-08）

Stage 0 的任务是"颜色 x 形状计数"（13 类答案）。在它上面做 8L/16L/32L
深度扫描，检验视觉通道是否会制造文本任务没有的深度需求：

| 深度 | 最好 val acc | 每 epoch |
|---|---|---|
| 8L | 0.983 | 56s |
| 16L | 0.986 | 96s |
| 32L | 0.987 | 195s |

16L 只比 8L 高 0.003（阈值 0.01），32L 再高 0.001 但耗时翻倍。
**视觉计数没有制造深度需求**，和文本侧的 `loop-spec.md` 8.9/8.11b 一致。

含义：VLM 方向要继续，必须换"真正需要多步空间推理"的任务
（如多跳视觉关系、图结构或多图对比），而不是靠堆层数解决问题。

## 5. 资源"""
s = s.replace(anchor, add, 1)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
