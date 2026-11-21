import io
p = r"D:\项目\Jaume\code\repo\scripts\train_vlm_probe.py"
lines = io.open(p, encoding="utf-8").read().split("\n")
# lines are 0-based; replace index 216..227 (1-based 217..228)
start = 216
end = 228
assert "visual_backbone" in lines[start], lines[start]
clean = [
    '    if args.visual_backbone == "cnn":',
    "        vis = PatchEncoder(d).to(device)",
    "    else:",
    "        vis = ResNetStem(d, args.visual_backbone,",
    "                         args.visual_pretrained).to(device)",
    '    print("[vlm] visual backbone=%s" % args.visual_backbone, flush=True)',
]
lines[start:end] = clean
io.open(p, "w", encoding="utf-8", newline="\n").write("\n".join(lines))
print("fixed")
