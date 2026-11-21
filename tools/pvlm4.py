import io
p = "code/repo/scripts/train_vlm_probe.py"
s = io.open(p, encoding="utf-8").read()
a = s.find("    q_ids = {q: torch.tensor(")
b = s.find("\n\n    trainable = (", a)
assert a > 0 and b > a
lines = [
 "    def make_q(k):",
 "        if isinstance(k, tuple):",
 '            return "how many %s %s ?" % (k[1], k[0])',
 '        return "how many %s ?" % k',
 '    if args.task == "color_count":',
 "        keys = [(sh, co) for sh in SHAPES for co in COLORS]",
 "    else:",
 "        keys = list(SHAPES)",
 "    q_ids = {k: torch.tensor(",
 '        tok(make_q(k), add_special_tokens=False)["input_ids"],',
 "        dtype=torch.long, device=device) for k in keys}",
 "",
]
new = "\n".join(lines)
io.open(p, "w", encoding="utf-8", newline="\n").write(s[:a] + new + s[b:])
print("done")
