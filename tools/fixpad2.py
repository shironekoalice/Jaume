import io
p = "code/repo/scripts/train_lm_planning.py"
src = io.open(p, encoding="utf-8").read()
src = src.replace("            pad = pad_id\n", "            pad = cfg.pad_token_id\n")
io.open(p, "w", encoding="utf-8", newline="\n").write(src)
print("ok")
