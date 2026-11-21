import io
p = "code/repo/scripts/train_lm_planning.py"
src = io.open(p, encoding="utf-8").read()
i = src.index("    for ep in range(args.epochs):")
j = src.index("    vl = evaluate(model, val, device")
print(src[i:j])
