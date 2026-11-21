import io
p = "code/repo/scripts/train_lm_planning.py"
src = io.open(p, encoding="utf-8").read()
i = src.index("def evaluate(")
j = src.index("def main():")
print(src[i:j])
