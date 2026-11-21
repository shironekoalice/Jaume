import io
p = "code/repo/jaume/modeling/loop_lm.py"
src = io.open(p, encoding="utf-8").read()
i = src.find("loop_gate_param")
print("first mention at", i)
print(src[max(0,i-500):i+600])
