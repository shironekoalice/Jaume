import io
p = "tools/smoke_gate.py"
src = io.open(p, encoding="utf-8").read()
src = src.replace('''g = m.backbone.loop_gate_param.grad
print(f"gate grad: {g.tolist() if g is None else [round(float(x),4) for x in g]}")
print(f"round-0 grad pinned to 0: {float(g[0]) == 0.0}")''',
'''g = m.backbone.loop_gate_param.grad
print(f"gate grad (rounds 1..T-1): {[round(float(x),4) for x in g]}")
print(f"param count: {m.backbone.loop_gate_param.shape[0]} (expect T-1 = 3)")''')
io.open(p, "w", encoding="utf-8", newline=chr(10)).write(src)
print("ok")
