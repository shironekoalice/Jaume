import io
p = "tools/smoke_loop.py"
src = io.open(p, encoding="utf-8").read()
src = src.replace(
    '.mscache/LightNeko--Jaume/output/350m-causal-mlm/final/pytorch_model.bin',
    '.mscache/LightNeko/Jaume/output/350m-causal-mlm/final/pytorch_model.bin')
io.open(p, "w", encoding="utf-8", newline="\n").write(src)
print("ok")
