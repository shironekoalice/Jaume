import ast, faulthandler, time
path = "/mnt/workspace/venvs/jaume-omni/lib/python3.11/site-packages/funasr/datasets/audio_datasets/datasets.py"
t0=time.time()
with open(path, "r", encoding="utf-8") as f:
    src = f.read()
print("read", len(src), "chars in", time.time()-t0)
faulthandler.dump_traceback_later(8, exit=True)
t0=time.time()
tree = ast.parse(src, filename=path)
print("parse ok", time.time()-t0)