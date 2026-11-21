import io
p = "code/repo/scripts/probe_loop.py"
src = io.open(p, encoding="utf-8").read()
old = """    for T in depths:
        model.backbone.num_loops = T"""
new = """    for T in depths:
        model.backbone.num_loops = T
        # The gate parameter is sized num_loops-1 (round 0 is pinned), so
        # sweeping T on a gate-trained checkpoint needs the gate resized to
        # match.  Truncate or extend with 1.0: extending keeps the sweep
        # honest for rounds the model has never seen (they behave as plain
        # loop rounds), truncating drops rounds the model did train.
        gp = getattr(model.backbone, "loop_gate_param", None)
        if gp is not None and gp.shape[0] != max(0, T - 1):
            with torch.no_grad():
                if T - 1 <= 0:
                    model.backbone.loop_gate_param = None
                else:
                    newg = torch.ones(max(0, T - 1), device=gp.device,
                                      dtype=gp.dtype)
                    k = min(gp.shape[0], T - 1)
                    newg[:k] = gp[:k]
                    model.backbone.loop_gate_param = torch.nn.Parameter(newg)"""
assert old in src
src = src.replace(old, new)
io.open(p, "w", encoding="utf-8", newline="\n").write(src)
print("patched probe gate resize")
