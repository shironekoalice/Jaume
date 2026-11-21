import io
p = "code/repo/scripts/train_lm_planning.py"
src = io.open(p, encoding="utf-8").read()
old = """def evaluate(model, val_blocks, device, loss_chunk=0, max_batches=8,
             batch_size=8):"""
new = """def evaluate(model, val_blocks, device, pad_id, loss_chunk=0, max_batches=8,
             batch_size=8):"""
assert old in src
src = src.replace(old, new)
src = src.replace("        pad = cfg.pad_token_id\n", "        pad = pad_id\n")
src = src.replace("    vl = evaluate(model, val, device)",
                  "    vl = evaluate(model, val, device, cfg.pad_token_id)")
src = src.replace("                    vl = evaluate(model, val, device)",
                  "                    vl = evaluate(model, val, device, cfg.pad_token_id)")
io.open(p, "w", encoding="utf-8", newline="\n").write(src)
print("ok")
