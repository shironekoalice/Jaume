import io
p = "tools/smoke_loop.py"
src = io.open(p, encoding="utf-8").read()
src = src.replace('cfg2.num_loops = 1',
                  'cfg2.num_loops = 1\ncfg2.vocab_size = 151647  # the checkpoint was trained with [MASK]')
io.open(p, "w", encoding="utf-8", newline="\n").write(src)

# make init_from_causal explicit about a vocab mismatch instead of silently skipping
p = "code/repo/scripts/train_lm.py"
src = io.open(p, encoding="utf-8").read()
old = """    tgt = model.backbone.state_dict()
    loaded, skipped = [], []
    for k, v in src.items():
        if k in tgt and tgt[k].shape == v.shape:
            tgt[k] = v
            loaded.append(k)
        else:
            skipped.append(k)
    model.backbone.load_state_dict(tgt)
    return loaded, skipped"""
new = """    tgt = model.backbone.state_dict()
    loaded, skipped = [], []
    for k, v in src.items():
        if k in tgt and tgt[k].shape == v.shape:
            tgt[k] = v
            loaded.append(k)
        else:
            skipped.append(k)
    model.backbone.load_state_dict(tgt)
    if skipped:
        # A skipped embedding matrix means the vocab sizes differ, and the
        # LM head is sized from the tokenizer -- so this is a real problem,
        # not a detail.  Say so loudly instead of silently running with a
        # randomly initialised embedding.
        raise RuntimeError(
            f"warm start skipped {len(skipped)} tensors, e.g. {skipped[:5]}. "
            "Check that cfg.vocab_size == len(tokenizer) and that the "
            "checkpoint is a Jaume-Causal (model_type='jaume-causal') one.")
    return loaded, skipped"""
assert old in src
src = src.replace(old, new)
io.open(p, "w", encoding="utf-8", newline="\n").write(src)
print("patched")
