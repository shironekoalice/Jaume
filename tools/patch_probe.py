import io
p = "code/repo/scripts/probe_loop.py"
src = io.open(p, encoding="utf-8").read()
old = """    tokenizer = load_tokenizer(tokdir)
    cfg.vocab_size = len(tokenizer)"""
new = """    tokenizer = load_tokenizer(tokdir)
    # The causal twin's vocabulary includes a [MASK] token appended after the
    # base tokenizer was built; without it the embedding matrix is one row
    # short and the warm start drops it.
    tokenizer.add_tokens(["[MASK]"])
    cfg.vocab_size = len(tokenizer)"""
assert old in src
src = src.replace(old, new)
old = """    loaded = [k for k, v in src.items() if k in tgt and tgt[k].shape == v.shape]
    for k in loaded:
        tgt[k] = src[k]
    model.backbone.load_state_dict(tgt)"""
new = """    loaded = [k for k, v in src.items() if k in tgt and tgt[k].shape == v.shape]
    for k in loaded:
        tgt[k] = src[k]
    model.backbone.load_state_dict(tgt)
    if len(loaded) != len(src):
        print(f"[probe] WARNING: {len(src)-len(loaded)} tensors skipped, e.g. "
              f"{[k for k in src if k not in loaded][:5]}", flush=True)"""
assert old in src
src = src.replace(old, new)
io.open(p, "w", encoding="utf-8", newline="\n").write(src)
print("patched probe vocab")
