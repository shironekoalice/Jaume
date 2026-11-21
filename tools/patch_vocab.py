import io
p = "code/repo/scripts/train_lm.py"
src = io.open(p, encoding="utf-8").read()
old = """    tokenizer = load_tokenizer(tokdir)
    cfg.vocab_size = len(tokenizer)"""
new = """    tokenizer = load_tokenizer(tokdir)
    # The causal twin was trained with a [MASK] token appended to the
    # vocabulary (151646 -> 151647), so the tokenizer must be resized the
    # same way before the embedding matrix can be warm-started.  This mirrors
    # train_mlm_causal.py; without it the embedding row count is off by one
    # and the warm start silently drops the whole embedding matrix.
    num_added = tokenizer.add_tokens(["[MASK]"])
    cfg.vocab_size = len(tokenizer)
    if num_added:
        print(f"[lm] tokenizer resized to {cfg.vocab_size} "
              f"(+{num_added} [MASK])", flush=True)"""
assert old in src
src = src.replace(old, new)
io.open(p, "w", encoding="utf-8", newline="\n").write(src)
print("patched: vocab resize")
