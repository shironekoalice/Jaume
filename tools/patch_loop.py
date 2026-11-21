import io
p = "code/repo/jaume/modeling/loop_lm.py"
src = io.open(p, encoding="utf-8").read()
old = """        self.layers = nn.ModuleList(
            [CausalEncoderLayer(config) for _ in range(n_unique)])"""
new = """        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [CausalEncoderLayer(config) for _ in range(n_unique)])"""
assert old in src
src = src.replace(old, new)
io.open(p, "w", encoding="utf-8", newline="\n").write(src)
print("patched")
