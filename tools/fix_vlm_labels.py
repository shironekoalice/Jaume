import io
p = "code/repo/scripts/train_vlm_probe.py"
s = io.open(p, encoding="utf-8").read()
old = """    def gen(n, base):
        out = []
        for i in range(n):
            k = random.Random(base + i).randint(1, args.max_shapes)
            q = SHAPES[i % 3]
            img, cnt = make_image(k, q, base + i)
            out.append((to_tensor(img), q, cnt))
        return out"""
new = """    def gen(n, base):
        # The query shape is FORCED to appear at least once.  Without it the
        # answer distribution is heavily skewed towards 0 and 1 (35%/31% at
        # max_shapes=6), so \"always answer 1\" scores 0.355 and accuracy
        # numbers are not comparable across runs.
        out = []
        for i in range(n):
            rng = random.Random(base + i)
            k = rng.randint(2, args.max_shapes)
            q = SHAPES[i % 3]
            # force the query shape into the image
            img, cnt = make_image(k, q, base + i, force_first=q)
            out.append((to_tensor(img), q, cnt))
        return out"""
assert old in s
s = s.replace(old, new)

old2 = """def make_image(n_shapes, query_shape, seed):
    rng = random.Random(seed)
    img = [[[0, 0, 0] for _ in range(IMG)] for _ in range(IMG)]
    counts = {s: 0 for s in SHAPES}
    placed = []
    for _ in range(n_shapes):
        s = rng.choice(SHAPES)"""
new2 = """def make_image(n_shapes, query_shape, seed, force_first=None):
    rng = random.Random(seed)
    img = [[[0, 0, 0] for _ in range(IMG)] for _ in range(IMG)]
    counts = {s: 0 for s in SHAPES}
    placed = []
    for idx in range(n_shapes):
        s = force_first if (idx == 0 and force_first) else rng.choice(SHAPES)"""
assert old2 in s
s = s.replace(old2, new2)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("patched: balanced labels")
