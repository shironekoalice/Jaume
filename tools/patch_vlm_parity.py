import io
p = r"code\repo\scripts\train_vlm_probe.py"
s = io.open(p, encoding="utf-8").read()

anchor = "def to_tensor(img):"
assert anchor in s
gen = '''GRID = 4
CELL = IMG // GRID


def make_image_parity(n_on, mod, rng):
    """A 4x4 grid of cells, `n_on` lit red, the rest black.

    Two properties matter:
      * the answer is NOT present as a token anywhere in the input (unlike
        the chain task, whose answer was literally one of the symbols);
      * producing it requires accumulating a running state over cells, the
        serial component attention cannot reduce in parallel.
    Parity (mod=2) is the canonical non-parallelisable boolean function.
    """
    cells = [1] * n_on + [0] * (GRID * GRID - n_on)
    rng.shuffle(cells)
    img = [[[0, 0, 0] for _ in range(IMG)] for _ in range(IMG)]
    for idx, on in enumerate(cells):
        if not on:
            continue
        gy, gx = divmod(idx, GRID)
        for y in range(gy * CELL + 1, (gy + 1) * CELL - 1):
            for x in range(gx * CELL + 1, (gx + 1) * CELL - 1):
                img[y][x][0] = 255
    return img, n_on % mod


def to_tensor(img):'''
s = s.replace(anchor, gen, 1)

old = '''    ap.add_argument("--task", choices=["count", "color_count"],
                    default="count")'''
new = '''    ap.add_argument("--task",
                    choices=["count", "color_count", "parity", "count_mod3"],
                    default="count")'''
assert old in s
s = s.replace(old, new, 1)

old = '''            target = (i % C) + 1
            q = SHAPES[i % 3]
            if args.task == "color_count":'''
new = '''            if args.task in ("parity", "count_mod3"):
                mod = 2 if args.task == "parity" else 3
                n_on = (i * 7 + (i // 16)) % (GRID * GRID)
                img, lab = make_image_parity(n_on, mod, rng)
                out.append((to_tensor(img), ("mod", mod), lab))
                continue
            target = (i % C) + 1
            q = SHAPES[i % 3]
            if args.task == "color_count":'''
assert old in s
s = s.replace(old, new, 1)

old = '''    head = nn.Linear(d, 16).to(device)'''
new = '''    n_out = 3 if args.task == "count_mod3" else 16
    head = nn.Linear(d, n_out).to(device)'''
assert old in s
s = s.replace(old, new, 1)

old = '''    def make_q(k):
        if isinstance(k, tuple):
            return "how many %s %s ?" % (k[1], k[0])
        return "how many %s ?" % k
    if args.task == "color_count":
        keys = [(sh, co) for sh in SHAPES for co in COLORS]
    else:
        keys = list(SHAPES)'''
new = '''    def make_q(k):
        if isinstance(k, tuple) and k[0] == "mod":
            kind = "parity" if k[1] == 2 else "count mod 3"
            return "what is the %s of the red cells ?" % kind
        if isinstance(k, tuple):
            return "how many %s %s ?" % (k[1], k[0])
        return "how many %s ?" % k
    if args.task in ("parity", "count_mod3"):
        keys = [("mod", 2 if args.task == "parity" else 3)]
    elif args.task == "color_count":
        keys = [(sh, co) for sh in SHAPES for co in COLORS]
    else:
        keys = list(SHAPES)'''
assert old in s
s = s.replace(old, new, 1)

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("patched parity task")
