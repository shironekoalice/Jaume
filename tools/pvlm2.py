import io
p = "code/repo/scripts/train_vlm_probe.py"
s = io.open(p, encoding="utf-8").read()

# widen the head to cover counts up to 16 (already 16) and add a harder
# generator: colour-conditioned counting over 10 targets
helper = (
    "COLORS = ['red', 'green', 'blue']\n"
    "\n"
    "\n"
    "def make_image_color(target, query_shape, query_color, rng):\n"
    "    img = [[[0, 0, 0] for _ in range(IMG)] for _ in range(IMG)]\n"
    "    shapes = [(query_shape, query_color)] * target\n"
    "    for _ in range(rng.randint(0, 2)):\n"
    "        shapes.append((rng.choice([x for x in SHAPES\n"
    "                                   if x != query_shape]),\n"
    "                       rng.choice(COLORS)))\n"
    "    rng.shuffle(shapes)\n"
    "    placed = []\n"
    "    for sh, col in shapes:\n"
    "        for _try in range(30):\n"
    "            cx = rng.randint(4, IMG - 5)\n"
    "            cy = rng.randint(4, IMG - 5)\n"
    "            if all((cx - x) ** 2 + (cy - y) ** 2 > 36\n"
    "                   for x, y in placed):\n"
    "                placed.append((cx, cy))\n"
    "                ci = COLORS.index(col)\n"
    "                r = 3\n"
    "                for y in range(max(0, cy - r), min(IMG, cy + r + 1)):\n"
    "                    for x in range(max(0, cx - r), min(IMG, cx + r + 1)):\n"
    "                        dx, dy = x - cx, y - cy\n"
    "                        if sh == 'circle' and dx * dx + dy * dy > r * r:\n"
    "                            continue\n"
    "                        if sh == 'square' and (abs(dx) > 2 or abs(dy) > 2):\n"
    "                            continue\n"
    "                        if sh == 'triangle' and (dy > 0 or\n"
    "                                                 abs(dx) > (dy + 3)):\n"
    "                            continue\n"
    "                        img[y][x][ci] = 255\n"
    "                break\n"
    "    return img, target\n"
    "\n"
    "\n"
)
anchor = "def to_tensor(img):"
assert anchor in s
s = s.replace(anchor, helper + anchor, 1)

# CLI: --task {count, color_count}
s = s.replace(
    "    ap.add_argument('--visual-backbone',",
    "    ap.add_argument('--task', choices=['count', 'color_count'],\n"
    "                    default='count')\n"
    "    ap.add_argument('--visual-backbone',")

# gen() branch: colour counting uses the new generator + colour questions
old_gen = (
    "            target = (i % C) + 1\n"
    "            q = SHAPES[i % 3]\n"
    "            img, cnt = make_image_exact(target, q, rng)\n"
    "            out.append((to_tensor(img), q, cnt))"
)
new_gen = (
    "            target = (i % C) + 1\n"
    "            q = SHAPES[i % 3]\n"
    "            if args.task == 'color_count':\n"
    "                qc = COLORS[i % 3]\n"
    "                img, cnt = make_image_color(target, q, qc, rng)\n"
    "                out.append((to_tensor(img), (q, qc), cnt))\n"
    "            else:\n"
    "                img, cnt = make_image_exact(target, q, rng)\n"
    "                out.append((to_tensor(img), q, cnt))"
)
assert old_gen in s
s = s.replace(old_gen, new_gen)

# question ids must handle the tuple form
old_q = (
    "    q_ids = {q: torch.tensor(\n"
    "        tok('how many %s ?' % q,\n"
    "            add_special_tokens=False)['input_ids'],\n"
    "        dtype=torch.long, device=device) for q in SHAPES}"
)
new_q = (
    "    def make_q(k):\n"
    "        if isinstance(k, tuple):\n"
    "            return 'how many %s %s ?' % (k[1], k[0])\n"
    "        return 'how many %s ?' % k\n"
    "    if args.task == 'color_count':\n"
    "        keys = [(sh, co) for sh in SHAPES for co in COLORS]\n"
    "    else:\n"
    "        keys = list(SHAPES)\n"
    "    q_ids = {k: torch.tensor(\n"
    "        tok(make_q(k), add_special_tokens=False)['input_ids'],\n"
    "        dtype=torch.long, device=device) for k in keys}"
)
assert old_q in s, "q_ids block not found"
s = s.replace(old_q, new_q)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("patched: color task")
