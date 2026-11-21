import io
p = r"D:\项目\Jaume\code\repo\scripts\train_vlm_probe.py"
s = io.open(p, encoding="utf-8").read()

helper = r'''COLORS = ["red", "green", "blue"]


def make_image_color(target, query_shape, query_color, rng):
    img = [[[0, 0, 0] for _ in range(IMG)] for _ in range(IMG)]
    shapes = [(query_shape, query_color)] * target
    for _ in range(rng.randint(0, 2)):
        alt = rng.choice([x for x in SHAPES if x != query_shape])
        shapes.append((alt, rng.choice(COLORS)))
    rng.shuffle(shapes)
    placed = []
    for sh, col in shapes:
        for _try in range(30):
            cx = rng.randint(4, IMG - 5)
            cy = rng.randint(4, IMG - 5)
            if all((cx - x) ** 2 + (cy - y) ** 2 > 36 for x, y in placed):
                placed.append((cx, cy))
                ci = COLORS.index(col)
                for y in range(max(0, cy - 3), min(IMG, cy + 4)):
                    for x in range(max(0, cx - 3), min(IMG, cx + 4)):
                        dx, dy = x - cx, y - cy
                        if sh == "circle" and dx * dx + dy * dy > 9:
                            continue
                        if sh == "square" and (abs(dx) > 2 or abs(dy) > 2):
                            continue
                        if sh == "triangle" and (dy > 0 or abs(dx) > (dy + 3)):
                            continue
                        img[y][x][ci] = 255
                break
    return img, target


'''
anchor = "def to_tensor(img):"
assert anchor in s, "to_tensor anchor missing"
if "def make_image_color" not in s:
    s = s.replace(anchor, helper + anchor, 1)

old_vis = "    vis = PatchEncoder(d).to(device)"
new_vis = '''    if args.visual_backbone == "cnn":
        vis = PatchEncoder(d).to(device)
    else:
        vis = ResNetStem(d, args.visual_backbone,
                         args.visual_pretrained).to(device)
    print("[vlm] visual backbone=%s" % args.visual_backbone, flush=True)'''
if old_vis in s:
    s = s.replace(old_vis, new_vis, 1)

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("done: color gen + backbone wiring")
