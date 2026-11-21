"""Object detection / counting with coordinate tokens (WeVisOne-style).

Our text_locate task asked "which box contains X" and forced the model to
encode position implicitly in a feature vector. WeVisOne-VL instead emits
explicit coordinate tokens over a normalised grid, which is why one head can
serve detection, counting, pointing and OCR localisation.

This environment generates multi-object scenes and asks for either a count,
normalised boxes, or centre points. Coordinates are quantised to a 32x32 grid
so the answer stays inside our character vocabulary.
"""
import random
from PIL import Image, ImageDraw

W = H = 320
SHAPES = ["circle", "square", "triangle", "diamond"]
COLORS = {
    "red": (220, 40, 40),
    "green": (40, 170, 60),
    "blue": (40, 90, 220),
    "yellow": (240, 200, 20),
}
COORD_BINS = 8


def _draw(d, shape, cx, cy, r, color):
    if shape == "circle":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif shape == "square":
        d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif shape == "triangle":
        d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=color)
    else:
        d.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)],
                  fill=color)


def _bin(v, total):
    return max(0, min(COORD_BINS - 1, int(v / total * COORD_BINS)))


def gen_scene(rng, n_target=None):
    target_shape = rng.choice(SHAPES)
    target_color = rng.choice(list(COLORS))
    n = n_target if n_target is not None else rng.randint(1, 3)
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    placed, objs = [], []
    attempts = 0
    while len(placed) < n and attempts < 300:
        attempts += 1
        r = rng.randint(14, 26)
        cx = rng.randint(r + 4, W - r - 4)
        cy = rng.randint(r + 4, H - r - 4)
        if any((cx - x) ** 2 + (cy - y) ** 2 < (r + rr + 8) ** 2
               for x, y, rr in placed):
            continue
        placed.append((cx, cy, r))
        _draw(d, target_shape, cx, cy, r, COLORS[target_color])
        objs.append((cx, cy, r))
    for _ in range(rng.randint(0, 2)):
        for _try in range(50):
            r = rng.randint(12, 22)
            cx = rng.randint(r + 4, W - r - 4)
            cy = rng.randint(r + 4, H - r - 4)
            if any((cx - x) ** 2 + (cy - y) ** 2 < (r + rr + 8) ** 2
                   for x, y, rr in placed):
                continue
            placed.append((cx, cy, r))
            alt = rng.choice([s for s in SHAPES if s != target_shape])
            altc = rng.choice([c for c in COLORS if c != target_color])
            _draw(d, alt, cx, cy, r, COLORS[altc])
            break
    return img, target_shape, target_color, objs


def boxes_text(objs):
    parts = []
    for (cx, cy, r) in objs:
        x1, y1 = _bin(cx - r, W), _bin(cy - r, H)
        x2, y2 = _bin(cx + r, W), _bin(cy + r, H)
        parts.append("%d%d%d%d" % (x1, y1, x2, y2))
    return " ".join(parts)


def points_text(objs):
    return " ".join("%d%d" % (_bin(cx, W), _bin(cy, H))
                    for (cx, cy, r) in objs)


GENERATORS = ["det_count", "det_boxes", "det_point"]
DET_FAMILIES = GENERATORS


def build(n, seed):
    out = []
    for i in range(n):
        rng = random.Random(seed * 1000003 + i)
        img, shape, color, objs = gen_scene(rng)
        fam = GENERATORS[i % len(GENERATORS)]
        if fam == "det_count":
            ans = str(len(objs))
            q = "How many %s %s are there?" % (color, shape)
        elif fam == "det_boxes":
            ans = boxes_text(objs)
            q = "Where are the %s %s? Give boxes." % (color, shape)
        else:
            ans = points_text(objs)
            q = "Point at each %s %s." % (color, shape)
        out.append((img, fam, shape, ans, "free", q))
    return out
