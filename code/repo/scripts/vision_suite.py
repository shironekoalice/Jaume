"""A basic-visual-competence suite: what can a frozen Jaume encoder see?

Why this and not OCR
--------------------
The product target is general image understanding (OCR being one capability);
UI/GUI/game recognition is a separate follow-up model. So the useful question
is which basic visual abilities the Jaume bridge already has, and where the
boundary is.

Six families, all rendered locally (no download dependency):

  attr        colour / shape / size of a single object
  count       how many objects of a kind (parallel aggregation)
  spatial     left/right/above/below relation between two objects
  compare     which of two colours has more objects
  scene       layout type: single / row / grid
  multi_attr  shape or colour of one object

Every task is multiple choice over a small fixed answer set, so accuracy is
comparable across families and a majority baseline is well defined.
"""
import math
import random
from PIL import Image, ImageDraw

W = H = 96
SHAPES = ["circle", "square", "triangle", "diamond", "star"]
COLORS = {
    "red": (220, 40, 40),
    "green": (40, 170, 60),
    "blue": (40, 90, 220),
    "yellow": (240, 200, 20),
    "purple": (150, 60, 190),
    "orange": (240, 130, 20),
}
SIZES = ["small", "large"]


def _draw_shape(d, shape, cx, cy, r, color):
    if shape == "circle":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif shape == "square":
        d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif shape == "triangle":
        d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=color)
    elif shape == "diamond":
        d.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)],
                  fill=color)
    elif shape == "star":
        pts = []
        for i in range(10):
            ang = -1.5708 + i * 0.6283
            rr = r if i % 2 == 0 else r * 0.45
            pts.append((cx + rr * math.cos(ang), cy + rr * math.sin(ang)))
        d.polygon(pts, fill=color)


def _blank():
    img = Image.new("RGB", (W, H), "white")
    return img, ImageDraw.Draw(img)


def _grid_slots(n, rng):
    slots = [(24, 24), (72, 24), (48, 50), (24, 74), (72, 74), (48, 80)]
    rng.shuffle(slots)
    return slots[:n]


# A fixed reference disc drawn in the corner of every attr image. Without a
# reference the model cannot know what "large" means in absolute pixels, and
# the size questions were the reason this family never saturated.
REF_R = 8


def gen_attr(rng):
    shape = rng.choice(SHAPES)
    color = rng.choice(list(COLORS))
    size = rng.choice(SIZES)
    # sizes are far apart so the reference disc is unambiguous
    r = 30 if size == "large" else 10
    cx, cy = 52, 56
    img, d = _blank()
    # faint background rules everywhere (never a size cue)
    for x in range(10, W, 20):
        d.line([x, 0, x, H], fill=(242, 242, 242))
    for y in range(10, H, 20):
        d.line([0, y, W, y], fill=(242, 242, 242))
    _draw_shape(d, shape, cx, cy, r, COLORS[color])
    # the reference disc: black outline, constant radius
    d.ellipse([W - 26 - REF_R, 10, W - 26 + REF_R, 10 + 2 * REF_R],
              outline=(40, 40, 40), width=2)
    which = rng.choice(["shape", "color", "size"])
    ans = {"shape": shape, "color": color, "size": size}[which]
    opts = {"shape": SHAPES, "color": list(COLORS), "size": SIZES}[which]
    return img, "attr", which, ans, opts


def gen_count(rng):
    shape = rng.choice(SHAPES)
    color = rng.choice(list(COLORS))
    n = rng.randint(1, 5)
    img, d = _blank()
    slots = _grid_slots(6, rng)
    for (cx, cy) in slots[:n]:
        _draw_shape(d, shape, cx, cy, 9, COLORS[color])
    others = rng.randint(0, 2)
    alt_sh = [s for s in SHAPES if s != shape]
    alt_co = [c for c in COLORS if c != color]
    for (cx, cy) in slots[n:n + others]:
        _draw_shape(d, rng.choice(alt_sh), cx, cy, 9,
                    COLORS[rng.choice(alt_co)])
    return img, "count", shape, str(n), [str(i) for i in range(1, 6)]


def gen_spatial(rng):
    a = rng.choice(SHAPES)
    b = rng.choice([s for s in SHAPES if s != a])
    rel = rng.choice(["left", "right", "above", "below"])
    img, d = _blank()
    off = 26
    if rel == "left":
        pa, pb = (48 - off, 48), (48 + off, 48)
    elif rel == "right":
        pa, pb = (48 + off, 48), (48 - off, 48)
    elif rel == "above":
        pa, pb = (48, 48 - off), (48, 48 + off)
    else:
        pa, pb = (48, 48 + off), (48, 48 - off)
    _draw_shape(d, a, pa[0], pa[1], 12, COLORS["red"])
    _draw_shape(d, b, pb[0], pb[1], 12, COLORS["blue"])
    return img, "spatial", a, rel, ["left", "right", "above", "below"]


def gen_compare(rng):
    c1, c2 = rng.sample(list(COLORS), 2)
    n1 = rng.randint(1, 4)
    n2 = rng.randint(1, 4)
    while n2 == n1:
        n2 = rng.randint(1, 4)
    img, d = _blank()
    slots = _grid_slots(6, rng)
    for i, (cx, cy) in enumerate(slots):
        if i < n1:
            _draw_shape(d, "circle", cx, cy, 8, COLORS[c1])
        elif i < n1 + n2:
            _draw_shape(d, "square", cx, cy, 8, COLORS[c2])
    return img, "compare", "color", (c1 if n1 > n2 else c2), [c1, c2]


def gen_scene(rng):
    layout = rng.choice(["single", "row", "grid"])
    color = rng.choice(list(COLORS))
    shape = rng.choice(SHAPES)
    img, d = _blank()
    if layout == "single":
        _draw_shape(d, shape, 48, 48, 24, COLORS[color])
    elif layout == "row":
        for i in range(4):
            _draw_shape(d, shape, 14 + i * 23, 48, 8, COLORS[color])
    else:
        for gy in range(3):
            for gx in range(3):
                _draw_shape(d, shape, 24 + gx * 24, 24 + gy * 24, 7,
                            COLORS[color])
    return img, "scene", "layout", layout, ["single", "row", "grid"]


def gen_multi_attr(rng):
    shape = rng.choice(SHAPES)
    color = rng.choice(list(COLORS))
    img, d = _blank()
    _draw_shape(d, shape, 48, 48, 22, COLORS[color])
    which = rng.choice(["shape", "color"])
    ans = shape if which == "shape" else color
    opts = SHAPES if which == "shape" else list(COLORS)
    return img, "multi_attr", which, ans, opts


GENERATORS = [gen_attr, gen_count, gen_spatial, gen_compare, gen_scene,
              gen_multi_attr]

QUESTIONS = {
    "attr": {"shape": "What shape is shown?",
             "color": "What colour is shown?",
             "size": "Is the object small or large?"},
    "count": "How many {x} are there?",
    "spatial": "Where is the {x} relative to the other object?",
    "compare": "Which colour has more objects?",
    "scene": "What layout is shown?",
    "multi_attr": {"shape": "What shape is shown?",
                   "color": "What colour is shown?"},
}


def make_question(family, which, x=None):
    q = QUESTIONS[family]
    if isinstance(q, dict):
        q = q[which]
    if "{x}" in q:
        q = q.format(x=x)
    return q


def build(n, seed):
    out = []
    for i in range(n):
        rng = random.Random(seed * 1000003 + i)
        g = GENERATORS[i % len(GENERATORS)]
        img, fam, which, ans, opts = g(rng)
        x = which if fam in ("count", "spatial") else None
        q = make_question(fam, which, x)
        out.append((img, fam, which, ans, opts, q))
    return out
