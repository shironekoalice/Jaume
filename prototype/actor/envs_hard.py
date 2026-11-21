"""Harder UI benchmark: position disambiguation.

The first version of this benchmark was trivially solvable by the element crop
alone (all variants hit 1.000), which measured nothing. Fixed here:

  * every screen contains **decoy twins**: elements with the SAME color and
    SAME shape as the target, differing only in position band / column.
  * the instruction therefore cannot be resolved by a single crop; the model
    must combine a local appearance feature with a global position feature.
  * `heuristic_ui` (color+shape match) is no longer a winning control — it
    picks a random twin. It is reported as the *floor* to beat, and the
    oracle (color+shape+band+column) is 1.000 by construction.

Kept honest: this is still a synthetic proxy, not a real browser. Its job is
to check that the *architecture* (candidate scoring, no autoregressive
decode) can represent the decision, before spending on real web data.
"""
from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageDraw

from envs import BANDS, CROP, COLORS, COLS, IMG, SHAPES, Candidate, VOCAB, word_id

LATTICE = 3 * 2  # 3 rows x 2 cols


def _draw_shape(d, shape, box, color):
    x0, y0, x1, y1 = box
    if shape == "circle":
        d.ellipse(box, fill=color, outline=(20, 20, 20))
    elif shape == "square":
        d.rectangle(box, fill=color, outline=(20, 20, 20))
    else:
        d.polygon([((x0 + x1) // 2, y0), (x0, y1), (x1, y1)], fill=color, outline=(20, 20, 20))


def make_ui_task_hard(rng: random.Random, n_cand: int = 6):
    im = Image.new("RGB", (IMG, IMG), (248, 248, 246))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, IMG, 14], fill=(228, 230, 236))
    for i in range(3):
        y = 22 + i * 12
        d.rectangle([8, y, 8 + rng.randint(40, 110), y + 4], fill=(205, 208, 214))

    cells = [(c, r) for r in range(3) for c in range(2)]
    rng.shuffle(cells)
    cells = cells[:n_cand]

    # the target's appearance is drawn from a small pool so that twins exist
    color = rng.choice(list(COLORS))
    shape = rng.choice(SHAPES)
    ans_idx = rng.randrange(n_cand)
    tr, tc = cells[ans_idx]

    cands = []
    for i, (c, r) in enumerate(cells):
        # twins of the target share its color+shape; others get a different color
        same = (i == ans_idx) or (rng.random() < 0.45)
        cc = color if same else rng.choice([k for k in COLORS if k != color])
        sh = shape if same else (shape if rng.random() < 0.5 else rng.choice([s for s in SHAPES if s != shape]))
        cw, ch = 22, 22
        x0 = 10 + c * 58 + rng.randint(0, 6)
        y0 = 20 + r * 36 + rng.randint(0, 4)
        box = [x0, y0, x0 + cw, y0 + ch]
        _draw_shape(d, sh, box, COLORS[cc])
        crop = np.asarray(im.crop(box).resize((CROP, CROP)), dtype=np.float32) / 255.0
        cands.append(
            Candidate(
                crop=crop.transpose(2, 0, 1),
                box=np.array([box[0] / IMG, box[1] / IMG, box[2] / IMG, box[3] / IMG], dtype=np.float32),
                color=cc,
                shape=sh,
                cx=(box[0] + box[2]) / 2 / IMG,
                cy=(box[1] + box[3]) / 2 / IMG,
            )
        )

    tgt = cands[ans_idx]
    band = BANDS[min(2, int(tgt.cy * 3))]
    col = COLS[0] if tgt.cx < 0.5 else COLS[1]
    words = ["click", tgt.color, tgt.shape, "at", band, col]
    instruction = [word_id(w) for w in words]
    screen = (np.asarray(im, dtype=np.float32) / 255.0).transpose(2, 0, 1)
    return screen, cands, instruction, ans_idx


def heuristic_hard(cands, instruction) -> int:
    """Color+shape only. Cannot resolve twins -> this is the floor, not the target."""
    words = [VOCAB[i - 1] if 0 < i <= len(VOCAB) else "" for i in instruction]
    color = next((w for w in words if w in COLORS), None)
    shape = next((w for w in words if w in SHAPES), None)
    best, best_key = 0, -1.0
    for i, c in enumerate(cands):
        key = 2.0 * (c.color == color) + 1.0 * (c.shape == shape)
        if key > best_key:
            best, best_key = i, key
    return best


def oracle_hard(cands, instruction) -> int:
    """color+shape+band+column: exact by construction."""
    words = [VOCAB[i - 1] if 0 < i <= len(VOCAB) else "" for i in instruction]
    color = next((w for w in words if w in COLORS), None)
    shape = next((w for w in words if w in SHAPES), None)
    band = next((w for w in words if w in BANDS), None)
    col = next((w for w in words if w in COLS), None)
    for i, c in enumerate(cands):
        b = BANDS[min(2, int(c.cy * 3))]
        cl = COLS[0] if c.cx < 0.5 else COLS[1]
        if c.color == color and c.shape == shape and b == band and cl == col:
            return i
    return 0
