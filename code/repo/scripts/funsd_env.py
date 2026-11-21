"""FUNSD: real scanned forms for document understanding.

Each image is a real 762x1000 grayscale form. Annotations give text blocks
with a label in {question, answer, header, other}. We turn this into two task
families that exercise the dual tower on REAL data:

  funsd_read   : "What is the text in the box at ...?"  -> exact string
                 (tests glyph reading on real scan noise)
  funsd_find   : "Which label is 'X'?"                  -> question/answer/other
                 (tests form structure understanding)

Unlike the synthetic suites, nothing here is rendered by us: the noise,
skew, and font variation are real.
"""
import json
import os
import random
from PIL import Image

LABELS = ["question", "answer", "header", "other"]
L2I = {l: i for i, l in enumerate(LABELS)}


def _load_annotations(root):
    out = []
    ann_dir = os.path.join(root, "annotations")
    img_dir = os.path.join(root, "imgs")
    for split in ("training", "test"):
        d = os.path.join(ann_dir, split)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".json"):
                continue
            stem = f[:-5]
            img_path = os.path.join(img_dir, stem + ".png")
            if not os.path.exists(img_path):
                continue
            with open(os.path.join(d, f), encoding="utf-8") as fh:
                data = json.load(fh)
            out.append((img_path, data.get("form", [])))
    return out


def build(n, root, seed=0, max_text=24):
    """Return items: (PIL image, family, which, answer, opts, question)."""
    docs = _load_annotations(root)
    if not docs:
        raise RuntimeError("no FUNSD annotations under %s" % root)
    rng = random.Random(seed)
    out = []
    guard = 0
    while len(out) < n and guard < n * 20:
        guard += 1
        img_path, form = rng.choice(docs)
        if not form:
            continue
        item = rng.choice(form)
        text = (item.get("text") or "").strip()
        label = item.get("label") or "other"
        if not text or label not in L2I:
            continue
        img = Image.open(img_path).convert("RGB")
        box = item.get("box")
        if rng.random() < 0.5:
            # read the text: crop the target box so the question is answerable.
            # Reading a 762x1000 page downscaled to 224 destroys the glyphs
            # (see docs/ocr-diagnosis.md); a crop keeps native resolution.
            if not box or len(box) != 4:
                continue
            x0, y0, x1, y1 = [int(v) for v in box]
            pad = 6
            crop = img.crop((max(0, x0 - pad), max(0, y0 - pad),
                             min(img.width, x1 + pad), min(img.height, y1 + pad)))
            if crop.width < 8 or crop.height < 8:
                continue
            img = crop
            ans = text[:max_text]
            q = "What is the text of this form field?"
            fam, which, opts = "funsd_read", "text", "free"
        else:
            # classify the block type
            ans = label
            q = "Is this form field a question, answer, header, or other?"
            fam, which, opts = "funsd_find", "label", LABELS
        out.append((img, fam, which, ans, opts, q))
    return out


FUNSD_FAMILIES = ["funsd_read", "funsd_find"]


def crop_box(item):
    """FUNSD boxes are [x0,y0,x1,y1]."""
    b = item.get("box")
    if not b or len(b) != 4:
        return None
    return tuple(int(v) for v in b)
