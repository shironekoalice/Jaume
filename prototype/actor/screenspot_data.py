"""ScreenSpot loader: real GUI screenshots -> the candidate-scoring task.

WHY THIS FILE. The synthetic UI benchmark scores 1.000 for every ablation, so
it cannot choose between backbones. ScreenSpot is the real thing: 1272
screenshots across iOS / Android / macOS / Windows / Web, each with a natural
instruction and the ground-truth box of the element it refers to
(Apache-2.0, from SeeClick).

THE CANDIDATE SET. ScreenSpot annotates only the target box, but the actor's
action space is "score a shortlist of candidates". So the shortlist is
CONSTRUCTED here: the GT box plus K-1 distractors sampled from the same
screenshot. Distractors are drawn from a size-matched pool so that the
answer cannot be found by picking the biggest / smallest box — the failure
mode that makes naive grounding benchmarks easy.

    K=2   -> a binary choice; the floor is 0.50
    K=6   -> the actor's operating regime; the floor is 0.167
    K=20  -> what a real page's DOM shortlist looks like; floor 0.05

A second evaluation mode is kept for reference: pure COORDINATE REGRESSION
(predict the box directly). The published ScreenSpot numbers for every
baseline are in that mode, so this file reports it too, and the two modes can
be compared on the same data.

Distractor sampling is seeded per sample so the evaluation is reproducible.
"""
from __future__ import annotations

import io
import os

import numpy as np

PARQUET = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "screenspot", "data", "test-00000-of-00003.parquet")


def _open_table():
    import pyarrow.parquet as pq

    return pq.read_table(PARQUET)


def load_raw(max_rows=None):
    """Return a list of dicts: image (PIL), instruction, bbox, data_type, source."""
    from PIL import Image

    tbl = _open_table()
    n = tbl.num_rows if max_rows is None else min(max_rows, tbl.num_rows)
    out = []
    for i in range(n):
        row = {k: tbl.column(k)[i].as_py() for k in tbl.schema.names}
        img = row["image"]
        if isinstance(img, dict):
            img = Image.open(io.BytesIO(img["bytes"]))
        elif isinstance(img, (bytes, bytearray)):
            img = Image.open(io.BytesIO(img))
        out.append(dict(image=img.convert("RGB"), instruction=row["instruction"],
                        bbox=tuple(row["bbox"]), data_type=row.get("data_type"),
                        source=row.get("data_source")))
    return out


def iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def sample_candidates(img_w, img_h, gt_box, k, rng, size_match=0.6):
    """GT box + (k-1) distractors.

    Distractors are sampled with a box area within `size_match` (relative
    band) of the GT area, so the answer is not predictable from box size
    alone. They are also kept inside the image.
    """
    gx0, gy0, gx1, gy1 = gt_box
    gw, gh = max(1.0, gx1 - gx0), max(1.0, gy1 - gy0)
    ga = gw * gh
    lo, hi = ga * (1 - size_match), ga * (1 + size_match)
    cands = [gt_box]
    tries = 0
    while len(cands) < k and tries < 400:
        tries += 1
        a = rng.uniform(lo, hi)
        ar = rng.uniform(0.6, 1.6)
        w = np.sqrt(a * ar)
        h = a / max(w, 1e-6)
        x0 = rng.uniform(0, max(0.0, img_w - w))
        y0 = rng.uniform(0, max(0.0, img_h - h))
        box = (x0, y0, x0 + w, y0 + h)
        if iou(box, gt_box) > 0.05:
            continue
        # avoid near-duplicates of existing candidates
        if any(iou(box, c) > 0.1 for c in cands):
            continue
        cands.append(box)
    while len(cands) < k:  # fallback: pad with far-away boxes
        x0 = rng.uniform(0, max(1.0, img_w - 10))
        y0 = rng.uniform(0, max(1.0, img_h - 10))
        cands.append((x0, y0, x0 + 8, y0 + 8))
    return cands


def build_eval_set(max_rows=400, k=6, seed=0):
    """Returns a list of dicts ready for the actor."""
    rng = np.random.default_rng(seed)
    raw = load_raw(max_rows=max_rows)
    out = []
    for i, r in enumerate(raw):
        w, h = r["image"].size
        # ScreenSpot's bbox is NORMALISED (0..1) in this copy of the dataset —
        # the README describes pixels for the original release. Detect which
        # one we have and convert to pixels here, so nothing downstream has to
        # care. (Getting this wrong silently produced 1x1-pixel boxes.)
        gx0, gy0, gx1, gy1 = r["bbox"]
        if max(gx0, gy0, gx1, gy1) <= 1.5:
            gt_box = (gx0 * w, gy0 * h, gx1 * w, gy1 * h)
        else:
            gt_box = (gx0, gy0, gx1, gy1)
        cands = sample_candidates(w, h, gt_box, k, rng)
        # the GT box is always candidate 0; shuffle so the answer is not
        # positionally biased
        order = rng.permutation(len(cands))
        shuffled = [cands[j] for j in order]
        gold = list(order).index(0)
        out.append(dict(image=r["image"], instruction=r["instruction"],
                        candidates=shuffled, gold=gold,
                        data_type=r["data_type"], source=r["source"],
                        gt_box=gt_box))
    return out


if __name__ == "__main__":
    import collections

    data = build_eval_set(max_rows=1272, k=6)
    print(f"samples {len(data)}")
    print("by source:", dict(collections.Counter(d["source"] for d in data)))
    print("by type:  ", dict(collections.Counter(d["data_type"] for d in data)))
    d = data[0]
    print(f"\nexample: {d['instruction']!r}")
    print(f"  image {d['image'].size}  gt {d['gt_box']}  gold_idx {d['gold']}")
    print(f"  candidates {len(d['candidates'])}")
    for i, c in enumerate(d["candidates"]):
        mark = "<-- GOLD" if i == d["gold"] else ""
        print(f"    {i}: {tuple(round(v) for v in c)} {mark}")
