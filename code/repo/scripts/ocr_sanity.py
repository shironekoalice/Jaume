"""Sanity check: is the text_locate task actually solvable from pixels?

If a trivial baseline (e.g. counting dark pixels per quadrant) can solve it,
the task is fine and the vision towers are the problem. If even that fails,
the task generation is broken.
"""
import sys
sys.path.insert(0, "/home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master/code/repo/scripts")
from ocr_suite import build
import numpy as np
from collections import Counter

items = [b for b in build(400, seed=31) if b[1] == "text_locate"]
print("n =", len(items))
print("label dist:", Counter(b[3] for b in items))

# The words differ, but every box has the SAME structure. Is there any pixel
# signal that identifies which box holds the target word?
for i in range(3):
    img, fam, which, ans, _, q = items[i]
    a = np.asarray(img.convert("L"))
    print("\nsample", i, "answer:", ans, "| question:", q)
    print("  image", a.shape, "min/max", a.min(), a.max())
    # ink per quadrant (words are dark on white)
    H, W = a.shape
    for name, (ys, xs) in {
        "TL": (slice(0, H//2), slice(0, W//2)),
        "TR": (slice(0, H//2), slice(W//2, W)),
        "BL": (slice(H//2, H), slice(0, W//2)),
        "BR": (slice(H//2, H), slice(W//2, W)),
    }.items():
        q_ = a[ys, xs]
        print("   %s ink=%d" % (name, int((q_ < 128).sum())), end="")
    print()

# crude baseline: pick the quadrant with the most ink inside the box area
correct = 0
for img, fam, which, ans, _, q in items:
    a = np.asarray(img.convert("L"))
    H, W = a.shape
    # box word regions (from ocr_suite layout)
    regions = {
        "box 1": (slice(40, 190), slice(20, 150)),
        "box 2": (slice(40, 190), slice(170, 300)),
        "box 3": (slice(230, 380), slice(20, 150)),
        "box 4": (slice(230, 380), slice(170, 300)),
    }
    # the word is drawn at (bx+30, by+60) with 20px bold; measure ink in that strip
    inks = {}
    for k, (ys, xs) in regions.items():
        inks[k] = int((a[ys, xs] < 128).sum())
    pred = max(inks, key=inks.get)
    correct += int(pred == ans)
print("\n[baseline] most-ink box acc = %.3f" % (correct / len(items)))
