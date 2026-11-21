"""Proper sanity: render each word at a KNOWN location and verify pixels.

Instead of guessing crop offsets, build a clean minimal task: one word per
image at a fixed position. If a linear probe on raw pixels cannot classify
that, the pipeline is broken; if it can, our earlier crops were simply
misaligned.
"""
import sys, os, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
WORDS = ["ALPHA", "BRAVO", "DELTA", "OMEGA", "SIGMA", "THETA"]


def render(word, size=128):
    img = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(FONT_B, 28)
    bbox = d.textbbox((0, 0), word, font=f)
    w = bbox[2] - bbox[0]; h = bbox[3] - bbox[1]
    d.text(((size - w) / 2, (size - h) / 2 - bbox[1]), word, font=f, fill="black")
    return img


def main():
    import numpy as np
    # 1) pixel-level separability of the six words
    imgs = {w: np.asarray(render(w).convert("L")).astype(np.float32) for w in WORDS}
    print("[sanity3] rendered 6 words at 128px, fixed centered position")
    for a in range(len(WORDS)):
        for b in range(a + 1, len(WORDS)):
            diff = np.abs(imgs[WORDS[a]] - imgs[WORDS[b]]).mean()
            if a == 0:
                print("  |%s-%s| = %.2f" % (WORDS[a], WORDS[b], diff))
    # 2) linear probe on raw pixels (no vision tower at all)
    X = torch.tensor(np.stack([imgs[w] for w in WORDS])).float().reshape(6, -1)
    X = (X - X.mean()) / (X.std() + 1e-6)
    y = torch.arange(6)
    lin = nn.Linear(X.shape[1], 6)
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-2)
    for _ in range(500):
        loss = F.cross_entropy(lin(X), y)
        opt.zero_grad(); loss.backward(); opt.step()
    acc = float((lin(X).argmax(-1) == y).float().mean())
    print("[sanity3] raw-pixel linear probe on 6 words: acc=%.3f (6 samples,"
          " just checks the render is separable at all)" % acc)


if __name__ == "__main__":
    main()
