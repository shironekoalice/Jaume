"""Diagnose the real-browser gap: is it the crop, the candidate set, or the model?

The synthetic benchmark scores 1.000 while the real browser scores 0.200.
Three candidate explanations, each with a different fix:

  H1 CROPPING IS WRONG — the DOM box and the screenshot pixel coordinates do
     not agree (device scale factor, scroll offset, viewport resize), so the
     "crop" is not the element. Fix: align them, then re-measure.

  H2 THE INSTRUCTION IS ILL-FORMED — the synthetic instruction says
     "click the green circle", but a real page has no circles; it has
     rectangular buttons. The instruction names a property the target does
     not have, so the task is genuinely ambiguous and 0.200 is chance.

  H3 THE TASK IS OUT OF DISTRIBUTION — the model saw synthetic shapes
     (circle/square/triangle) and never saw rendered text or anti-aliased
     edges. Fix: train on real screenshots.

This script separates them by dumping, for one page: the DOM boxes, the
corresponding crops, the model's scores, and the crops of the gold element.
The screenshots are written to disk so they can be looked at.
"""
from __future__ import annotations

import argparse
import io
import json
import os

import numpy as np
import torch
from PIL import Image

from envs import CROP, IMG, VOCAB, word_id
from model import UIActor

HTML = """<!doctype html><html><head><meta charset="utf-8"><style>
body{font-family:sans-serif;margin:0;background:#f7f7f5}
header{background:#e4e6ec;padding:12px 16px;font-weight:600}
.row{padding:10px 16px;border-bottom:1px solid #e8e8e8}
.btn{display:inline-block;padding:10px 18px;border-radius:6px;color:#fff;margin:6px}
.blue{background:#3d6ec8}.green{background:#3ca05a}.red{background:#d64541}.gray{background:#8c8c91}
.card{width:170px;height:80px;border-radius:8px;margin:8px;display:inline-block;vertical-align:top}
</style></head><body><header>Invoices</header>
<div class="row">Report for the period ending March</div>
<div class="row">Reference number 88213</div>
<div class="row"><span class="btn blue">Submit</span><span class="btn green">Export</span>
<span class="btn red">Cancel</span><span class="btn gray">Help</span></div>
<div class="row"><span class="card" style="background:#ffd9d6"></span>
<span class="card" style="background:#d7f0dd"></span><span class="card" style="background:#d9e4fb"></span></div>
</body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--ckpt", default="ui_hard_model.pt")
    ap.add_argument("--outdir", default="_diag")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    device = args.device

    from playwright.sync_api import sync_playwright

    model = UIActor().to(device).eval()
    sd = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(sd)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            args=["--headless=new", "--no-sandbox", "--disable-gpu"],
        )
        for vw, vh in [(128, 128), (640, 480)]:
            page = browser.new_page(viewport={"width": vw, "height": vh})
            page.set_content(HTML)
            page.wait_for_timeout(200)
            shot = page.screenshot()
            im = Image.open(io.BytesIO(shot)).convert("RGB")
            print(f"\n=== viewport {vw}x{vh} -> screenshot {im.size} ===")
            im.save(os.path.join(args.outdir, f"page_{vw}x{vh}.png"))

            cands = page.evaluate("""() => Array.from(document.querySelectorAll('.btn,.card')).map(e => {
                const r = e.getBoundingClientRect();
                return {cls: e.className, text: e.textContent.trim(),
                        x: r.x, y: r.y, w: r.width, h: r.height};
            })""")
            for i, c in enumerate(cands):
                print(f"  cand{i}: cls={c['cls'][:28]:28s} text={c['text'][:8]:8s} "
                      f"box=({c['x']:.0f},{c['y']:.0f},{c['w']:.0f}x{c['h']:.0f})")

            # crop from the screenshot using the DOM box, in screenshot pixels
            sw, sh = im.size
            sx, sy = sw / vw, sh / vh
            crops, boxes = [], []
            for i, c in enumerate(cands):
                x0 = int(c["x"] * sx); y0 = int(c["y"] * sy)
                x1 = int((c["x"] + c["w"]) * sx); y1 = int((c["y"] + c["h"]) * sy)
                crop = im.crop((x0, y0, max(x0 + 1, x1), max(y0 + 1, y1))).resize((CROP, CROP))
                crop.save(os.path.join(args.outdir, f"crop_{vw}x{vh}_{i}.png"))
                crops.append(np.asarray(crop, dtype=np.float32).transpose(2, 0, 1) / 255.0)
                boxes.append([x0 / sw, y0 / sh, x1 / sw, y1 / sh])
            print(f"  saved {len(crops)} crops to {args.outdir}")

            # model scores for one instruction
            screen = np.asarray(im.resize((IMG, IMG)), dtype=np.float32).transpose(2, 0, 1) / 255.0
            for color in ("green", "blue", "red"):
                instr = [word_id(w) for w in ["click", color, "circle", "at", "top", "left"]]
                crops_t = torch.from_numpy(np.stack(crops)).unsqueeze(0).to(device)
                boxes_t = torch.from_numpy(np.array(boxes, dtype=np.float32)).unsqueeze(0).to(device)
                scr_t = torch.from_numpy(screen).unsqueeze(0).to(device)
                txt = torch.tensor([instr], device=device)
                mask = torch.ones(1, len(crops), dtype=torch.bool, device=device)
                with torch.no_grad():
                    lg = model(crops_t, boxes_t, scr_t, txt, mask)[0]
                order = torch.argsort(lg, descending=True).tolist()
                print(f"  instr '{color} circle': ranked {order[:3]}  "
                      f"scores {[round(float(lg[i]),2) for i in order[:3]]}")
            page.close()
        browser.close()


if __name__ == "__main__":
    main()
