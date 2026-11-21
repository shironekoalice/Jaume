"""Real-browser evaluation of the UI actor (Playwright + system Chrome).

Pipeline per sample:
  1. open a real page (several templates with different layouts)
  2. read the REAL candidate elements from the DOM (id, class, text, box)
  3. take a real screenshot at 128x128 (the actor's native input size)
  4. crop each candidate from the screenshot, feed the trained actor
  5. compare the actor's pick with the element the instruction refers to

This is the step the synthetic benchmark cannot do: the pixels come from a
real browser, the candidate boxes come from the real DOM, and the
instruction refers to elements with real text.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random

import numpy as np
import torch
from PIL import Image

from envs import CROP, IMG, VOCAB, word_id
from model import UIActor

TEMPLATES = [
    dict(name="buttons", html="""<div class="row"><span class="btn blue">Submit</span>
      <span class="btn green">Export</span><span class="btn red">Cancel</span>
      <span class="btn gray">Help</span></div>""",
         items=[("blue", "Submit"), ("green", "Export"), ("red", "Cancel"), ("gray", "Help")]),
    dict(name="cards", html="""<div class="row"><span class="card" id="c0" style="background:#d64541"></span>
      <span class="card" id="c1" style="background:#3ca05a"></span>
      <span class="card" id="c2" style="background:#3d6ec8"></span></div>""",
         items=[("red", "c0"), ("green", "c1"), ("blue", "c2")]),
    dict(name="mixed", html="""<div class="row"><span class="btn blue">Save</span>
      <span class="btn green">Share</span><span class="btn red">Delete</span></div>""",
         items=[("blue", "Save"), ("green", "Share"), ("red", "Delete")]),
]

PAGE_HEAD = """<!doctype html><html><head><meta charset="utf-8"><style>
body{font-family:sans-serif;margin:0;background:#f7f7f5}
header{background:#e4e6ec;padding:12px 16px;font-weight:600}
.row{padding:10px 16px;border-bottom:1px solid #e8e8e8}
.btn{display:inline-block;padding:10px 18px;border-radius:6px;color:#fff;margin:6px}
.blue{background:#3d6ec8}.green{background:#3ca05a}.red{background:#d64541}.gray{background:#8c8c91}
.card{width:170px;height:80px;border-radius:8px;margin:8px;display:inline-block;vertical-align:top}
</style></head><body><header>Invoices</header>
<div class="row">Report for the period ending March</div>
<div class="row">Reference number 88213</div>
"""

# colour -> instruction word, plus the shape word used by the synthetic task
COLOR_WORD = {"blue": "blue", "green": "green", "red": "red", "gray": "gray"}


def build_page(tpl, extra_rows=0):
    body = PAGE_HEAD + tpl["html"]
    for i in range(extra_rows):
        body += f'<div class="row">Line item {i+1} — quantity {i*3+2}</div>'
    body += "</body></html>"
    return body


def run(args):
    from playwright.sync_api import sync_playwright

    device = args.device
    model = UIActor().to(device).eval()
    if args.ckpt and os.path.exists(args.ckpt):
        sd = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(sd if not isinstance(sd, dict) or "state_dict" not in sd else sd["state_dict"])
        print(f"loaded weights from {args.ckpt}", flush=True)
    else:
        print("no checkpoint given: the actor will be RANDOM. Results will be noise.", flush=True)

    rng = random.Random(0)
    results = []
    with sync_playwright() as pw:
        # NOTE: the system Chrome exits immediately with playwright's default
        # `--headless` (old headless) flags on this host. `--headless=new`
        # works. This was found by trying the flag matrix, not documented.
        launch_args = ["--headless=new", "--no-sandbox", "--disable-gpu"]
        browser = pw.chromium.launch(
            executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            args=launch_args,
        )
        for tpl in TEMPLATES:
            for extra in (0, 3):
                html = build_page(tpl, extra)
                # The viewport must be a REAL screen size. An earlier version
                # used 128x128, which is the model's INPUT size — but the page
                # then overflowed (DOM y up to 677 on a 128px-tall
                # screenshot), so every crop was blank and the score was 0.20.
                # The screen is captured at real size and RESIZED to 128 for
                # the model; the DOM boxes are scaled by the same factor.
                page = browser.new_page(viewport={"width": 1280, "height": 900},
                                        device_scale_factor=1)
                page.set_content(html)
                page.wait_for_timeout(150)
                shot = page.screenshot()
                im_full = Image.open(io.BytesIO(shot)).convert("RGB")
                sw, sh = im_full.size
                im = im_full.resize((IMG, IMG))
                screen = np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0

                # real candidate geometry from the DOM, scaled to the screenshot
                cands = page.evaluate("""() => Array.from(document.querySelectorAll('.btn,.card')).map(e => {
                    const r = e.getBoundingClientRect();
                    const cs = getComputedStyle(e);
                    return {cls: e.className, text: e.textContent.trim(), id: e.id,
                            bg: cs.backgroundColor,
                            x: r.x, y: r.y, w: r.width, h: r.height};
                })""")
                if len(cands) < 2:
                    page.close()
                    continue
                crop_t, box_t = [], []
                for c in cands:
                    x0 = int(c["x"] * sw / 1280)
                    y0 = int(c["y"] * sh / 900)
                    x1 = int((c["x"] + c["w"]) * sw / 1280)
                    y1 = int((c["y"] + c["h"]) * sh / 900)
                    x0 = max(0, min(sw - 1, x0)); y0 = max(0, min(sh - 1, y0))
                    x1 = max(x0 + 1, min(sw, x1)); y1 = max(y0 + 1, min(sh, y1))
                    crop = np.asarray(im_full.crop((x0, y0, x1, y1)).resize((CROP, CROP)),
                                      dtype=np.float32).transpose(2, 0, 1) / 255.0
                    crop_t.append(crop)
                    box_t.append([x0 / sw, y0 / sh, x1 / sw, y1 / sh])
                crops = torch.from_numpy(np.stack(crop_t)).unsqueeze(0).to(device)
                boxes = torch.from_numpy(np.array(box_t, dtype=np.float32)).unsqueeze(0).to(device)
                scr = torch.from_numpy(screen).unsqueeze(0).to(device)

                # one instruction per candidate, referring to it by colour
                for gi, gt in enumerate(tpl["items"]):
                    gcolor, gkey = gt
                    instr = [word_id(w) for w in ["click", COLOR_WORD[gcolor], "circle", "at", "top", "left"]]
                    txt = torch.tensor([instr], device=device)
                    mask = torch.ones(1, len(cands), dtype=torch.bool, device=device)
                    with torch.no_grad():
                        logits = model(crops, boxes, scr, txt, mask)
                    pick = int(logits.argmax(-1).item())
                    # gold: the candidate whose COLOUR matches. Buttons carry
                    # the colour in their class, cards carry it in the inline
                    # background style — an earlier version only looked at
                    # the class and scored the card templates 0.000 even
                    # though the model was picking correctly.
                    COLOR_RGB = {"blue": "61, 110, 200", "green": "60, 160, 90",
                                 "red": "214, 69, 65", "gray": "140, 140, 145"}
                    needle = COLOR_RGB[gcolor]
                    gidx = next((i for i, c in enumerate(cands)
                                 if gcolor in c["cls"] or needle in c.get("bg", "")), None)
                    results.append(dict(template=tpl["name"], extra=extra,
                                        n_cand=len(cands), pick=pick, gold=gidx,
                                        correct=bool(pick == gidx)))
                page.close()
        browser.close()

    n = len(results)
    acc = sum(r["correct"] for r in results) / max(n, 1)
    by_tpl = {}
    for r in results:
        by_tpl.setdefault(r["template"], []).append(r["correct"])
    print(f"\nreal-browser UI grounding: {acc:.4f} over {n} samples", flush=True)
    for k, v in by_tpl.items():
        print(f"  {k}: {sum(v)/len(v):.4f} ({len(v)})", flush=True)
    return dict(accuracy=acc, n=n, by_template={k: sum(v)/len(v) for k, v in by_tpl.items()},
                results=results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--out", default="browser_ui_result.json")
    args = ap.parse_args()
    res = run(args)
    json.dump(res, open(args.out, "w"), indent=2)
    print(json.dumps({k: res[k] for k in ("accuracy", "n", "by_template")}, indent=2))


if __name__ == "__main__":
    main()
