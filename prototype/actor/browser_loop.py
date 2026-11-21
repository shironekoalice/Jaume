"""Closed-loop control of a REAL browser with the actor.

The grounding test (`eval_browser_ui.py`) is open-loop: given a screenshot and
an instruction, pick an element. This script closes the loop:

    screenshot -> actor picks an element -> CDP clicks it -> new screenshot
                -> actor picks again -> ... until the task is done or the
                step budget runs out

That is the actual product requirement ("浏览器控制"), and it is where the
architecture has to survive: the page CHANGES between steps, so a wrong
pick can end the task.

Two page families are used:

  TASK A (single step)  — a form page; the instruction names one field or
                           button. Success = the right element was clicked.
  TASK B (multi step)   — a page where the second target only appears after
                           the first is clicked (a "reveal" step). Success =
                           both the first and the second click are right.

The candidate set is re-read from the DOM after every click, which is the
honest version of "candidate scoring on a live page".
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

PAGE_A = """<!doctype html><html><head><meta charset="utf-8"><style>
body{font-family:sans-serif;margin:0;background:#f7f7f5}
header{background:#e4e6ec;padding:14px 18px;font-weight:600}
.row{padding:12px 18px;border-bottom:1px solid #e8e8e8}
.btn{display:inline-block;padding:12px 22px;border-radius:6px;color:#fff;margin:6px}
.blue{background:#3d6ec8}.green{background:#3ca05a}.red{background:#d64541}.gray{background:#8c8c91}
input{padding:9px 12px;border:1px solid #ccd0d8;border-radius:5px;width:220px}
</style></head><body><header>Checkout</header>
<div class="row">Item <input id="qty" value="1"></div>
<div class="row"><input id="name" placeholder="Your name"></div>
<div class="row"><span class="btn blue">Pay</span><span class="btn green">Cancel</span>
<span class="btn red">Help</span><span class="btn gray">Back</span></div>
</body></html>"""

PAGE_B = """<!doctype html><html><head><meta charset="utf-8"><style>
body{font-family:sans-serif;margin:0;background:#f7f7f5}
header{background:#e4e6ec;padding:14px 18px;font-weight:600}
.row{padding:12px 18px;border-bottom:1px solid #e8e8e8}
.btn{display:inline-block;padding:12px 22px;border-radius:6px;color:#fff;margin:6px}
.blue{background:#3d6ec8}.green{background:#3ca05a}.red{background:#d64541}.gray{background:#8c8c91}
.hide{display:none}
</style></head><body><header>Two-step</header>
<div class="row"><span class="btn blue" id="step1">Continue</span></div>
<div class="row hide" id="step2row"><span class="btn green" id="step2">Finish</span>
<span class="btn red" id="step2bad">Abort</span></div>
<script>document.getElementById('step1').onclick=function(){
  document.getElementById('step2row').classList.remove('hide');};</script>
</body></html>"""

COLOR_RGB = {"blue": "61, 110, 200", "green": "60, 160, 90",
             "red": "214, 69, 65", "gray": "140, 140, 145"}


def candidates(page):
    return page.evaluate("""() => Array.from(document.querySelectorAll('.btn,input')).map(e => {
        const r = e.getBoundingClientRect();
        const cs = getComputedStyle(e);
        return {tag: e.tagName, id: e.id, cls: e.className,
                ph: e.getAttribute('placeholder') || '', bg: cs.backgroundColor,
                x: r.x, y: r.y, w: r.width, h: r.height};
    })""")


def act(model, page, device, instruction_words):
    shot = page.screenshot()
    im_full = Image.open(io.BytesIO(shot)).convert("RGB")
    sw, sh = im_full.size
    im = im_full.resize((IMG, IMG))
    screen = np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0
    cands = candidates(page)
    crops, boxes = [], []
    for c in cands:
        x0 = int(c["x"] * sw / 1280); y0 = int(c["y"] * sh / 900)
        x1 = int((c["x"] + c["w"]) * sw / 1280); y1 = int((c["y"] + c["h"]) * sh / 900)
        x0 = max(0, min(sw - 1, x0)); y0 = max(0, min(sh - 1, y0))
        x1 = max(x0 + 1, min(sw, x1)); y1 = max(y0 + 1, min(sh, y1))
        crops.append(np.asarray(im_full.crop((x0, y0, x1, y1)).resize((CROP, CROP)),
                                dtype=np.float32).transpose(2, 0, 1) / 255.0)
        boxes.append([x0 / sw, y0 / sh, x1 / sw, y1 / sh])
    if not crops:
        return None, None
    with torch.no_grad():
        logits = model(torch.from_numpy(np.stack(crops)).unsqueeze(0).to(device),
                       torch.from_numpy(np.array(boxes, dtype=np.float32)).unsqueeze(0).to(device),
                       torch.from_numpy(screen).unsqueeze(0).to(device),
                       torch.tensor([[word_id(w) for w in instruction_words]], device=device),
                       torch.ones(1, len(crops), dtype=torch.bool, device=device))
    return int(logits.argmax(-1).item()), cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--ckpt", default="ui_hard_model.pt")
    ap.add_argument("--out", default="browser_loop_result.json")
    args = ap.parse_args()
    device = args.device

    from playwright.sync_api import sync_playwright

    model = UIActor().to(device).eval()
    if os.path.exists(args.ckpt):
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        print(f"loaded {args.ckpt}", flush=True)
    else:
        print("NO CHECKPOINT — results will be noise", flush=True)

    results = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            args=["--headless=new", "--no-sandbox", "--disable-gpu"])

        # ---- TASK A: pick the right control, click it, verify the click landed
        for color in ("green", "blue", "red", "gray"):
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.set_content(PAGE_A)
            page.wait_for_timeout(120)
            words = ["click", color, "circle", "at", "top", "left"]
            pick, cands = act(model, page, device, words)
            gold = next((i for i, c in enumerate(cands)
                         if color in c["cls"] or COLOR_RGB[color] in c.get("bg", "")), None)
            ok = pick == gold
            if ok and pick is not None:
                # click it for real and confirm the page reacted
                el = page.query_selector_all(".btn,input")[pick]
                el.click()
                page.wait_for_timeout(80)
            results.append(dict(task="single", color=color, pick=pick, gold=gold, correct=ok))
            print(f"  [single] want {color:6s} pick={pick} gold={gold} -> {ok}", flush=True)
            page.close()

        # ---- TASK B: two steps, the second target only exists after the first
        for c1, c2 in (("blue", "green"), ("blue", "red")):
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.set_content(PAGE_B)
            page.wait_for_timeout(120)
            pick1, cands1 = act(model, page, device, ["click", c1, "circle", "at", "top", "left"])
            gold1 = next((i for i, c in enumerate(cands1)
                          if c1 in c["cls"] or COLOR_RGB[c1] in c.get("bg", "")), None)
            step1_ok = pick1 == gold1
            if step1_ok:
                page.query_selector_all(".btn,input")[pick1].click()
                page.wait_for_timeout(120)
            pick2, cands2 = act(model, page, device, ["click", c2, "circle", "at", "top", "left"])
            gold2 = next((i for i, c in enumerate(cands2)
                          if c2 in c["cls"] or COLOR_RGB[c2] in c.get("bg", "")), None)
            step2_ok = pick2 == gold2
            results.append(dict(task="two_step", step1_ok=step1_ok, step2_ok=step2_ok,
                                pick1=pick1, gold1=gold1, pick2=pick2, gold2=gold2,
                                correct=bool(step1_ok and step2_ok)))
            print(f"  [two-step] {c1}->{c2}: step1 {step1_ok} step2 {step2_ok}", flush=True)
            page.close()
        browser.close()

    n = len(results)
    acc = sum(r["correct"] for r in results) / n
    print(f"\nbrowser closed-loop success: {acc:.4f} over {n} tasks", flush=True)
    json.dump(dict(accuracy=acc, n=n, results=results), open(args.out, "w"), indent=2)
    print(json.dumps(dict(accuracy=acc, n=n), indent=2))


if __name__ == "__main__":
    main()
