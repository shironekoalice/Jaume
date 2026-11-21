"""OCR on SINGLE crops: the cheapest way to read an element's label.

WHY THIS FILE. The two-stage architecture needs a step that can read the
text on one element at a time. The measurements so far:

  * SigLIP2 cannot read text at all (6 rendered labels -> 1 correct).
  * TrOCR-small-printed reads garbage: "Submit" -> "TABLE", "Action 3" -> ":".
    It is trained on PRINTED documents, and a rendered UI button is not
    printed text. (Its checkpoint also warns that the encoder pooler was
    not initialised.)
  * Qwen3-VL-2B zero-shot with K crops in one prompt: 0.1429 (floor 0.125),
    16530 ms per decision.

But the last measurement conflated two things: OCR capability and the
cost of putting many images in one prompt. This file isolates them. The
VLM is asked to read ONE crop at a time, and the read text is then matched
against the instruction with plain string matching — no scoring head, no
cross-attention, no giant prompt.

If that works, the architecture is:

    stage 1  locate    one forward on the screen -> candidate boxes
    stage 2  confirm   read each of the K crops (K small VLM calls, or one
                       batched call) -> match against the instruction

and the cost is K cheap calls instead of one enormous one.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import time
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

from eval_browser_gated import dom_candidates, page_html

DEVICE = "cuda" if torch.cuda.is_available() else (
    "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")


def norm(s):
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


@torch.no_grad()
def read_one(model, proc, image, prompt="Read the text in this image. Reply with the text only."):
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt}]}]
    p = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[p], images=[image], return_tensors="pt")
    inp = {k: (v.to(DEVICE) if hasattr(v, "to") else v) for k, v in inp.items()}
    out = model.generate(**inp, max_new_tokens=24, do_sample=False)
    txt = proc.batch_decode(out, skip_special_tokens=True)[0]
    # The decode includes the PROMPT (this checkpoint echoes the chat
    # template), so the answer is whatever follows the last "assistant"
    # marker. Without this cut the output starts with "user\nRead the text
    # ..." and an exact-match metric reads 0.0000 even though the model
    # answered correctly.
    if "assistant" in txt:
        txt = txt.rsplit("assistant", 1)[-1]
    return txt.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="qwen3vl_2b")
    ap.add_argument("--pages", type=int, default=2)
    ap.add_argument("--max-candidates", type=int, default=8)
    ap.add_argument("--out", default="single_crop_ocr.json")
    args = ap.parse_args()

    from backbone_registry import CANDIDATES
    spec = CANDIDATES[args.backbone]
    print(f"loading {spec['id']}...", flush=True)
    scorer = spec["build"]()
    model, proc = scorer.model, scorer.proc

    from playwright.sync_api import sync_playwright

    n_read = 0
    ocr_correct = 0
    task_correct = 0
    task_n = 0
    lat = []
    dump = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            args=["--headless=new", "--no-sandbox", "--disable-gpu"])
        for pi in range(args.pages):
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.set_content(page_html([("blue", "Submit"), ("green", "Export"),
                                        ("red", "Delete")], n_filler=8))
            page.wait_for_timeout(150)
            im_full = Image.open(io.BytesIO(page.screenshot())).convert("RGB")
            cands = dom_candidates(page)[:args.max_candidates] if args.max_candidates \
                else dom_candidates(page)
            crops, dom_texts = [], []
            for c in cands:
                x0, y0 = int(c["x"]), int(c["y"])
                x1, y1 = int(c["x"] + c["w"]), int(c["y"] + c["h"])
                crops.append(im_full.crop((x0, y0, x1, y1)).resize((224, 224), Image.BICUBIC))
                dom_texts.append(c["text"])

            t0 = time.time()
            ocr_texts = []
            for cr in crops:
                ocr_texts.append(read_one(model, proc, cr))
            lat.append((time.time() - t0) * 1000)
            n_read += len(crops)

            for i, (dt, ot) in enumerate(zip(dom_texts, ocr_texts)):
                if not dt:
                    continue
                if norm(ot) == norm(dt):
                    ocr_correct += 1
            dump.append(dict(dom=dom_texts, ocr=ocr_texts))

            # task level: instruction -> which candidate
            for i, dt in enumerate(dom_texts):
                if not dt:
                    continue
                instr = f"click {dt}"
                scores = []
                for ot in ocr_texts:
                    s = 1.0 if norm(ot) == norm(dt) else 0.0
                    if norm(ot) and (norm(ot) in norm(dt) or norm(dt) in norm(ot)):
                        s = max(s, 0.6)
                    scores.append(s)
                pick = int(np.argmax(scores))
                task_correct += int(pick == i)
                task_n += 1
            page.close()
        browser.close()

    res = dict(backbone=args.backbone,
               ocr_exact=ocr_correct / max(n_read, 1),
               task_top1=task_correct / max(task_n, 1),
               floor=1.0 / max(args.max_candidates, 1),
               n_crops_read=n_read, n_tasks=task_n,
               ms_per_page=lat, ms_per_crop=float(np.mean(lat)) / max(n_read, 1) if lat else 0)
    print(f"\n{args.backbone} reading ONE crop at a time:")
    print(f"  OCR exact match (crop text == DOM text): {res['ocr_exact']:.4f}")
    print(f"  task top-1 (instruction -> candidate):   {res['task_top1']:.4f} "
          f"(floor {res['floor']:.4f})")
    print(f"  {res['ms_per_crop']:.0f} ms per crop, {len(lat)} pages")
    json.dump(dict(summary=res, dump=dump), open(args.out, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
