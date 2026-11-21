"""Who can actually solve the REAL-DOM grounding task?

The gated head (SigLIP2 + a trained scorer) scored 0.000 on real DOM
candidates, with the no-instruction control ALSO at 0.000. Zero for both
means the head is not the problem — the FEATURES are.

WHY. A diagnostic on rendered text showed that SigLIP2's crop-vs-text
cosine similarity does not identify even a rendered "Action 0" button:
6 candidates, 1 correct. SigLIP2 is an image-TEXT ALIGNMENT model, not an
OCR model. It cannot read the label on a button, and a real DOM's candidate
set is full of buttons whose identity IS their label.

That reframes the task. Real-DOM grounding needs TWO capabilities:
  (1) READ the element's label      -> needs OCR
  (2) MATCH it against the instruction -> needs alignment
SigLIP2 has (2) but not (1). A VLM has both.

This script measures the two families on the SAME real DOM pages, in the
same action space (candidate scoring, one forward), and reports:
  top-1            the accuracy
  no-instruction   the control (does it use the instruction at all?)
  by kind          text-referring vs appearance-referring instructions

The candidates are the page's real clickable elements, read from the DOM.
"""
from __future__ import annotations

import argparse
import io
import json
import time
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

from eval_browser_gated import build_prompts, dom_candidates, page_html

DEVICE = "cuda" if torch.cuda.is_available() else (
    "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")


def vlm_letter_scores(model, proc, image, instruction, crops, letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    """One forward; read the option-letter logits."""
    lab = "\n".join(f"{letters[i]}. candidate {i+1}" for i in range(len(crops)))
    text = (f"The first image is the full screen. The next {len(crops)} images are "
            f"candidate crops, in order, labelled A to {letters[len(crops)-1]}.\n"
            f"{lab}\n"
            f"Instruction: {instruction}\n"
            f"Which candidate crop does the instruction refer to? Answer with one letter.")
    content = [{"type": "image", "image": image}]
    for c in crops:
        content.append({"type": "image", "image": c})
    content.append({"type": "text", "text": text})
    msgs = [{"role": "user", "content": content}]
    prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[prompt], images=[image] + crops, return_tensors="pt")
    inputs = {k: (v.to(DEVICE) if hasattr(v, "to") else v) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1].float()
    tok = proc.tokenizer
    return np.array([float(logits[tok.encode(letters[i], add_special_tokens=False)[-1]])
                      for i in range(len(crops))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="qwen3vl_2b")
    ap.add_argument("--pages", type=int, default=4)
    ap.add_argument("--max-candidates", type=int, default=0)
    ap.add_argument("--out", default="browser_dom_backbones.json")
    args = ap.parse_args()

    from backbone_registry import CANDIDATES

    spec = CANDIDATES[args.backbone]
    print(f"loading {spec['id']}...", flush=True)
    scorer = spec["build"]()
    model, proc = scorer.model, scorer.proc

    results = []
    lat = []
    from playwright.sync_api import sync_playwright

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
            sw, sh = im_full.size
            cands, prompts = build_prompts(page, args.max_candidates or None)
            crops = []
            for c in cands:
                x0 = int(c["x"]); y0 = int(c["y"])
                x1 = int(c["x"] + c["w"]); y1 = int(c["y"] + c["h"])
                crops.append(im_full.crop((x0, y0, x1, y1)).resize((224, 224), Image.BICUBIC))
            for p in prompts:
                t0 = time.time()
                sc = vlm_letter_scores(model, proc, im_full, p["instruction"], crops)
                lat.append((time.time() - t0) * 1000)
                pick = int(np.argmax(sc))
                # control: a constant instruction
                sc0 = vlm_letter_scores(model, proc, im_full, "click something", crops)
                pick0 = int(np.argmax(sc0))
                results.append(dict(page=pi, kind=p["kind"], K=len(cands),
                                    pick=pick, gold=p["idx"],
                                    correct=bool(pick == p["idx"]),
                                    correct_no_instr=bool(pick0 == p["idx"])))
            page.close()
        browser.close()

    n = len(results)
    K = results[0]["K"] if results else 1
    acc = sum(r["correct"] for r in results) / max(n, 1)
    acc0 = sum(r["correct_no_instr"] for r in results) / max(n, 1)
    by = defaultdict(lambda: [0, 0])
    for r in results:
        by[r["kind"]][0] += r["correct"]; by[r["kind"]][1] += 1
    out = dict(backbone=args.backbone, top1=acc, top1_no_instruction=acc0,
               floor=1.0 / K, K=K, n=n,
               by_kind={k: dict(acc=c / max(v, 1), n=v) for k, (c, v) in by.items()},
               latency_ms=float(np.mean(lat)))
    print(f"\n{args.backbone} on real DOM (K={K}):")
    print(f"  top-1 {acc:.4f}  (floor {1/K:.4f})")
    print(f"  no-instruction control {acc0:.4f}")
    print(f"  by kind {out['by_kind']}")
    print(f"  {out['latency_ms']:.0f} ms/decision")
    json.dump(dict(summary=out, results=results), open(args.out, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
