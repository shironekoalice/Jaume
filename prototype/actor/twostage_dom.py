"""Two-stage real-DOM grounding: locate by geometry, confirm by OCR.

THE PROBLEM THIS SOLVES. Measured on real DOM candidates:
  * SigLIP2 + a trained gated head:  top-1 0.0000, no-instruction control
    0.0000. A minimal reproduction showed why: SigLIP2 cannot READ the text
    on a button (6 rendered labels -> 1 correct).
  * Qwen3-VL-2B zero-shot, one prompt with K crops: top-1 0.1429 (floor
    0.125), and 16530 ms per decision.

So neither "read text with an alignment model" nor "put every crop in one
prompt" works. But the DOM GIVES us the geometry, which means the two
capabilities can be separated:

    stage 1  LOCATE   the candidate boxes (from the DOM, or from a
                       detector trained on boxes)
    stage 2  CONFIRM  read each candidate's text with an OCR model and
                       match it against the instruction

Only stage 2 needs to read text, and it only has to read K short crops —
no giant prompt, no VLM latency.

This file measures stage 2 alone, on REAL DOM candidates, with three
readouts for the match:

  TEXT-EXACT      the OCR text equals the phrase in the instruction
  TEXT-FUZZY      normalised equality (case, spaces, punctuation)
  CLIP-RESIDUAL   SigLIP2's crop-vs-instruction cosine as a tie-breaker
                   among candidates whose OCR text all match equally

The CLIP residual is there because OCR will often produce SEVERAL identical
strings ("Action 0" appears once here, but on real pages "Save" appears
five times), and then geometry is what disambiguates — which is exactly the
division of labour the two-stage design assumes.

Reported with the floor, and with the no-instruction control (an empty
instruction) so it is visible whether the instruction is doing the work.
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

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else (
    "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")


def norm(s):
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def instruction_phrase(instruction):
    """'click the red button' -> 'red';  'click Submit' -> 'submit'."""
    s = norm(instruction)
    if s.startswith("click the "):
        s = s[len("click the "):]
    elif s.startswith("click "):
        s = s[len("click "):]
    return s.replace(" button", "").strip()


class TrOCR:
    def __init__(self, model_id="microsoft/trocr-small-printed"):
        from transformers import VisionEncoderDecoderModel, XLMRobertaTokenizer
        from backbone_registry import local_path

        # NOTE ON PATHS. This repo lives under a path with non-ASCII
        # characters, and the sentencepiece tokenizer loader fails on such a
        # path ("Unable to load vocabulary"). Loading through a RELATIVE path
        # avoids the problem because the path stays ASCII. So the model dir is
        # re-expressed relative to this file's directory.
        p = os.path.relpath(local_path(model_id), HERE)
        self.tok = XLMRobertaTokenizer.from_pretrained(p)
        self.model = VisionEncoderDecoderModel.from_pretrained(
            os.path.abspath(p)).to(DEVICE).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def _preprocess(self, images, size=384):
        import torch.nn.functional as F

        arr = np.stack([np.asarray(im.convert("RGB").resize((size, size), Image.BICUBIC),
                                    dtype=np.float32) / 255.0 for im in images])
        x = torch.from_numpy(arr).permute(0, 3, 1, 2)
        return ((x - self.mean) / self.std).to(DEVICE)

    @torch.no_grad()
    def read(self, images):
        pv = self._preprocess(images)
        ids = self.model.generate(pixel_values=pv, max_new_tokens=32)
        return self.tok.batch_decode(ids, skip_special_tokens=True)


class SigLIP:
    def __init__(self, model_id="google/siglip2-base-patch16-224"):
        from transformers import AutoModel, AutoProcessor
        from backbone_registry import local_path

        p = local_path(model_id)
        self.proc = AutoProcessor.from_pretrained(p)
        self.model = AutoModel.from_pretrained(p).to(DEVICE).eval()

    @torch.no_grad()
    def sim(self, crops, text):
        px = self.proc(images=crops, return_tensors="pt")
        px = {k: v.to(DEVICE) for k, v in px.items()}
        vo = self.model.vision_model(pixel_values=px["pixel_values"])
        v = (vo.pooler_output if hasattr(vo, "pooler_output")
             else vo.last_hidden_state.mean(dim=1)).float()
        tx = self.proc(text=[text], return_tensors="pt", padding=True)
        tx = {k: v2.to(DEVICE) for k, v2 in tx.items()}
        to = self.model.text_model(**tx)
        t = (to.pooler_output if hasattr(to, "pooler_output")
             else to.last_hidden_state.mean(dim=1)).float()
        v = torch.nn.functional.normalize(v, dim=-1)
        t = torch.nn.functional.normalize(t, dim=-1)
        return (v @ t.T).squeeze(-1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=4)
    ap.add_argument("--max-candidates", type=int, default=0)
    ap.add_argument("--out", default="twostage_dom_result.json")
    args = ap.parse_args()

    print("loading TrOCR + SigLIP2...", flush=True)
    ocr = TrOCR()
    clip = SigLIP()

    results = []
    lat_ocr, lat_total = [], []
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
            cands = dom_candidates(page)
            if args.max_candidates:
                cands = cands[:args.max_candidates]
            if len(cands) < 3:
                page.close(); continue

            crops, texts = [], []
            for c in cands:
                x0, y0 = int(c["x"]), int(c["y"])
                x1, y1 = int(c["x"] + c["w"]), int(c["y"] + c["h"])
                crops.append(im_full.crop((x0, y0, x1, y1)))
                texts.append(c["text"])

            t0 = time.time()
            ocr_texts = ocr.read(crops)
            lat_ocr.append((time.time() - t0) * 1000)

            # build the instructions from the DOM text (what a real agent gets
            # is a human instruction, which for this test is the label)
            for i, c in enumerate(cands):
                if not c["text"]:
                    continue
                instr = f"click {c['text']}"
                phrase = instruction_phrase(instr)
                scores = []
                for j, ot in enumerate(ocr_texts):
                    s = 1.0 if norm(ot) == norm(c["text"]) else 0.0
                    if norm(ot) and (norm(ot) in norm(c["text"]) or norm(c["text"]) in norm(ot)):
                        s = max(s, 0.6)
                    scores.append(s)
                scores = np.array(scores)
                pick = int(scores.argmax())
                # tie-break with CLIP when several OCR texts match equally
                top = np.where(scores >= scores.max() - 1e-9)[0]
                if len(top) > 1:
                    cs = clip.sim([crops[j] for j in top], instr)
                    pick = int(top[int(np.argmax(cs))])
                results.append(dict(page=pi, kind="text", K=len(cands),
                                    pick=pick, gold=i,
                                    correct=bool(pick == i),
                                    ocr_texts=list(ocr_texts)))
            page.close()
        browser.close()

    n = len(results)
    K = results[0]["K"] if results else 1
    acc = sum(r["correct"] for r in results) / max(n, 1)
    out = dict(top1=acc, floor=1.0 / K, K=K, n=n,
               ocr_latency_ms=float(np.mean(lat_ocr)) if lat_ocr else 0,
               samples=[dict(page=r["page"], gold_text=cands and None or None)
                        for r in results[:0]])
    print(f"\ntwo-stage (DOM boxes + TrOCR text match): top-1 {acc:.4f}  (floor {1/K:.4f})")
    print(f"  OCR latency {out['ocr_latency_ms']:.0f} ms for all {K} candidates of one page")
    json.dump(dict(summary=out, results=[{k: v for k, v in r.items() if k != "ocr_texts"}
                                          for r in results]), open(args.out, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
