"""Evaluate the GATED HEAD on REAL DOM candidates in a REAL browser.

This is the P0 of the handoff document: the gated head scored 0.6483 on
seeclick's real-element distractors, but that is still a CONSTRUCTED
candidate set. The regime the actor will actually run in is:

  * a real page,
  * the candidate set is the page's ACTUAL clickable elements, read from
    the DOM — dozens of them, many of them the same widget,
  * the instruction refers to one of them by its visible text.

That is a harder task than anything measured so far, and it is the one that
decides whether the actor is deployable.

WHAT IS MEASURED
  top-1        the fraction of instructions where the head picks the right
               DOM element
  by_text      instructions that name the element's TEXT ("click Submit")
  by_visual    instructions that name only an APPEARANCE ("the red button")
               — these are the hard ones, and they are what the gate is for
  latency      ms per decision, with the frozen tower in the loop

The pages are generated locally and rendered by real Chrome, so the pixels
and the DOM geometry are real. The candidate list comes from the DOM, not
from an annotation file.

CONTROLS REPORTED ALONGSIDE
  no-instruction   the same head with the instruction replaced by a constant.
                   If this scores well, the head is not using the
                   instruction and the benchmark is measuring nothing.
  random           the floor, 1/K
"""
from __future__ import annotations

import argparse
import io
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else (
    "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")

CSS = """
body{font-family:sans-serif;margin:0;background:#f7f7f5;font-size:13px}
header{background:#e4e6ec;padding:12px 16px;font-weight:600}
.row{padding:9px 16px;border-bottom:1px solid #ececec}
.btn{display:inline-block;padding:9px 16px;border-radius:5px;color:#fff;margin:4px}
.blue{background:#3d6ec8}.green{background:#3ca05a}.red{background:#d64541}
.gray{background:#8c8c91}.orange{background:#d98a2b}
a{color:#2255aa;text-decoration:underline}
input{padding:6px 9px;border:1px solid #ccd0d8;border-radius:4px;width:180px}
.card{display:inline-block;width:120px;height:56px;border-radius:6px;margin:5px;
      vertical-align:top}
"""


def page_html(items, n_filler=8):
    """A realistic-looking page: a header, some rows, and MANY buttons of
    which only one is the target. The filler buttons are what make this
    hard — they are the same widget type as the target."""
    body = [f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>"]
    body.append("<header>Customer records</header>")
    body.append("<div class='row'>Report for the period ending March</div>")
    body.append("<div class='row'>Reference number 88213</div>")
    body.append("<div class='row'><input id='search' placeholder='Search'></div>")
    body.append("<div class='row'>")
    for i in range(n_filler):
        cls = ["blue", "green", "red", "gray", "orange"][i % 5]
        body.append(f"<span class='btn {cls}' id='f{i}'>Action {i}</span>")
    body.append("</div>")
    body.append("<div class='row'>")
    for i, (cls, label) in enumerate(items):
        body.append(f"<span class='btn {cls}' id='t{i}'>{label}</span>")
    body.append("</div>")
    body.append("<div class='row'>")
    for i in range(6):
        body.append(f"<span class='card' id='c{i}' style='background:#dde4f0'></span>")
    body.append("</div>")
    body.append("</body></html>")
    return "".join(body)


def dom_candidates(page):
    return page.evaluate("""() => Array.from(document.querySelectorAll('.btn,.card,input')).map(e => {
        const r = e.getBoundingClientRect();
        const cs = getComputedStyle(e);
        return {id: e.id, tag: e.tagName, cls: e.className,
                text: (e.textContent||'').trim(), ph: e.getAttribute('placeholder')||'',
                bg: cs.backgroundColor, x: r.x, y: r.y, w: r.width, h: r.height};
    })""")


def build_prompts(page, max_candidates=None):
    """One instruction per candidate, in two flavours.

    text    "click Submit"  — names the visible text
    visual  "click the red button" — names only the colour, so the head must
            use the IMAGE and ignore the text

    `max_candidates` truncates the candidate list (and the prompts are built
    from the truncated list), which is how the slow VLM baseline is kept
    inside a sane time budget.
    """
    cands = dom_candidates(page)
    if max_candidates:
        cands = cands[:max_candidates]
    prompts = []
    for i, c in enumerate(cands):
        if not c["text"] and c["tag"] != "INPUT":
            continue
        colour = None
        for name, rgb in [("blue", "61, 110, 200"), ("green", "60, 160, 90"),
                          ("red", "214, 69, 65"), ("gray", "140, 140, 145"),
                          ("orange", "217, 138, 43")]:
            if rgb in c["bg"]:
                colour = name
                break
        if c["text"]:
            prompts.append(dict(idx=i, kind="text",
                                 instruction=f"click {c['text']}"))
        if colour:
            prompts.append(dict(idx=i, kind="visual",
                                 instruction=f"click the {colour} button"))
    return cands, prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="gated_head.pt")
    ap.add_argument("--device", default=DEVICE)
    ap.add_argument("--out", default="browser_gated_result.json")
    ap.add_argument("--pages", type=int, default=6)
    args = ap.parse_args()

    from transformers import AutoModel, AutoProcessor
    from backbone_registry import local_path
    from real_distractor import GatedScorer, crop_image

    mid = "google/siglip2-base-patch16-224"
    proc = AutoProcessor.from_pretrained(local_path(mid))
    tower = AutoModel.from_pretrained(local_path(mid)).to(args.device).eval()

    ck = torch.load(args.ckpt, map_location=args.device)
    head = GatedScorer(ck["dv"], ck["dt"]).to(args.device)
    head.load_state_dict(ck["head"])
    head.eval()
    print(f"loaded head from {args.ckpt}", flush=True)

    from playwright.sync_api import sync_playwright

    results = []
    lat = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            args=["--headless=new", "--no-sandbox", "--disable-gpu"])
        for pi in range(args.pages):
            items = [("blue", "Submit"), ("green", "Export"), ("red", "Delete")]
            page = browser.new_page(viewport={"width": 1280, "height": 900},
                                    device_scale_factor=1)
            page.set_content(page_html(items, n_filler=8))
            page.wait_for_timeout(150)
            shot = page.screenshot()
            im_full = Image.open(io.BytesIO(shot)).convert("RGB")
            sw, sh = im_full.size
            cands, prompts = build_prompts(page)
            if len(cands) < 4:
                page.close()
                continue
            crops, boxes = [], []
            for c in cands:
                x0 = max(0, min(sw - 1, int(c["x"] * sw / 1280)))
                y0 = max(0, min(sh - 1, int(c["y"] * sh / 900)))
                x1 = max(x0 + 1, min(sw, int((c["x"] + c["w"]) * sw / 1280)))
                y1 = max(y0 + 1, min(sh, int((c["y"] + c["h"]) * sh / 900)))
                crops.append(np.asarray(
                    im_full.crop((x0, y0, x1, y1)).resize((224, 224), Image.BICUBIC),
                    dtype=np.float32).transpose(2, 0, 1) / 255.0)
                boxes.append([x0 / sw, y0 / sh, x1 / sw, y1 / sh])

            # frozen features for the crops (one batch)
            px = torch.from_numpy(np.stack(crops)).to(args.device, torch.float32)
            with torch.no_grad():
                pv = ((px - 0.5) / 0.5)
                vo = tower.vision_model(pixel_values=pv)
                v = (vo.pooler_output if hasattr(vo, "pooler_output")
                     else vo.last_hidden_state.mean(dim=1)).float()
            V = v.unsqueeze(0)
            X = torch.tensor(np.array(boxes, dtype=np.float32)).unsqueeze(0).to(args.device)
            M = torch.ones(1, len(cands), dtype=torch.bool, device=args.device)

            uniq = sorted({p["instruction"] for p in prompts})
            tin = proc(text=uniq, return_tensors="pt", padding=True, truncation=True, max_length=32)
            tin = {k: v2.to(args.device) for k, v2 in tin.items()}
            with torch.no_grad():
                to = tower.text_model(**tin)
                tp = (to.pooler_output if hasattr(to, "pooler_output")
                      else to.last_hidden_state.mean(dim=1)).float()
            tmap = {u: tp[i] for i, u in enumerate(uniq)}

            for p in prompts:
                t0 = time.time()
                T = tmap[p["instruction"]].unsqueeze(0)      # (1, Dt)
                with torch.no_grad():
                    # GatedScorer.forward(v, t, box, mask) — the text goes
                    # in the SECOND slot, not the third.
                    logits = head(V, T, X, M)
                lat.append((time.time() - t0) * 1000)
                pick = int(logits.argmax(-1).item())
                # control: no instruction (a constant text embedding)
                T0 = tmap[uniq[0]].unsqueeze(0) * 0.0
                with torch.no_grad():
                    logits0 = head(V, T0, X, M)
                pick0 = int(logits0.argmax(-1).item())
                results.append(dict(page=pi, kind=p["kind"], K=len(cands),
                                    pick=pick, gold=p["idx"],
                                    correct=bool(pick == p["idx"]),
                                    pick_no_instr=pick0,
                                    correct_no_instr=bool(pick0 == p["idx"])))
            page.close()
        browser.close()

    n = len(results)
    acc = sum(r["correct"] for r in results) / max(n, 1)
    acc0 = sum(r["correct_no_instr"] for r in results) / max(n, 1)
    by = defaultdict(lambda: [0, 0])
    for r in results:
        by[r["kind"]][0] += r["correct"]; by[r["kind"]][1] += 1
    K = results[0]["K"] if results else 0
    out = dict(top1=acc, top1_no_instruction=acc0, floor=1.0 / max(K, 1),
               by_kind={k: dict(acc=c / max(v, 1), n=v) for k, (c, v) in by.items()},
               K=K, n=n, latency_ms=float(np.mean(lat)) if lat else 0)
    print(f"\nreal-DOM gated head: top-1 {acc:.4f}  (floor {1/max(K,1):.4f})", flush=True)
    print(f"  no-instruction control: {acc0:.4f}", flush=True)
    print(f"  by kind: {out['by_kind']}", flush=True)
    print(f"  {out['latency_ms']:.1f} ms/decision (head only, tower excluded)", flush=True)
    json.dump(dict(summary=out, results=results), open(args.out, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
