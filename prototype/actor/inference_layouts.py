"""Latency and accuracy of DIFFERENT inference layouts for one decision.

One decision on ScreenSpot can be made in several ways, and they differ by
an order of magnitude in cost:

  L1  FULL-PROMPT      screenshot + K candidate crops + instruction, one
                       forward, read the option-letter logits.
                       Cost: K+1 images through the vision tower.
                       Measured: ~8.7 s/decision on this XPU (7 images).

  L2  SCREEN-ONLY      screenshot + instruction, one forward, the model
                       produces a box (coordinate regression).
                       Cost: 1 image. This is the published ScreenSpot mode.

  L3  CROP-ONLY        candidate crops one at a time, score each against
                       the instruction, pick the best. Cost: K forwards of
                       ONE image each.

  L4  PROBE + PLAN     freeze the backbone, train a linear head on the
                       frozen features, and at inference run the backbone
                       ONCE on the screen (K crops is the expensive part).
                       Cost: 1 image.

L1 is the most natural but the most expensive. This script measures L1 and
L2 on the same data with the same model, and reports the accuracy/latency
trade-off, because the choice between them is a budget decision and should
be made on numbers rather than intuition.

Every number is reported next to its floor.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict

import numpy as np
import torch

from backbone_registry import CANDIDATES
from screenspot_data import build_eval_set
from zeroshot_vlm import build_messages, letter_scores


@torch.no_grad()
def regression_box(model, proc, image, instruction, device, max_new=48):
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text":
         f"Locate the element on the screen that this instruction refers to: {instruction}\n"
         f"Answer with the bounding box in the form [[x1, y1, x2, y2]]."}]}]
    prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[prompt], images=[image], return_tensors="pt")
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    text = proc.batch_decode(out, skip_special_tokens=False)[0]
    m = re.search(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", text)
    if not m:
        return None, text
    v = [float(x) for x in m.groups()]
    if max(v) <= 100:
        return v, text
    w, h = image.size
    return [v[0] / w, v[1] / h, v[2] / w, v[3] / h], text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=60)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--backbone", default="qwen3vl_2b")
    ap.add_argument("--out", default="inference_layouts.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else (
        "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
    data = build_eval_set(max_rows=args.rows, k=args.k, seed=0)
    spec = CANDIDATES[args.backbone]
    print(f"{args.backbone}: {len(data)} samples, floor {1/args.k:.4f}", flush=True)

    scorer = spec["build"]()
    model, proc = scorer.model, scorer.proc
    k = len(data[0]["candidates"])

    # ---- L1 full prompt (screenshot + K crops), letter logits
    hits, lat = 0, []
    for d in data:
        t0 = time.time()
        sc = letter_scores(model, proc, d["image"], d["instruction"], d["candidates"], device)
        lat.append((time.time() - t0) * 1000)
        hits += int(np.argmax(sc) == d["gold"])
    l1 = dict(top1=hits / len(data), floor=1.0 / k, latency_ms=float(np.mean(lat)),
              images_per_forward=k + 1)

    # ---- L2 screen only, coordinate regression
    reg_hits, reg_lat, parsed = 0, [], 0
    for d in data:
        t0 = time.time()
        box, text = regression_box(model, proc, d["image"], d["instruction"], device)
        reg_lat.append((time.time() - t0) * 1000)
        if box is None:
            continue
        parsed += 1
        gx0, gy0, gx1, gy1 = d["gt_box"]
        w, h = d["image"].size
        cx, cy = (box[0] + box[2]) / 2 * w, (box[1] + box[3]) / 2 * h
        reg_hits += int(gx0 <= cx <= gx1 and gy0 <= cy <= gy1)
    l2 = dict(regression_acc=reg_hits / len(data), parsed=parsed / len(data),
              latency_ms=float(np.mean(reg_lat)), images_per_forward=1)

    scorer.unload()
    res = {args.backbone: dict(L1_full_prompt=l1, L2_screen_only_regression=l2)}
    json.dump(res, open(args.out, "w"), indent=2)
    print("\nL1 full prompt (K+1 images, letter logits):", json.dumps(l1, indent=2))
    print("L2 screen only (1 image, box):", json.dumps(l2, indent=2))


if __name__ == "__main__":
    main()
