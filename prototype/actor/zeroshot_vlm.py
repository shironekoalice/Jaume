"""Zero-shot single-forward grounding with a VLM, and the same thing trained.

Two things are measured here, and they answer different questions.

1. ZERO-SHOT, SINGLE FORWARD (the deployment path this project wants).
   The screenshot and the candidate crops go into the VLM together, the
   candidates are lettered, and the answer is read from the LOGIT of each
   option letter at the first answer position. No autoregressive decoding,
   no training. This is the cheapest possible actor: the model's own
   knowledge, read out in one pass.

   The same mechanism is also measured in COORDINATE-REGRESSION mode — the
   model is asked to emit the box of the referred element, and the score is
   whether the box centre lands inside the GT box. That is the published
   ScreenSpot metric, and it lets the numbers be compared with the
   literature.

2. TRAINED (the real actor).
   The letter-logit head is trained with a softmax CE over the candidates —
   the same loss the whole project uses (§4.7 of the planning report:
   per-pair BCE collapses; state-internal softmax CE does not).

The comparison (1) vs (2) answers the only question that matters for the
budget: does the actor need training at all, or is a frozen backbone
already enough?

Metrics are reported with the FLOOR next to them, always. On a K-candidate
set the random floor is 1/K; a number without its floor is not a result.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from backbone_registry import CANDIDATES, crop_image
from screenspot_data import build_eval_set


def build_messages(image, instruction, candidates, crop_size=224, letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    crops = [crop_image(image, b, crop_size) for b in candidates]
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
    return [{"role": "user", "content": content}], [image] + crops


@torch.no_grad()
def letter_scores(model, proc, image, instruction, candidates, device, crop_size=224):
    msgs, images = build_messages(image, instruction, candidates, crop_size)
    prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[prompt], images=images, return_tensors="pt")
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    logits = model(**inputs).logits[0, -1].float()
    tok = proc.tokenizer
    out = []
    for i in range(len(candidates)):
        ids = tok.encode("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i], add_special_tokens=False)
        out.append(float(logits[ids[-1]]))
    return np.array(out)


@torch.no_grad()
def regression_box(model, proc, image, instruction, device):
    """Coordinate regression in the published ScreenSpot mode."""
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text":
         f"Locate the element on the screen that this instruction refers to: {instruction}\n"
         f"Answer with the bounding box in the form [[x1, y1, x2, y2]]."}]}]
    prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[prompt], images=[image], return_tensors="pt")
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    out = model.generate(**inputs, max_new_tokens=48, do_sample=False)
    text = proc.batch_decode(out, skip_special_tokens=False)[0]
    import re
    m = re.search(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", text)
    if not m:
        return None
    v = [float(x) for x in m.groups()]
    if max(v) <= 100:  # normalised
        return v
    w, h = image.size
    return [v[0] / w, v[1] / h, v[2] / w, v[3] / h]


def eval_zeroshot(spec, data, device, crop_size=224, do_regression=False):
    scorer = spec["build"]()
    model, proc = scorer.model, scorer.proc
    k = len(data[0]["candidates"])
    hits = 0
    lat = []
    reg_hits = 0
    by_type = defaultdict(lambda: [0, 0])
    for d in data:
        t0 = time.time()
        sc = letter_scores(model, proc, d["image"], d["instruction"], d["candidates"],
                           device, crop_size)
        lat.append((time.time() - t0) * 1000)
        ok = int(np.argmax(sc) == d["gold"])
        hits += ok
        by_type[d["data_type"]][0] += ok
        by_type[d["data_type"]][1] += 1
        if do_regression:
            box = regression_box(model, proc, d["image"], d["instruction"], device)
            if box is not None:
                gx0, gy0, gx1, gy1 = d["gt_box"]
                w, h = d["image"].size
                cx, cy = (box[0] + box[2]) / 2 * w, (box[1] + box[3]) / 2 * h
                reg_hits += int(gx0 <= cx <= gx1 and gy0 <= cy <= gy1)
    out = dict(top1=hits / len(data), top1_floor=1.0 / k,
               latency_ms=float(np.mean(lat)), n=len(data),
               by_type={t: dict(acc=c / max(v, 1), n=v) for t, (c, v) in by_type.items()})
    if do_regression:
        out["regression_acc"] = reg_hits / len(data)
    scorer.unload()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=120)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--backbones", default="qwen3vl_2b")
    ap.add_argument("--regression", action="store_true")
    ap.add_argument("--out", default="zeroshot_vlm.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else (
        "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
    print(f"device {device}", flush=True)
    data = build_eval_set(max_rows=args.rows, k=args.k, seed=0)
    print(f"{len(data)} samples, floor {1/args.k:.4f}", flush=True)

    results = {}
    for name in args.backbones.split(","):
        spec = CANDIDATES[name]
        print(f"\n=== {name} ({spec['id']}) zero-shot single-forward ===", flush=True)
        try:
            r = eval_zeroshot(spec, data, device, do_regression=args.regression)
            results[name] = r
            print(f"  top1        {r['top1']:.4f}  (floor {r['top1_floor']:.4f})", flush=True)
            if "regression_acc" in r:
                print(f"  regression  {r['regression_acc']:.4f}", flush=True)
            print(f"  by type     {r['by_type']}", flush=True)
            print(f"  {r['latency_ms']:.0f} ms/decision", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            results[name] = dict(error=f"{type(e).__name__}: {str(e)[:200]}")
    json.dump(results, open(args.out, "w"), indent=2)
    print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
