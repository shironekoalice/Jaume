"""Backbone selection: the summary.

WHAT WAS MEASURED. ScreenSpot (1272 real screenshots, Apache-2.0) with a
constructed candidate set (the ground-truth box + size-matched distractors
from the same screenshot, so the answer cannot be found from box size).
The metric is the ACTOR's action space — score the candidates, take the
argmax — not the published coordinate-regression leaderboard.

TWO READOUTS PER BACKBONE.

  ZERO-SHOT   the backbone's own answer, read out in one forward pass
              (a VLM reads the logits of the option letters; a dual tower
              reads the cosine similarity of crop and text embeddings).

  PROBE       a linear head trained on the FROZEN features, evaluated on a
              HELD-OUT split of the screenshots. This is the repo's
              standard method ("a three-minute linear probe on the frozen
              encoder answers that before any training run").

The gap between the two is the headline: it says how much of the
backbone's knowledge is actually reachable by a cheap head, and therefore
how much training the actor needs.

TWO CORRECTIONS MADE DURING THIS EXPERIMENT, both recorded because they
would have produced wrong conclusions:

  1. The probe was first trained AND evaluated on the same screenshots. Its
     "AUROC 0.99" was a fit number. Every probe number here is on a
     held-out split (70/30).
  2. The zero-shot dual-tower readout scored BELOW the random floor
     (0.083 vs 0.167). That is a property of the readout, not of the
     features: the probe on the same features reaches 0.85.
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

FILES = {
    "siglip2_base": "probe_siglip.json",
    "qwen3vl_2b": "probe_qwen3vl.json",
    "qwen2vl_2b": "probe_qwen2vl.json",
}

META = {
    "siglip2_base": dict(params="93M", kind="dual tower (vision + text, no LM)",
                          ms_per_crop=44),
    "qwen3vl_2b": dict(params="2B", kind="VLM", ms_per_crop=481),
    "qwen2vl_2b": dict(params="2B", kind="VLM (previous generation)", ms_per_crop=580),
}


def load():
    out = {}
    for name, f in FILES.items():
        p = os.path.join(HERE, f)
        if not os.path.exists(p):
            continue
        d = json.load(open(p, encoding="utf-8"))
        if name in d and "auroc" in d[name]:
            out[name] = d[name]
    zs = os.path.join(HERE, "zs_qwen3vl.json")
    if os.path.exists(zs):
        out.setdefault("_zeroshot", {})["qwen3vl_2b"] = json.load(open(zs, encoding="utf-8"))["qwen3vl_2b"]
    return out


def main():
    d = load()
    probes = {k: v for k, v in d.items() if k != "_zeroshot"}
    zs = d.get("_zeroshot", {})

    print(f"{'backbone':<16}{'params':<8}{'kind':<38}"
          f"{'AUROC':>8}{'top1':>8}{'floor':>8}")
    print("-" * 86)
    rows = []
    for name, r in sorted(probes.items(), key=lambda kv: -kv[1]["top1"]):
        m = META.get(name, {})
        print(f"{name:<16}{m.get('params','?'):<8}{m.get('kind','?'):<38}"
              f"{r['auroc']:>8.4f}{r['top1']:>8.4f}{r['top1_floor']:>8.4f}")
        rows.append(dict(name=name, **m, **{k: v for k, v in r.items() if k != "per_seed"}))

    if zs:
        print("\nzero-shot single-forward (no training):")
        for name, r in zs.items():
            print(f"  {name:<16} top1 {r['top1']:.4f} (floor {r['top1_floor']:.4f})  "
                  f"{r['latency_ms']:.0f} ms/decision  {r['images_per_forward'] if 'images_per_forward' in r else '?'} images")

    print("\nreading:")
    if probes:
        best = max(probes.items(), key=lambda kv: kv[1]["top1"])
        small = min((k for k in probes if META.get(k, {}).get("params") == "93M"), default=None)
        print(f"  best held-out top-1: {best[0]} at {best[1]['top1']:.4f}")
        if small and small != best[0]:
            r = probes[small]
            print(f"  smallest backbone ({small}, 93M): {r['top1']:.4f} "
                  f"({100*r['top1']/best[1]['top1']:.0f}% of the best) at 1/22 the parameters")
    json.dump(dict(probes=rows, zeroshot=zs), open(os.path.join(HERE, "backbone_summary.json"), "w"),
              indent=2)


if __name__ == "__main__":
    main()
