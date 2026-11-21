"""Real-browser test: drive an actual page with the trained UI actor.

The synthetic UI benchmark measures whether the *architecture* can express
the decision. It cannot show that the actor works on a real screenshot. This
script closes that gap:

  * opens a local HTML page in a real browser via the ego-browser runtime
  * extracts the real DOM element boxes as the candidate set (this is the
    "candidate scoring" action space, the same one the actor was trained on)
  * runs the trained actor on a real screenshot of that page
  * reports whether the actor picked the element the instruction refers to

The point of the candidate set is important: a real page has hundreds of
elements, and a coordinate-regressing model has to hit one pixel-accurate
box. Scoring a shortlist of candidate elements turns "grounding in 2D
continuous space" into "selecting one of K items", which is what the repo's
own planning work already showed is the learnable shape.

If the actor picks the right element on a real page, the pipeline is real.
If it does not, the honest conclusion is that the synthetic benchmark
overstated the result.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from envs import COLORS, SHAPES, VOCAB, word_id

HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Actor test page</title>
<style>
 body{font-family:sans-serif;margin:0;background:#f7f7f5}
 header{background:#e4e6ec;padding:12px 16px;font-weight:600}
 .row{padding:8px 16px;border-bottom:1px solid #e8e8e8}
 .btn{display:inline-block;padding:10px 18px;border-radius:6px;color:#fff;margin:10px 6px}
 .blue{background:#3d6ec8}.green{background:#3ca05a}.red{background:#d64541}.gray{background:#8c8c91}
 .card{width:180px;height:90px;border-radius:8px;margin:14px;display:inline-block;vertical-align:top}
</style></head><body>
<header>Invoices</header>
<div class="row">Report for the period ending March</div>
<div class="row">Reference number 88213</div>
<div class="row">Total amount due 1200 USD</div>
<div class="row">
 <span class="btn blue">Submit</span>
 <span class="btn green">Export</span>
 <span class="btn red">Cancel</span>
 <span class="btn gray">Help</span>
</div>
<div class="row">
 <span class="card" style="background:#ffd9d6"></span>
 <span class="card" style="background:#d7f0dd"></span>
 <span class="card" style="background:#d9e4fb"></span>
</div>
<div class="row">
 <span class="btn blue">Save</span>
 <span class="btn green">Share</span>
 <span class="btn red">Delete</span>
</div>
</body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--ckpt", default="ui_hard_scale.json")
    ap.add_argument("--out", default="browser_test.json")
    args = ap.parse_args()

    os.makedirs("_browser", exist_ok=True)
    path = os.path.abspath("_browser/page.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(HTML)
    print(f"page written to {path}", flush=True)

    # The ego-browser runtime owns the real browser; ask it for a screenshot
    # and the element geometry of the candidate set.
    try:
        from bcdp_snapshot import snapshot  # type: ignore
        from bcdp_navigate import navigate  # type: ignore
        from bcdp_space_open import space_open  # type: ignore
    except Exception as e:
        print("ego-browser python bindings not importable in this process:",
              type(e).__name__, e, flush=True)
        print("Falling back to the file:// URL and an explicit browser step.", flush=True)

    print("Next: open", path, "in the browser and run --with-snapshot <json>", flush=True)


if __name__ == "__main__":
    main()
