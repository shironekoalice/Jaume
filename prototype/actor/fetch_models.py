"""Fetch backbone weights from ModelScope (HuggingFace is unreachable here).

Kept as a script so the download is reproducible and restartable: each model
is fetched into prototype/actor/models/<id>/ and skipped if already there.
"""
from __future__ import annotations

import os
import sys
import time

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# The selection set spans the design space on purpose:
#   a pure vision tower with no LM at all
#   a task-prompted grounding specialist
#   the smallest general VLM
#   the current SOTA shape (2B) and its previous generation as a control
MODELS = [
    "google/siglip2-base-patch16-224",
    "microsoft/Florence-2-base",
    "OpenGVLab/InternVL2-1B",
    "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen/Qwen2-VL-2B-Instruct",
]


def main():
    from modelscope.hub.snapshot_download import snapshot_download

    only = sys.argv[1:] or MODELS
    os.makedirs(MODELS_DIR, exist_ok=True)
    for mid in only:
        local = os.path.join(MODELS_DIR, mid.replace("/", "__"))
        if os.path.isdir(local) and os.listdir(local):
            print(f"skip (exists) {mid}", flush=True)
            continue
        t0 = time.time()
        try:
            snapshot_download(mid, local_dir=local)
            print(f"OK   {mid}  {time.time()-t0:.0f}s  -> {local}", flush=True)
        except Exception as e:
            print(f"FAIL {mid}: {type(e).__name__}: {str(e)[:140]}", flush=True)


if __name__ == "__main__":
    main()
