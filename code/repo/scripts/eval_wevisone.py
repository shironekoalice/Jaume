"""Evaluate WeVisOne-VL on the SAME test sets as our dual tower.

Purpose: establish a target water line. WeVisOne-VL is a 2B model built on
Qwen3-VL with structured coordinate tokens; if it scores much higher than our
350M dual tower on the same data, that gap tells us what Jev must reach (and
whether our architecture or our base model is the limiter).

Scoring mirrors our suites exactly:
  * OCR / vision / FUNSD are QA: we compare the normalised answer string
  * CIFAR is 10-way: we ask the model to name the object
"""
import argparse, os, sys, string, time
import torch

if "torchaudio" not in sys.modules:
    import importlib.machinery as _im, types
    def _stub(n):
        m = types.ModuleType(n); m.__version__ = "0"
        m.__spec__ = _im.ModuleSpec(n, loader=None); m.__path__ = []
        m.__getattr__ = lambda a: None; sys.modules[n] = m; return m
    _stub("torchaudio")
    for _s in ("torchaudio.transforms", "torchaudio.functional",
               "torchaudio.models", "torchaudio.utils"):
        _stub(_s)

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code", "repo"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jaume.device import resolve_device
from ocr_suite import build as build_ocr, OCR_FAMILIES
from vision_suite import build as build_vis
from funsd_env import build as build_funsd

VIS_FAMILIES = ["attr", "count", "spatial", "compare", "scene", "multi_attr"]


import re as _re


def strip_tags(s):
    """WeVisOne emits structured tags (<ref>..</ref>, <box>.., <OBJ>).

    Generation is often truncated mid-tag, so a strict regex misses the
    unclosed case and every prediction scores zero even when the reading is
    perfect. Strip anything from a literal '<' to the next space/end.
    """
    s = s or ""
    s = s.replace("<OBJ>", " ")
    out = []
    depth = 0
    for ch in s:
        if ch == "<":
            depth += 1
        elif ch == ">" and depth:
            depth -= 1
        elif depth == 0:
            out.append(ch)
    return "".join(out)


def norm(s):
    return " ".join(strip_tags(s).split()).upper()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--funsd-root", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-new", type=int, default=64)
    a = ap.parse_args()

    dev = resolve_device(a.device)
    print("[wvo] device=%s" % dev, flush=True)

    from transformers import AutoProcessor, AutoModelForImageTextToText
    proc = AutoProcessor.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        a.model, dtype=torch.bfloat16, trust_remote_code=True,
        device_map=None).to(dev).eval()
    print("[wvo] model loaded", flush=True)

    def ask(img, question):
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": question}]}]
        text = proc.apply_chat_template(msgs, tokenize=False,
                                        add_generation_prompt=True)
        inputs = proc(text=[text], images=[img], return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=a.max_new,
                                 do_sample=False)
        gen = out[:, inputs["input_ids"].shape[1]:]
        return proc.batch_decode(gen, skip_special_tokens=True)[0].strip()

    def run_suite(name, items, families):
        per = {f: [0, 0] for f in families}
        for i, (img, fam, which, ans, opts, q) in enumerate(items):
            got = ask(img, q)
            per[fam][1] += 1
            per[fam][0] += int(norm(got) == norm(ans))
            if i < 3:
                print("[wvo]   %s want=%-14r got=%-14r" % (
                    fam, ans[:14], got[:14]), flush=True)
        m = sum(per[f][0] / max(per[f][1], 1) for f in families) / len(families)
        print("[wvo] %-8s mean=%.3f  %s" % (
            name, m, " ".join("%s=%.2f" % (f, per[f][0] / max(per[f][1], 1))
                              for f in families)), flush=True)
        return m

    t0 = time.time()
    ocr = build_ocr(a.n, seed=909)
    vis = build_vis(a.n, seed=777)
    print("[wvo] data ocr=%d vis=%d" % (len(ocr), len(vis)), flush=True)
    run_suite("OCR", ocr, OCR_FAMILIES)
    run_suite("VIS", vis, VIS_FAMILIES)
    if a.funsd_root and os.path.isdir(a.funsd_root):
        fun = build_funsd(a.n, a.funsd_root, seed=99)
        run_suite("FUNSD", fun, ["funsd_read", "funsd_find"])
    print("[wvo] total %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
