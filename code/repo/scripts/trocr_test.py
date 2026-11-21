"""Can TrOCR actually read the words in our OCR task? Direct test."""
import sys, os, torch

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

sys.path.insert(0, "/home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master/code/repo")
sys.path.insert(0, "/home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master/code/repo/scripts")
from jaume.device import resolve_device
from ocr_suite import build

TROCR = "/home/developer/.cache/modelscope/models/microsoft--trocr-base-printed/snapshots/master"
BOXES = [(20, 40, 150, 190), (170, 40, 300, 190),
         (20, 230, 150, 380), (170, 230, 300, 380)]


def main():
    dev = resolve_device("npu")
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    proc = TrOCRProcessor.from_pretrained(TROCR)
    model = VisionEncoderDecoderModel.from_pretrained(TROCR).to(dev).eval()

    items = [b for b in build(60, seed=31) if b[1] == "text_locate"]
    correct = 0
    shown = 0
    for img, fam, which, ans, _, q in items:
        target = q.split()[-1].rstrip("?")
        reads = []
        for (x0, y0, x1, y1) in BOXES:
            crop = img.crop((x0, y0, x1, y1)).convert("RGB")
            px = proc(images=crop, return_tensors="pt").pixel_values.to(dev)
            with torch.no_grad():
                ids = model.generate(px, max_new_tokens=8)
            reads.append(proc.batch_decode(ids, skip_special_tokens=True)[0].strip())
        # which box reads closest to the target word?
        def score(s):
            s = s.upper().replace(" ", "")
            t = target.upper()
            return sum(1 for a, b in zip(s, t) if a == b) - abs(len(s) - len(t))
        pick = max(range(4), key=lambda i: score(reads[i]))
        ok = ("box %d" % (pick + 1)) == ans
        correct += int(ok)
        if shown < 6:
            print("[trocr] target=%-6s reads=%s -> pick=box%d ans=%s %s"
                  % (target, reads, pick + 1, ans, "OK" if ok else "X"),
                  flush=True)
            shown += 1
    print("[trocr] box-selection acc = %.3f over %d samples (chance 0.25)"
          % (correct / len(items), len(items)), flush=True)


if __name__ == "__main__":
    main()
