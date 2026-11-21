"""Is funsd_read solvable? Test TrOCR directly on the same crops."""
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
sys.path.insert(0, "/home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master/code/repo/scripts")
from jaume.device import resolve_device
from funsd_env import build

TRO = "/home/developer/.cache/modelscope/models/microsoft--trocr-base-printed/snapshots/master"

def main():
    dev = resolve_device("npu")
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    proc = TrOCRProcessor.from_pretrained(TRO)
    model = VisionEncoderDecoderModel.from_pretrained(TRO).to(dev).eval()

    items = [b for b in build(80, "/mnt/workspace/data/funsd/FUNSD/funsd", seed=5)
             if b[1] == "funsd_read"]
    print("[fdiag] n =", len(items), flush=True)
    exact = partial = 0
    for i, (img, fam, which, ans, opts, q) in enumerate(items):
        px = proc(images=img.convert("RGB"), return_tensors="pt").pixel_values.to(dev)
        with torch.no_grad():
            ids = model.generate(px, max_new_tokens=32)
        got = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
        want = ans.strip()
        e = got == want
        p = want and want[:4] in got
        exact += int(e); partial += int(p)
        if i < 8:
            print("[fdiag] want=%-22r got=%-22r %s" % (want[:22], got[:22],
                  "EXACT" if e else ("partial" if p else "X")), flush=True)
    print("[fdiag] TrOCR exact=%.3f partial=%.3f over %d"
          % (exact/len(items), partial/len(items), len(items)), flush=True)

if __name__ == "__main__":
    main()
