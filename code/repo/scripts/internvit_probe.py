"""Probe InternViT-300M-448px on the OCR tasks: does it discriminate glyphs?

This is the decisive test. SigLIP2 scored 0.25 (chance) even with tiling.
InternViT is trained for document/OCR understanding at 448px, so it should
either clearly beat chance (route validated) or fail (route wrong).
"""
import sys, os, torch, torch.nn as nn, torch.nn.functional as F

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
from jaume.device import resolve_device
from ocr_suite import build

IVIT = "/home/developer/.cache/modelscope/models/OpenGVLab--InternViT-300M-448px/snapshots/master"


def main():
    dev = resolve_device("npu")
    # the checkpoint ships its own modeling files that use relative imports,
    # so load it as a package rather than a bare file
    sys.path.insert(0, os.path.dirname(IVIT))
    pkg = "ivit_pkg"
    os.makedirs("/tmp/ivit_pkg", exist_ok=True)
    for f in ("modeling_intern_vit.py", "configuration_intern_vit.py",
              "flash_attention.py"):
        src = os.path.join(IVIT, f)
        if os.path.exists(src):
            import shutil
            shutil.copy(src, "/tmp/ivit_pkg/" + f)
    open("/tmp/ivit_pkg/__init__.py", "w").close()
    sys.path.insert(0, "/tmp")
    from ivit_pkg.modeling_intern_vit import InternVisionModel
    from ivit_pkg.configuration_intern_vit import InternVisionConfig
    from safetensors.torch import load_file
    import json
    cfgd = json.load(open(os.path.join(IVIT, "config.json")))
    vcfg = cfgd.get("vision_config", cfgd)
    conf = InternVisionConfig(**vcfg)
    model = InternVisionModel(conf).to(dev)
    sd = load_file(os.path.join(IVIT, "model.safetensors"))
    own = model.state_dict()
    hit = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    own.update(hit)
    model.load_state_dict(own)
    model.eval()
    print("[ivit] loaded %d/%d tensors" % (len(hit), len(own)), flush=True)
    for p in model.parameters():
        p.requires_grad_(False)
    print("[ivit] params %.1fM" % (sum(p.numel() for p in model.parameters()) / 1e6),
          flush=True)

    import torchvision.transforms.functional as TF
    def prep(imgs, size=448):
        xs = []
        for im in imgs:
            x = TF.to_tensor(im)
            x = TF.resize(x, [size, size], antialias=True)
            xs.append((x - 0.5) / 0.5)
        return torch.stack(xs).to(dev)

    items = [b for b in build(400, seed=31) if b[1] == "text_locate"]
    labels = {"box 1": 0, "box 2": 1, "box 3": 2, "box 4": 3}
    y = torch.tensor([labels[b[3]] for b in items], device=dev)
    n = len(y) // 2

    for size in (448,):
        feats = []
        with torch.no_grad():
            for i in range(0, len(items), 4):
                x = prep([b[0] for b in items[i:i+4]], size)
                out = model(pixel_values=x)
                h = out.last_hidden_state.mean(1) if hasattr(out, "last_hidden_state")                     else out[0].mean(1)
                feats.append(h)
        X = torch.cat(feats)
        lin = nn.Linear(X.shape[1], 4).to(dev)
        opt = torch.optim.AdamW(lin.parameters(), lr=1e-2)
        for _ in range(400):
            loss = F.cross_entropy(lin(X[:n]), y[:n])
            opt.zero_grad(); loss.backward(); opt.step()
        acc = float((lin(X[n:]).argmax(-1) == y[n:]).float().mean().item())
        print("[ivit] text_locate probe @%dpx: acc=%.3f (chance 0.25)" % (size, acc),
              flush=True)


if __name__ == "__main__":
    main()
