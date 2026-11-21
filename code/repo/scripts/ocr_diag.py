"""Diagnose why OCR stays at chance: is it the vision resolution or the head?"""
import sys, os, torch, torch.nn as nn, torch.nn.functional as F

if "torchaudio" not in sys.modules:
    import importlib.machinery as _im, types
    def _stub(n):
        m = types.ModuleType(n); m.__version__="0"; 
        m.__spec__ = _im.ModuleSpec(n, loader=None); m.__path__=[]
        m.__getattr__ = lambda a: None; sys.modules[n]=m; return m
    _stub("torchaudio")
    for _s in ("torchaudio.transforms","torchaudio.functional",
               "torchaudio.models","torchaudio.utils"):
        _stub(_s)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from jaume.device import resolve_device
from jaume.modeling.config import JaumeConfig
from jaume.modeling.embedding import JaumeEmbeddingModel
from jaume.modeling.checkpoint import load_state_dict
from ocr_suite import build

def main():
    dev = resolve_device("npu")
    md = "/mnt/workspace/output/350m-causal-mlm/final"
    cfg = JaumeConfig.load(os.path.join(md, "config.json"))
    enc = JaumeEmbeddingModel(cfg).to(dev)
    sd = load_state_dict(os.path.join(md, "pytorch_model.bin"))
    own = enc.state_dict(); merged = {}
    for k in own:
        merged[k] = sd[k] if (k in sd and sd[k].shape == own[k].shape) else own[k]
    enc.load_state_dict(merged, strict=True)
    enc.eval()

    from transformers import SiglipModel
    S = "/home/developer/.cache/modelscope/models/google--siglip2-base-patch16-224/snapshots/master"
    full = SiglipModel.from_pretrained(S, dtype=torch.float32)
    tower = full.vision_model.to(dev).eval()
    for p in tower.parameters():
        p.requires_grad_(False)

    import torchvision.transforms.functional as TF
    def prep(imgs, size):
        xs = []
        for im in imgs:
            x = TF.to_tensor(im)
            x = TF.resize(x, [size, size], antialias=True)
            xs.append((x - 0.5) / 0.5)
        return torch.stack(xs).to(dev)

    # Pure vision check: can SigLIP2 alone separate the 4 text_locate classes?
    # If a linear probe on pooled SigLIP features cannot do it, resolution is
    # the bottleneck, not the Jaume bridge.
    items = build(400, seed=31)
    items = [b for b in items if b[1] == "text_locate"]
    labels = {"box 1": 0, "box 2": 1, "box 3": 2, "box 4": 3}
    for size in (224, 384, 512):
        with torch.no_grad():
            feats = []
            for i in range(0, len(items), 16):
                x = prep([b[0] for b in items[i:i+16]], size)
                h = tower(pixel_values=x).last_hidden_state.mean(1)
                feats.append(h)
            X = torch.cat(feats)
        y = torch.tensor([labels[b[3]] for b in items], device=dev)
        n = len(y) // 2
        lin = nn.Linear(X.shape[1], 4).to(dev)
        opt = torch.optim.AdamW(lin.parameters(), lr=1e-2)
        for _ in range(300):
            loss = F.cross_entropy(lin(X[:n]), y[:n])
            opt.zero_grad(); loss.backward(); opt.step()
        acc = float((lin(X[n:]).argmax(-1) == y[n:]).float().mean().item())
        print("[diag] text_locate linear probe @%dpx: acc=%.3f" % (size, acc),
              flush=True)

if __name__ == "__main__":
    main()
