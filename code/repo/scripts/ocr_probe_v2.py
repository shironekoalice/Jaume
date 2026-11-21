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
sys.path.insert(0, "/home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master/code/repo/scripts")
from jaume.device import resolve_device
from ocr_suite import build

SIGLIP = "/home/developer/.cache/modelscope/models/google--siglip2-base-patch16-224/snapshots/master"
BOXES = [(20, 40, 150, 190), (170, 40, 300, 190),
         (20, 230, 150, 380), (170, 230, 300, 380)]


def main():
    dev = resolve_device("npu")
    from transformers import SiglipModel
    full = SiglipModel.from_pretrained(SIGLIP, dtype=torch.float32)
    tower = full.vision_model.to(dev).eval()
    for p in tower.parameters():
        p.requires_grad_(False)

    import torchvision.transforms.functional as TF
    def prep(boxes):
        xs = []
        for b in boxes:
            x = TF.to_tensor(b)
            x = TF.resize(x, [224, 224], antialias=True)
            xs.append((x - 0.5) / 0.5)
        return torch.stack(xs).to(dev)

    items = [b for b in build(600, seed=31) if b[1] == "text_locate"]
    labels = {"box 1": 0, "box 2": 1, "box 3": 2, "box 4": 3}
    y = torch.tensor([labels[b[3]] for b in items], device=dev)
    n = len(y) // 2

    feats = []
    with torch.no_grad():
        for i in range(0, len(items), 8):
            batch = items[i:i + 8]
            crops = []
            for b in batch:
                for (x0, y0, x1, y1) in BOXES:
                    crops.append(b[0].crop((x0, y0, x1, y1)))
            h = tower(pixel_values=prep(crops)).last_hidden_state.mean(1)
            h = h.view(len(batch), 4, -1)          # [B, 4 boxes, d]
            feats.append(h.flatten(1))             # concat box features
    X = torch.cat(feats)
    print("[probe2] features", tuple(X.shape), flush=True)

    lin = nn.Linear(X.shape[1], 4).to(dev)
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-2)
    for _ in range(500):
        loss = F.cross_entropy(lin(X[:n]), y[:n])
        opt.zero_grad(); loss.backward(); opt.step()
    acc = float((lin(X[n:]).argmax(-1) == y[n:]).float().mean().item())
    print("[probe2] per-box SigLIP2 probe acc=%.3f (chance 0.25)" % acc,
          flush=True)


if __name__ == "__main__":
    main()
