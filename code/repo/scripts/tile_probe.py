"""Does tiling actually restore information? Linear probe, whole vs tiled."""
import sys, os, torch, torch.nn as nn, torch.nn.functional as F

if "torchaudio" not in sys.modules:
    import importlib.machinery as _im, types
    def _stub(n):
        m = types.ModuleType(n); m.__version__="0"
        m.__spec__ = _im.ModuleSpec(n, loader=None); m.__path__=[]
        m.__getattr__ = lambda a: None; sys.modules[n]=m; return m
    _stub("torchaudio")
    for _s in ("torchaudio.transforms","torchaudio.functional",
               "torchaudio.models","torchaudio.utils"):
        _stub(_s)

sys.path.insert(0, "/home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master/code/repo")
from jaume.device import resolve_device
from ocr_suite import build


def split_tiles(img, grid, overlap=0.15):
    W, H = img.size
    tw, th = W / grid, H / grid
    ow, oh = tw * overlap, th * overlap
    out = []
    for gy in range(grid):
        for gx in range(grid):
            x0 = max(0, int(gx * tw - ow)); y0 = max(0, int(gy * th - oh))
            x1 = min(W, int((gx + 1) * tw + ow)); y1 = min(H, int((gy + 1) * th + oh))
            out.append(img.crop((x0, y0, x1, y1)))
    return out


def main():
    dev = resolve_device("npu")
    from transformers import SiglipModel
    S = "/home/developer/.cache/modelscope/models/google--siglip2-base-patch16-224/snapshots/master"
    full = SiglipModel.from_pretrained(S, dtype=torch.float32)
    tower = full.vision_model.to(dev).eval()
    for p in tower.parameters():
        p.requires_grad_(False)

    import torchvision.transforms.functional as TF
    def prep(imgs):
        xs = []
        for im in imgs:
            x = TF.to_tensor(im); x = TF.resize(x, [224,224], antialias=True)
            xs.append((x - 0.5) / 0.5)
        return torch.stack(xs).to(dev)

    items = [b for b in build(400, seed=31) if b[1] == "text_locate"]
    labels = {"box 1":0,"box 2":1,"box 3":2,"box 4":3}
    y = torch.tensor([labels[b[3]] for b in items], device=dev)
    n = len(y)//2

    for grid in (1, 2, 3):
        feats = []
        with torch.no_grad():
            for i in range(0, len(items), 8):
                batch = items[i:i+8]
                tiles = []
                for b in batch:
                    tiles.extend(split_tiles(b[0], grid))
                x = prep(tiles)
                h = tower(pixel_values=x).last_hidden_state.mean(1)
                h = h.view(len(batch), grid*grid, -1).mean(1)  # pool tiles
                feats.append(h)
        X = torch.cat(feats)
        lin = nn.Linear(X.shape[1], 4).to(dev)
        opt = torch.optim.AdamW(lin.parameters(), lr=1e-2)
        for _ in range(400):
            loss = F.cross_entropy(lin(X[:n]), y[:n])
            opt.zero_grad(); loss.backward(); opt.step()
        acc = float((lin(X[n:]).argmax(-1) == y[n:]).float().mean().item())
        print("[tileprobe] grid=%d (%d tiles) acc=%.3f" % (grid, grid*grid, acc),
              flush=True)


if __name__ == "__main__":
    main()
