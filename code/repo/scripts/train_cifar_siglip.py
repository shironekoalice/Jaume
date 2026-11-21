"""Fine-tune the Jaume + SigLIP2 bridge on real images (CIFAR-10).

Purpose: the basic-vision suite (0.95 mean) is synthetic. Before trusting the
bridge on real UI/document images we need one real-image checkpoint. CIFAR-10
is the cheapest real dataset that fits the 61GB box and downloads from
ModelScope in seconds.

Setup mirrors the suite: frozen SigLIP2 tower (top-k blocks optional),
projector, frozen Jaume backbone with top-k layers unfrozen, linear head.
Metric: top-1 accuracy on the official 10k test set.
"""
import argparse
import os
import pickle
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if "torchaudio" not in sys.modules:
    import importlib.machinery as _im
    import types

    def _stub(name):
        m = types.ModuleType(name)
        m.__version__ = "0.0.0-stub"
        m.__spec__ = _im.ModuleSpec(name, loader=None)
        m.__path__ = []
        m.__getattr__ = lambda a: None
        sys.modules[name] = m
        return m

    _stub("torchaudio")
    for _s in ("torchaudio.transforms", "torchaudio.functional",
               "torchaudio.models", "torchaudio.utils"):
        _stub(_s)

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
REPO = os.path.join(PROJECT_ROOT, "code", "repo")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jaume.device import resolve_device, seed_all, describe

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def load_cifar(root):
    import glob
    xs, ys = [], []
    for f in sorted(glob.glob(os.path.join(root, "data_batch_*"))):
        with open(f, "rb") as fh:
            d = pickle.load(fh, encoding="bytes")
        xs.append(d[b"data"]); ys.extend(d[b"labels"])
    xtr = np.concatenate(xs).reshape(-1, 3, 32, 32)
    ytr = np.array(ys)
    with open(os.path.join(root, "test_batch"), "rb") as fh:
        d = pickle.load(fh, encoding="bytes")
    xte = d[b"data"].reshape(-1, 3, 32, 32)
    yte = np.array(d[b"labels"])
    return xtr, ytr, xte, yte


def to_chw(arr):
    # CIFAR stores row-major 1024 R, 1024 G, 1024 B
    return arr.transpose(0, 2, 3, 1)


def make_tensors(arr, device, size=224):
    import torchvision.transforms.functional as TF
    xs = torch.from_numpy(to_chw(arr)).float().permute(0, 3, 1, 2) / 255.0
    xs = TF.resize(xs, [size, size], antialias=True)
    mean = torch.tensor([0.4815, 0.4578, 0.4082]).view(1, 3, 1, 1)
    std = torch.tensor([0.2686, 0.2613, 0.2758]).view(1, 3, 1, 1)
    return ((xs - mean) / std).to(device)


class SiglipBridge(nn.Module):
    def __init__(self, d_out, path, unfreeze_blocks=0):
        super().__init__()
        from transformers import SiglipModel
        full = SiglipModel.from_pretrained(path, dtype=torch.float32)
        self.tower = full.vision_model
        hidden = self.tower.config.hidden_size
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, d_out), nn.GELU(),
            nn.Linear(d_out, d_out))
        for p in self.tower.parameters():
            p.requires_grad_(False)
        if unfreeze_blocks > 0:
            for blk in list(self.tower.encoder.layers)[-unfreeze_blocks:]:
                for p in blk.parameters():
                    p.requires_grad_(True)
            for nm in ("post_layernorm", "head"):
                mod = getattr(self.tower, nm, None)
                if mod is not None:
                    for p in mod.parameters():
                        p.requires_grad_(True)

    def forward(self, x):
        t = self.tower(pixel_values=x).last_hidden_state
        return self.proj(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--siglip-path", required=True)
    ap.add_argument("--cifar-root", required=True)
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--siglip-lr", type=float, default=2e-5)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--unfreeze-layers", type=int, default=4)
    ap.add_argument("--siglip-unfreeze-blocks", type=int, default=0)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)
    print("[cf] device=%s (%s)" % (device, describe(device)), flush=True)
    seed_all(device, 0)

    from jaume.modeling.config import JaumeConfig
    from jaume.modeling.embedding import JaumeEmbeddingModel
    from jaume.modeling.checkpoint import load_state_dict

    cfg = JaumeConfig.load(os.path.join(args.model, "config.json"))
    cfg.num_layers = args.depth
    cfg.causal = True

    enc = JaumeEmbeddingModel(cfg).to(device)
    pre = load_state_dict(os.path.join(args.model, "pytorch_model.bin"))
    own, merged, up = enc.state_dict(), {}, 0
    for k in own:
        if k in pre and pre[k].shape == own[k].shape:
            merged[k] = pre[k]
        elif k.startswith("backbone.layers."):
            rest = k.split(".", 3)[3]
            s = "backbone.layers.15.%s" % rest
            if s in pre:
                merged[k] = pre[s]; up += 1
            else:
                merged[k] = own[k]
        else:
            merged[k] = own[k]
    enc.load_state_dict(merged, strict=True)
    for p in enc.parameters():
        p.requires_grad_(False)
    if args.unfreeze_layers > 0:
        for layer in list(enc.backbone.layers)[-args.unfreeze_layers:]:
            for p in layer.parameters():
                p.requires_grad_(True)
        for p in enc.backbone.norm.parameters():
            p.requires_grad_(True)

    d = cfg.hidden_size
    bridge = SiglipBridge(d, args.siglip_path,
                          unfreeze_blocks=args.siglip_unfreeze_blocks).to(device)
    # 10-way classification head over pooled bridge output
    head = nn.Linear(d, 10).to(device)
    print("[cf] up=%d depth=%d" % (up, cfg.num_layers), flush=True)

    t0 = time.time()
    xtr_n, ytr, xte_n, yte = load_cifar(args.cifar_root)
    xtr = make_tensors(xtr_n[:args.n_train], device, args.img_size)
    ytr_t = torch.from_numpy(ytr[:args.n_train]).to(device)
    xte = make_tensors(xte_n, device, args.img_size)
    yte_t = torch.from_numpy(yte).to(device)
    print("[cf] data %d/%d (%.0fs)" % (
        len(xtr), len(xte), time.time() - t0), flush=True)

    vis_tr = [p for p in bridge.tower.parameters() if p.requires_grad]
    proj = list(bridge.proj.parameters())
    encp = [p for p in enc.parameters() if p.requires_grad]
    groups = [{"params": proj + list(head.parameters()) + encp, "lr": args.lr}]
    if vis_tr:
        groups.append({"params": vis_tr, "lr": args.siglip_lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    allp = proj + list(head.parameters()) + encp + vis_tr

    def embed(x):
        v = bridge(x)                       # B,196,d
        mask = torch.ones(v.shape[0], v.shape[1], dtype=torch.long,
                          device=device)
        h = v
        for layer in enc.backbone.layers:
            h, _a = layer(h, mask)
        h = enc.backbone.norm(h)
        return h.mean(1)

    def evaluate():
        bridge.eval(); head.eval(); enc.eval()
        correct = 0
        with torch.no_grad():
            for i in range(0, len(xte), 128):
                logits = head(embed(xte[i:i + 128]))
                correct += int((logits.argmax(-1) == yte_t[i:i + 128]).sum())
        return correct / len(xte)

    for ep in range(args.epochs):
        bridge.train(); head.train(); enc.train()
        perm = torch.randperm(len(xtr), device=device)
        tot = nb = 0
        t0 = time.time()
        for i in range(0, len(xtr) - args.batch_size + 1, args.batch_size):
            idx = perm[i:i + args.batch_size]
            logits = head(embed(xtr[idx]))
            loss = F.cross_entropy(logits, ytr_t[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(allp, 1.0)
            opt.step()
            tot += float(loss); nb += 1
        acc = evaluate()
        print("[cf] ep %d/%d loss %.3f TEST_ACC %.4f (%.0fs)" % (
            ep + 1, args.epochs, tot / max(nb, 1), acc, time.time() - t0),
            flush=True)

    acc = evaluate()
    print("[cf] FINAL %.4f" % acc, flush=True)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        torch.save({"bridge": bridge.state_dict(), "head": head.state_dict()},
                   os.path.join(args.out, "cifar_head.pt"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
