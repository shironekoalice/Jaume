import argparse
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
REPO = os.path.join(PROJECT_ROOT, "code", "repo")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jaume.device import resolve_device, seed_all, describe

IMG = 32
PATCH = 8
N_PATCH = (IMG // PATCH) ** 2
SHAPES = ["circle", "square", "triangle"]


def make_image(n_shapes, query_shape, seed, force_first=None):
    rng = random.Random(seed)
    img = [[[0, 0, 0] for _ in range(IMG)] for _ in range(IMG)]
    counts = {s: 0 for s in SHAPES}
    placed = []
    for idx in range(n_shapes):
        s = force_first if (idx == 0 and force_first) else rng.choice(SHAPES)
        for _try in range(20):
            cx = rng.randint(4, IMG - 5)
            cy = rng.randint(4, IMG - 5)
            if all((cx - x) ** 2 + (cy - y) ** 2 > 36 for x, y in placed):
                placed.append((cx, cy))
                counts[s] += 1
                ci = SHAPES.index(s)
                r = 3
                for y in range(max(0, cy - r), min(IMG, cy + r + 1)):
                    for x in range(max(0, cx - r), min(IMG, cx + r + 1)):
                        dx, dy = x - cx, y - cy
                        if s == "circle" and dx * dx + dy * dy > r * r:
                            continue
                        if s == "square" and (abs(dx) > 2 or abs(dy) > 2):
                            continue
                        if s == "triangle" and (dy > 0 or abs(dx) > (dy + 3)):
                            continue
                        img[y][x][ci] = 255
                break
    return img, counts[query_shape]


def make_image_exact(target, query_shape, rng):
    img = [[[0, 0, 0] for _ in range(IMG)] for _ in range(IMG)]
    shapes = [query_shape] * target
    for _ in range(rng.randint(0, 2)):
        shapes.append(rng.choice([x for x in SHAPES if x != query_shape]))
    rng.shuffle(shapes)
    placed = []
    for sh in shapes:
        for _try in range(30):
            cx = rng.randint(4, IMG - 5)
            cy = rng.randint(4, IMG - 5)
            if all((cx - x) ** 2 + (cy - y) ** 2 > 36 for x, y in placed):
                placed.append((cx, cy))
                ci = SHAPES.index(sh)
                r = 3
                for y in range(max(0, cy - r), min(IMG, cy + r + 1)):
                    for x in range(max(0, cx - r), min(IMG, cx + r + 1)):
                        dx, dy = x - cx, y - cy
                        if sh == 'circle' and dx * dx + dy * dy > r * r:
                            continue
                        if sh == 'square' and (abs(dx) > 2 or abs(dy) > 2):
                            continue
                        if sh == 'triangle' and (dy > 0 or abs(dx) > (dy + 3)):
                            continue
                        img[y][x][ci] = 255
                break
    return img, target


COLORS = ["red", "green", "blue"]


def make_image_color(target, query_shape, query_color, rng):
    img = [[[0, 0, 0] for _ in range(IMG)] for _ in range(IMG)]
    shapes = [(query_shape, query_color)] * target
    for _ in range(rng.randint(0, 2)):
        alt = rng.choice([x for x in SHAPES if x != query_shape])
        shapes.append((alt, rng.choice(COLORS)))
    rng.shuffle(shapes)
    placed = []
    for sh, col in shapes:
        for _try in range(30):
            cx = rng.randint(4, IMG - 5)
            cy = rng.randint(4, IMG - 5)
            if all((cx - x) ** 2 + (cy - y) ** 2 > 36 for x, y in placed):
                placed.append((cx, cy))
                ci = COLORS.index(col)
                for y in range(max(0, cy - 3), min(IMG, cy + 4)):
                    for x in range(max(0, cx - 3), min(IMG, cx + 4)):
                        dx, dy = x - cx, y - cy
                        if sh == "circle" and dx * dx + dy * dy > 9:
                            continue
                        if sh == "square" and (abs(dx) > 2 or abs(dy) > 2):
                            continue
                        if sh == "triangle" and (dy > 0 or abs(dx) > (dy + 3)):
                            continue
                        img[y][x][ci] = 255
                break
    return img, target


GRID = 4
CELL = IMG // GRID


def make_image_parity(n_on, mod, rng):
    """A 4x4 grid of cells, `n_on` lit red, the rest black.

    Two properties matter:
      * the answer is NOT present as a token anywhere in the input (unlike
        the chain task, whose answer was literally one of the symbols);
      * producing it requires accumulating a running state over cells, the
        serial component attention cannot reduce in parallel.
    Parity (mod=2) is the canonical non-parallelisable boolean function.
    """
    cells = [1] * n_on + [0] * (GRID * GRID - n_on)
    rng.shuffle(cells)
    img = [[[0, 0, 0] for _ in range(IMG)] for _ in range(IMG)]
    for idx, on in enumerate(cells):
        if not on:
            continue
        gy, gx = divmod(idx, GRID)
        for y in range(gy * CELL + 1, (gy + 1) * CELL - 1):
            for x in range(gx * CELL + 1, (gx + 1) * CELL - 1):
                img[y][x][0] = 255
    return img, n_on % mod


def to_tensor(img):
    return torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0


class ResNetStem(nn.Module):
    def __init__(self, d_out, arch='resnet18', pretrained=False):
        super().__init__()
        import torchvision.models as tvm
        net = getattr(tvm, arch)(weights='DEFAULT' if pretrained else None)
        self.body = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool,
            net.layer1, net.layer2, net.layer3, net.layer4)
        c = net.fc.in_features
        self.proj = nn.Linear(c, d_out)

    def forward(self, x):
        h = self.body(x)
        B, C, H, W = h.shape
        t = h.flatten(2).transpose(1, 2)
        return self.proj(t)


class PatchEncoder(nn.Module):
    def __init__(self, d_out, in_ch=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(128, d_out, 3, stride=2, padding=1), nn.GELU(),
        )

    def forward(self, x):
        h = self.conv(x)
        return h.flatten(2).transpose(1, 2)


class Projector(nn.Module):
    def __init__(self, d, n_out=N_PATCH):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(n_out)
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, v):
        x = self.pool(v.transpose(1, 2)).transpose(1, 2)
        return self.mlp(x)


PRETRAINED_LAYERS = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--max-shapes", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--task",
                    choices=["count", "color_count", "parity", "count_mod3"],
                    default="count")
    ap.add_argument("--visual-backbone",
                    choices=["cnn", "resnet18", "resnet34"],
                    default="cnn")
    ap.add_argument("--visual-pretrained", action="store_true")
    ap.add_argument("--unfreeze-layers", type=int, default=0,
                    help="unfreeze the top N Jaume layers so the encoder can "
                         "adapt to the visual prefix (0 = fully frozen)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)
    print("[vlm] device=%s (%s)" % (device, describe(device)), flush=True)

    from jaume.modeling.config import JaumeConfig
    from jaume.modeling.embedding import JaumeEmbeddingModel
    from jaume.modeling.checkpoint import load_state_dict
    from jaume.data.dataset import load_tokenizer

    cfg = JaumeConfig.load(os.path.join(args.model, "config.json"))
    cfg.num_layers = args.depth
    tokdir = os.path.join(args.model, "tokenizer")
    if not os.path.isdir(tokdir):
        tokdir = os.path.join(REPO, "jaume_tokenizer")
    tok = load_tokenizer(tokdir)
    tok.add_tokens(["[MASK]"])
    cfg.vocab_size = len(tok)
    enc = JaumeEmbeddingModel(cfg).to(device)
    pretrained = load_state_dict(os.path.join(args.model, "pytorch_model.bin"))
    if cfg.num_layers == PRETRAINED_LAYERS:
        enc.load_state_dict(pretrained, strict=True)
    else:
        # Depth scan: keep every pretrained weight and reuse the last trained
        # block for extra layers.  Otherwise 8L fails on unexpected keys and
        # 32L fails on missing keys.
        own = enc.state_dict()
        merged, reused = {}, 0
        for k in own:
            if k in pretrained:
                merged[k] = pretrained[k]
            elif k.startswith("backbone.layers."):
                rest = k.split(".", 3)[3]
                src = "backbone.layers.%d.%s" % (PRETRAINED_LAYERS - 1, rest)
                if src in pretrained:
                    merged[k] = pretrained[src]
                    reused += 1
                else:
                    merged[k] = own[k]
            else:
                merged[k] = own[k]
        enc.load_state_dict(merged, strict=True)
        print("[vlm] depth=%d depth-upcycled %d tensors from layer %d"
              % (cfg.num_layers, reused, PRETRAINED_LAYERS - 1), flush=True)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    if args.unfreeze_layers > 0:
        # The frozen encoder cannot re-interpret a visual prefix: its layers
        # were trained on token embeddings only.  Unfreezing the top N layers
        # lets it learn to consume the projector's output, which is the same
        # design choice modern VLMs make (train projector + top blocks).
        for layer in list(enc.backbone.layers)[-args.unfreeze_layers:]:
            for p in layer.parameters():
                p.requires_grad_(True)
        for p in enc.backbone.norm.parameters():
            p.requires_grad_(True)
    d = cfg.hidden_size
    print("[vlm] jaume frozen d_text=%d" % d, flush=True)

    if args.visual_backbone == "cnn":
        vis = PatchEncoder(d).to(device)
    else:
        vis = ResNetStem(d, args.visual_backbone,
                         args.visual_pretrained).to(device)
    print("[vlm] visual backbone=%s" % args.visual_backbone, flush=True)
    proj = Projector(d).to(device)
    n_out = 3 if args.task == "count_mod3" else 16
    head = nn.Linear(d, n_out).to(device)
    print("[vlm] trainable vis=%.2fM proj=%.2fM" % (
        sum(p.numel() for p in vis.parameters()) / 1e6,
        sum(p.numel() for p in proj.parameters()) / 1e6), flush=True)

    def gen(n, base):
        out = []
        C = args.max_shapes
        for i in range(n):
            rng = random.Random(base + i)
            if args.task in ("parity", "count_mod3"):
                mod = 2 if args.task == "parity" else 3
                # balance labels exactly: class round-robin, then spread n_on
                n_on = (i // mod) % (GRID * GRID)
                n_on = n_on - (n_on % mod) + (i % mod)
                if n_on >= GRID * GRID:
                    n_on -= mod
                img, lab = make_image_parity(n_on, mod, rng)
                out.append((to_tensor(img), ("mod", mod), lab))
                continue
            target = (i % C) + 1
            q = SHAPES[i % 3]
            if args.task == "color_count":
                qc = COLORS[i % 3]
                img, cnt = make_image_color(target, q, qc, rng)
                out.append((to_tensor(img), (q, qc), cnt))
            else:
                img, cnt = make_image_exact(target, q, rng)
                out.append((to_tensor(img), q, cnt))
        return out

    print("[vlm] generating ...", flush=True)
    tr = gen(args.n_train, 1000)
    va = gen(args.n_val, 900000)
    print("[vlm] train=%d val=%d" % (len(tr), len(va)), flush=True)

    def make_q(k):
        if isinstance(k, tuple) and k[0] == "mod":
            kind = "parity" if k[1] == 2 else "count mod 3"
            return "what is the %s of the red cells ?" % kind
        if isinstance(k, tuple):
            return "how many %s %s ?" % (k[1], k[0])
        return "how many %s ?" % k
    if args.task in ("parity", "count_mod3"):
        keys = [("mod", 2 if args.task == "parity" else 3)]
    elif args.task == "color_count":
        keys = [(sh, co) for sh in SHAPES for co in COLORS]
    else:
        keys = list(SHAPES)
    q_ids = {k: torch.tensor(
        tok(make_q(k), add_special_tokens=False)["input_ids"],
        dtype=torch.long, device=device) for k in keys}


    trainable = (list(vis.parameters()) + list(proj.parameters())
                 + list(head.parameters())
                 + [p for p in enc.parameters() if p.requires_grad])
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    print("[vlm] unfreezing top %d encoder layers -> %d trainable tensors" % (
        args.unfreeze_layers, len(trainable)), flush=True)

    def step(batch):
        imgs = torch.stack([b[0] for b in batch]).to(device)
        qs = [b[1] for b in batch]
        labs = torch.tensor([b[2] for b in batch], dtype=torch.long, device=device)
        v = proj(vis(imgs))
        L = max(len(q_ids[q]) for q in qs)
        ids = torch.zeros(len(qs), L, dtype=torch.long, device=device)
        am = torch.zeros(len(qs), L, dtype=torch.long, device=device)
        for i, q in enumerate(qs):
            n = len(q_ids[q])
            ids[i, :n] = q_ids[q]
            am[i, :n] = 1
        emb = enc.backbone.embed_tokens(ids)
        x = torch.cat([v, emb], 1)
        mask = torch.cat([torch.ones(v.shape[0], v.shape[1], dtype=torch.long,
                                     device=device), am], 1)
        h = x
        for layer in enc.backbone.layers:
            h, _aux = layer(h, mask)
        h = enc.backbone.norm(h)
        pooled = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        logits = head(pooled.float())
        loss = F.cross_entropy(logits, labs)
        acc = (logits.argmax(-1) == labs).float().mean()
        return loss, acc

    for ep in range(args.epochs):
        vis.train(); proj.train(); head.train()
        perm = list(range(len(tr)))
        random.Random(ep).shuffle(perm)
        tot = acc = 0.0
        n = 0
        t0 = time.time()
        for i in range(0, len(tr) - args.batch_size + 1, args.batch_size):
            batch = [tr[j] for j in perm[i:i + args.batch_size]]
            loss, a = step(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss); acc += float(a); n += 1
        vis.eval(); proj.eval(); head.eval()
        with torch.no_grad():
            vt = va_ = 0.0
            vn = 0
            for i in range(0, len(va) - 8 + 1, 8):
                loss, a = step(va[i:i + 8])
                vt += float(loss); va_ += float(a); vn += 1
        print("[vlm] ep %d/%d train loss %.4f acc %.3f | VAL loss %.4f acc %.3f (%.0fs)" % (
            ep + 1, args.epochs, tot / max(n, 1), acc / max(n, 1),
            vt / max(vn, 1), va_ / max(vn, 1), time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
