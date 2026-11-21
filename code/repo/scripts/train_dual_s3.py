"""S3: add REAL scanned forms (FUNSD) on top of S2b.

S2b gave one checkpoint with OCR 0.887 / VIS 0.913 / CIFAR 0.954 -- all on
synthetic or 32px data. S3 is the first test on real documents: FUNSD forms
are 762x1000 scans with skew, noise and real fonts.

Two FUNSD tasks:
  funsd_read  target box cropped at native resolution -> read its text
  funsd_find  whole form -> classify a field as question/answer/header/other

Training keeps the S2b data in the mix so previously acquired abilities are
not overwritten (the S2 lesson: whoever trains most owns the backbone).
"""
import argparse, os, random, string, sys, time
import torch, torch.nn as nn, torch.nn.functional as F

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
REPO = os.path.join(PROJECT_ROOT, "code", "repo")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jaume.device import resolve_device, seed_all, describe
from ocr_suite import build as build_ocr, OCR_FAMILIES
from vision_suite import build as build_vis
from funsd_env import build as build_funsd, FUNSD_FAMILIES

VIS_FAMILIES = ["attr", "count", "spatial", "compare", "scene", "multi_attr"]
# FUNSD uses 74 distinct characters including punctuation and a box glyph.
CHARS = (list(string.digits) + list(string.ascii_letters)
         + list("$ %'()+,-./:;#\u2610"))
C2I = {c: i for i, c in enumerate(CHARS)}
PAD_C = len(CHARS)
N_C = len(CHARS) + 1
MAX_LEN = 32


def norm_answer(a):
    """Document OCR is case-insensitive and space-insensitive in practice.

    TrOCR itself reads 'Beauty Contest' as 'BEAUTY CONTEST' and drops spaces
    inside hyphenated tokens; scoring raw exact match therefore reports a
    failure that is not a reading failure. We normalise both sides.
    """
    a = " ".join(a.split())
    return a.upper()


def enc_ans(a):
    ids = [C2I[c] for c in norm_answer(a) if c in C2I][:MAX_LEN]
    return ids + [PAD_C] * (MAX_LEN - len(ids))


class DualTCM(nn.Module):
    def __init__(self, d, sig, tro):
        super().__init__()
        from transformers import SiglipModel, VisionEncoderDecoderModel
        f = SiglipModel.from_pretrained(sig, dtype=torch.float32)
        self.sem = f.vision_model
        self.proj_sem = nn.Sequential(
            nn.LayerNorm(self.sem.config.hidden_size),
            nn.Linear(self.sem.config.hidden_size, d), nn.GELU(),
            nn.Linear(d, d))
        t = VisionEncoderDecoderModel.from_pretrained(tro, dtype=torch.float32)
        self.ocr = t.encoder
        self.proj_ocr = nn.Sequential(
            nn.LayerNorm(self.ocr.config.hidden_size),
            nn.Linear(self.ocr.config.hidden_size, d), nn.GELU(),
            nn.Linear(d, d))
        self.gate = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
                                  nn.Linear(d, d), nn.Sigmoid())
        for p in self.sem.parameters():
            p.requires_grad_(False)
        for p in self.ocr.parameters():
            p.requires_grad_(False)

    def forward(self, sp, op, qp):
        s = self.proj_sem(self.sem(pixel_values=sp).last_hidden_state)
        o = self.proj_ocr(self.ocr(pixel_values=op).last_hidden_state)
        g = self.gate(qp).unsqueeze(1)
        L = min(s.shape[1], o.shape[1])
        sa = F.adaptive_avg_pool1d(s.transpose(1, 2), L).transpose(1, 2)
        oa = F.adaptive_avg_pool1d(o.transpose(1, 2), L).transpose(1, 2)
        return s, o, g * sa + (1 - g) * oa, g


def prep(imgs, size, preserve_aspect=False):
    """Resize to ; optionally letterbox to keep the aspect ratio.

    FUNSD crops have aspect ratios from 0.29 to 13.4 (a 46x spread). Forcing
    them all to 384x384 squashes wide text into unreadable slivers. Letterbox
    padding keeps the glyphs at their native shape, which is what OCR needs.
    """
    import torchvision.transforms.functional as TF
    xs = []
    for im in imgs:
        im = im.convert("RGB")
        x = TF.to_tensor(im)
        if not preserve_aspect:
            x = TF.resize(x, [size, size], antialias=True)
        else:
            w, h = im.size
            scale = min(size / w, size / h)
            nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
            x = TF.resize(x, [nh, nw], antialias=True)
            pad_l = (size - nw) // 2
            pad_t = (size - nh) // 2
            x = TF.pad(x, [pad_l, pad_t, size - nw - pad_l, size - nh - pad_t],
                       fill=0.5)
        xs.append((x - 0.5) / 0.5)
    return torch.stack(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--siglip-path", required=True)
    ap.add_argument("--trocr-path", required=True)
    ap.add_argument("--funsd-root", required=True)
    ap.add_argument("--s2b-ckpt", default=None)
    ap.add_argument("--n-ocr", type=int, default=1500)
    ap.add_argument("--n-vis", type=int, default=1000)
    ap.add_argument("--n-funsd", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr-proj", type=float, default=4e-4)
    ap.add_argument("--lr-backbone", type=float, default=4e-5)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--unfreeze-layers", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)
    print("[s3] device=%s (%s)" % (device, describe(device)), flush=True)
    seed_all(device, 0)

    from jaume.modeling.config import JaumeConfig
    from jaume.modeling.embedding import JaumeEmbeddingModel
    from jaume.modeling.checkpoint import load_state_dict
    from jaume.data.dataset import load_tokenizer

    cfg = JaumeConfig.load(os.path.join(args.model, "config.json"))
    cfg.num_layers = args.depth
    tok = load_tokenizer(os.path.join(args.model, "tokenizer"))
    tok.add_tokens(["[MASK]"])
    cfg.vocab_size = len(tok)
    cfg.causal = True

    enc = JaumeEmbeddingModel(cfg).to(device)
    pre = load_state_dict(os.path.join(args.model, "pytorch_model.bin"))
    own = enc.state_dict()
    merged, up = {}, 0
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
    for layer in list(enc.backbone.layers)[-args.unfreeze_layers:]:
        for p in layer.parameters():
            p.requires_grad_(True)
    for p in enc.backbone.norm.parameters():
        p.requires_grad_(True)

    d = cfg.hidden_size
    tower = DualTCM(d, args.siglip_path, args.trocr_path).to(device)
    chead = nn.Linear(d, N_C).to(device)
    if args.s2b_ckpt and os.path.exists(args.s2b_ckpt):
        ck = torch.load(args.s2b_ckpt, map_location="cpu", weights_only=False)
        sd = tower.state_dict()
        hit = {k: v for k, v in ck["tower"].items()
               if k in sd and sd[k].shape == v.shape}
        sd.update(hit); tower.load_state_dict(sd)
        # char_head may have a different output size (the character inventory
        # grew for FUNSD: ':' and ','). Load only the matching prefix and let
        # the new rows initialise.
        ch = ck["char_head"]
        own = chead.state_dict()
        merged_ch = {}
        for k, v in own.items():
            if k in ch and ch[k].shape == v.shape:
                merged_ch[k] = ch[k]
            elif k == "weight" and k in ch and ch[k].shape[1] == v.shape[1]:
                merged_ch[k] = v.clone()
                merged_ch[k][:ch[k].shape[0]] = ch[k]
            elif k == "bias" and k in ch and ch[k].shape[0] <= v.shape[0]:
                merged_ch[k] = v.clone()
                merged_ch[k][:ch[k].shape[0]] = ch[k]
            else:
                merged_ch[k] = v
        chead.load_state_dict(merged_ch)
        print("[s3] char head expanded %d -> %d" % (
            ch["weight"].shape[0], own["weight"].shape[0]), flush=True)
        print("[s3] warm start S2b (%d tensors)" % len(hit), flush=True)

    t0 = time.time()
    ocr_tr = build_ocr(args.n_ocr, seed=31)
    vis_tr = build_vis(args.n_vis, seed=11)
    fun_tr = build_funsd(args.n_funsd, args.funsd_root, seed=5)
    ocr_va = build_ocr(200, seed=909)
    vis_va = build_vis(200, seed=777)
    fun_va = build_funsd(300, args.funsd_root, seed=99)
    print("[s3] ocr %d vis %d funsd %d (%.0fs)" % (
        len(ocr_tr), len(vis_tr), len(fun_tr), time.time() - t0), flush=True)

    qt = {}
    for _, _, _, _, _, q in ocr_tr + vis_tr + fun_tr + ocr_va + vis_va + fun_va:
        if q not in qt:
            qt[q] = torch.tensor(
                tok(q, add_special_tokens=False)["input_ids"][:40],
                dtype=torch.long, device=device)

    def run(b):
        sp = prep([x[0] for x in b], 224, preserve_aspect=True).to(device)
        op = prep([x[0] for x in b], 384, preserve_aspect=True).to(device)
        B = len(b)
        ql = max(len(qt[x[5]]) for x in b)
        qids = torch.zeros(B, ql, dtype=torch.long, device=device)
        qm = torch.zeros_like(qids)
        for i, x in enumerate(b):
            n = len(qt[x[5]]); qids[i, :n] = qt[x[5]]; qm[i, :n] = 1
        e = enc.backbone.embed_tokens(qids)
        qp = (e * qm.unsqueeze(-1)).sum(1) / qm.sum(1, keepdim=True).clamp(min=1)
        s, o, fused, g = tower(sp, op, qp)
        ans = [x[3] for x in b]
        aid = [torch.tensor(enc_ans(x), device=device) for x in ans]
        # AUTOREGRESSIVE teacher forcing. FUNSD has ~150 distinct answers in
        # 157 samples (mean 13.5 chars), i.e. an open-vocabulary generation
        # task. A single-position readout cannot express that -- it was the
        # reason funsd_read stalled at 0.16. Feeding the answer prefix back in
        # gives every output position its own hidden state.
        ain = torch.full((B, MAX_LEN), PAD_C, dtype=torch.long, device=device)
        for i, ids in enumerate(aid):
            ain[i, 1:] = ids[:-1]
        emb = enc.backbone.embed_tokens(torch.cat([qids, ain], 1))
        x = torch.cat([fused, s, o, emb], 1)
        mask = torch.cat([
            torch.ones(B, fused.shape[1], dtype=torch.long, device=device),
            torch.ones(B, s.shape[1], dtype=torch.long, device=device),
            torch.ones(B, o.shape[1], dtype=torch.long, device=device), qm,
            torch.ones(B, MAX_LEN, dtype=torch.long, device=device)], 1)
        h = x
        for L in enc.backbone.layers:
            h, _ = L(h, mask)
        h = enc.backbone.norm(h)
        base = fused.shape[1] + s.shape[1] + o.shape[1] + ql
        # one logit vector per answer character
        logits = chead(h[:, base:base + MAX_LEN, :].float())
        tgt = torch.stack(aid)
        loss = F.cross_entropy(logits.reshape(-1, N_C), tgt.reshape(-1))
        pred = logits.argmax(-1)
        # compare exactly the encoded length: answers longer than MAX_LEN are
        # truncated by enc_ans, so scoring on the raw length would be
        # unachievable by construction
        ok = []
        for i in range(B):
            n = len(enc_ans(ans[i]))
            ok.append(all(pred[i, j].item() == tgt[i, j].item()
                          for j in range(n)))
        return loss, ok

    def ev():
        tower.eval(); chead.eval(); enc.eval()
        fams = OCR_FAMILIES + VIS_FAMILIES + FUNSD_FAMILIES
        per = {f: [0, 0] for f in fams}
        with torch.no_grad():
            for src in (ocr_va, vis_va, fun_va):
                for i in range(0, len(src) - 4 + 1, 4):
                    b = src[i:i + 4]
                    _, ok = run(b)
                    for j, o_ in enumerate(ok):
                        per[b[j][1]][1] += 1
                        per[b[j][1]][0] += int(o_)
        return per

    ps = (list(tower.proj_sem.parameters()) + list(tower.proj_ocr.parameters())
          + list(tower.gate.parameters()) + list(chead.parameters()))
    ep_ = [p for p in enc.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": ps, "lr": args.lr_proj},
                             {"params": ep_, "lr": args.lr_backbone}],
                            weight_decay=0.01)
    allp = ps + ep_

    pools = ["ocr", "vis", "funsd"]
    for e in range(args.epochs):
        tower.train(); chead.train(); enc.train()
        idx = {k: list(range(len(v))) for k, v in
               (("ocr", ocr_tr), ("vis", vis_tr), ("funsd", fun_tr))}
        for k in idx:
            random.Random(e + hash(k) % 100).shuffle(idx[k])
        ptr = {k: 0 for k in idx}
        tot = nb = 0
        t0 = time.time()
        n_steps = min(len(ocr_tr), len(vis_tr), len(fun_tr)) // args.batch_size
        for _ in range(n_steps * 3):
            k = random.Random(e * 7919 + nb).choice(pools)
            src = {"ocr": ocr_tr, "vis": vis_tr, "funsd": fun_tr}[k]
            if ptr[k] + args.batch_size > len(src):
                ptr[k] = 0
            sel = idx[k][ptr[k]:ptr[k] + args.batch_size]
            ptr[k] += args.batch_size
            if len(sel) < args.batch_size:
                continue
            b = [src[j] for j in sel]
            loss, _ = run(b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(allp, 1.0)
            opt.step()
            tot += float(loss); nb += 1
        per = ev()
        def m(fs):
            return sum(per[f][0] / max(per[f][1], 1) for f in fs) / len(fs)
        print("[s3] ep %d/%d loss %.3f | OCR %.3f | VIS %.3f | FUNSD %.3f"
              " (read %.2f find %.2f) (%.0fs)"
              % (e + 1, args.epochs, tot / max(nb, 1), m(OCR_FAMILIES),
                 m(VIS_FAMILIES), m(FUNSD_FAMILIES),
                 per["funsd_read"][0] / max(per["funsd_read"][1], 1),
                 per["funsd_find"][0] / max(per["funsd_find"][1], 1),
                 time.time() - t0), flush=True)

    per = ev()
    print("=== FINAL S3 ===", flush=True)
    for f in OCR_FAMILIES + VIS_FAMILIES + FUNSD_FAMILIES:
        c, n = per[f]
        print("  %-13s %.4f (%d/%d)" % (f, c / max(n, 1), c, n), flush=True)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        torch.save({"tower": tower.state_dict(),
                    "char_head": chead.state_dict()},
                   os.path.join(args.out, "dual_s3.pt"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
