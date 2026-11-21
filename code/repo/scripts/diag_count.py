"""Why did count/compare drop in S2b? Isolate the cause with 3 short runs.

A) vision-only   -> does counting recover at all? (interference vs capacity)
B) count_over    -> is it a data-mix problem? (4x oversample count/compare)
C) neutral gate  -> is the question gate mis-routing counting tasks?
"""
import argparse, os, sys, random, string, time
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
from jaume.device import resolve_device, seed_all
from vision_suite import build as build_vis

VIS = ["attr", "count", "spatial", "compare", "scene", "multi_attr"]
CHARS = list(string.digits) + list(string.ascii_letters) + ["$", " ", ".", "-"]
C2I = {c: i for i, c in enumerate(CHARS)}
PAD_C = len(CHARS)
N_C = len(CHARS) + 1
MAX_LEN = 8


def enc_ans(a):
    ids = [C2I[c] for c in a if c in C2I][:MAX_LEN]
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

    def forward(self, sp, op, qp, force_neutral=False):
        s = self.proj_sem(self.sem(pixel_values=sp).last_hidden_state)
        o = self.proj_ocr(self.ocr(pixel_values=op).last_hidden_state)
        g = self.gate(qp).unsqueeze(1)
        if force_neutral:
            g = torch.full_like(g, 0.5)
        L = min(s.shape[1], o.shape[1])
        sa = F.adaptive_avg_pool1d(s.transpose(1, 2), L).transpose(1, 2)
        oa = F.adaptive_avg_pool1d(o.transpose(1, 2), L).transpose(1, 2)
        return s, o, g * sa + (1 - g) * oa, g


def prep(imgs, size):
    import torchvision.transforms.functional as TF
    xs = []
    for im in imgs:
        x = TF.to_tensor(im.convert("RGB"))
        x = TF.resize(x, [size, size], antialias=True)
        xs.append((x - 0.5) / 0.5)
    return torch.stack(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--siglip-path", required=True)
    ap.add_argument("--trocr-path", required=True)
    ap.add_argument("--s2b-ckpt", required=True)
    ap.add_argument("--mode", choices=["vision", "count_over", "neutral"],
                    required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    dev = resolve_device(a.device)
    seed_all(dev, 0)
    from jaume.modeling.config import JaumeConfig
    from jaume.modeling.embedding import JaumeEmbeddingModel
    from jaume.modeling.checkpoint import load_state_dict
    from jaume.data.dataset import load_tokenizer

    cfg = JaumeConfig.load(os.path.join(a.model, "config.json"))
    tok = load_tokenizer(os.path.join(a.model, "tokenizer"))
    tok.add_tokens(["[MASK]"])
    cfg.vocab_size = len(tok)
    cfg.causal = True
    enc = JaumeEmbeddingModel(cfg).to(dev)
    pre = load_state_dict(os.path.join(a.model, "pytorch_model.bin"))
    own = enc.state_dict()
    merged = {k: (pre[k] if (k in pre and pre[k].shape == own[k].shape)
                  else own[k]) for k in own}
    enc.load_state_dict(merged, strict=True)
    for p in enc.parameters():
        p.requires_grad_(False)
    for layer in list(enc.backbone.layers)[-4:]:
        for p in layer.parameters():
            p.requires_grad_(True)
    for p in enc.backbone.norm.parameters():
        p.requires_grad_(True)

    d = cfg.hidden_size
    tower = DualTCM(d, a.siglip_path, a.trocr_path).to(dev)
    chead = nn.Linear(d, N_C).to(dev)
    ck = torch.load(a.s2b_ckpt, map_location="cpu", weights_only=False)
    sd = tower.state_dict()
    hit = {k: v for k, v in ck["tower"].items()
           if k in sd and sd[k].shape == v.shape}
    sd.update(hit)
    tower.load_state_dict(sd)
    chead.load_state_dict(ck["char_head"])
    print("[cdiag] loaded S2b (%d tensors), mode=%s" % (len(hit), a.mode),
          flush=True)

    tr = build_vis(2400, seed=11)
    va = build_vis(300, seed=777)
    if a.mode == "count_over":
        extra = [b for b in tr if b[1] in ("count", "compare")]
        tr = tr + extra * 3
    qt = {}
    for _, _, _, _, _, q in tr + va:
        if q not in qt:
            qt[q] = torch.tensor(
                tok(q, add_special_tokens=False)["input_ids"][:32],
                dtype=torch.long, device=dev)

    def run(b):
        sp = prep([x[0] for x in b], 224).to(dev)
        op = prep([x[0] for x in b], 384).to(dev)
        B = len(b)
        ql = max(len(qt[x[5]]) for x in b)
        qids = torch.zeros(B, ql, dtype=torch.long, device=dev)
        qm = torch.zeros_like(qids)
        for i, x in enumerate(b):
            n = len(qt[x[5]]); qids[i, :n] = qt[x[5]]; qm[i, :n] = 1
        e = enc.backbone.embed_tokens(qids)
        qp = (e * qm.unsqueeze(-1)).sum(1) / qm.sum(1, keepdim=True).clamp(min=1)
        s, o, fused, g = tower(sp, op, qp,
                               force_neutral=(a.mode == "neutral"))
        ans = [x[3] for x in b]
        aid = [torch.tensor(enc_ans(x), device=dev) for x in ans]
        ain = torch.full((B, MAX_LEN), PAD_C, dtype=torch.long, device=dev)
        for i, ids in enumerate(aid):
            ain[i, 1:] = ids[:-1]
        emb = enc.backbone.embed_tokens(torch.cat([qids, ain], 1))
        x = torch.cat([fused, s, o, emb], 1)
        mask = torch.cat([
            torch.ones(B, fused.shape[1], dtype=torch.long, device=dev),
            torch.ones(B, s.shape[1], dtype=torch.long, device=dev),
            torch.ones(B, o.shape[1], dtype=torch.long, device=dev), qm,
            torch.ones(B, MAX_LEN, dtype=torch.long, device=dev)], 1)
        h = x
        for L in enc.backbone.layers:
            h, _ = L(h, mask)
        h = enc.backbone.norm(h)
        base = fused.shape[1] + s.shape[1] + o.shape[1] + ql
        logits = chead(h[:, base:base + MAX_LEN, :].float())
        tgt = torch.stack(aid)
        loss = F.cross_entropy(logits.reshape(-1, N_C), tgt.reshape(-1))
        pred = logits.argmax(-1)
        ok = [all(pred[i, j].item() == tgt[i, j].item()
                  for j in range(len(enc_ans(ans[i])))) for i in range(B)]
        return loss, ok, float(g.mean())

    def ev():
        tower.eval(); chead.eval(); enc.eval()
        per = {f: [0, 0] for f in VIS}
        gs = {}
        with torch.no_grad():
            for i in range(0, len(va) - 4 + 1, 4):
                b = va[i:i + 4]
                _, ok, gm = run(b)
                for j, o_ in enumerate(ok):
                    per[b[j][1]][1] += 1
                    per[b[j][1]][0] += int(o_)
                    gs.setdefault(b[j][1], []).append(gm)
        return per, {k: sum(v) / len(v) for k, v in gs.items()}

    ps = (list(tower.proj_sem.parameters()) + list(tower.proj_ocr.parameters())
          + list(tower.gate.parameters()) + list(chead.parameters()))
    ep_ = [p for p in enc.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": ps, "lr": 5e-4},
                             {"params": ep_, "lr": 5e-5}], weight_decay=0.01)
    allp = ps + ep_

    for e in range(a.epochs):
        tower.train(); chead.train(); enc.train()
        perm = list(range(len(tr)))
        random.Random(e).shuffle(perm)
        tot = nb = 0
        t0 = time.time()
        for i in range(0, len(tr) - a.batch_size + 1, a.batch_size):
            b = [tr[j] for j in perm[i:i + a.batch_size]]
            loss, _, _ = run(b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(allp, 1.0)
            opt.step()
            tot += float(loss); nb += 1
        per, gs = ev()
        vm = sum(per[f][0] / max(per[f][1], 1) for f in VIS) / 6
        print("[cdiag] %s ep%d loss %.3f | VIS %.3f | count %.2f compare %.2f"
              " | gate(count) %.2f (%.0fs)"
              % (a.mode, e + 1, tot / max(nb, 1), vm,
                 per["count"][0] / max(per["count"][1], 1),
                 per["compare"][0] / max(per["compare"][1], 1),
                 gs.get("count", 0.5), time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
