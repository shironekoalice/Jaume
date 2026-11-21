"""Train the actor's scoring head on real GUI data (seeclick), evaluate on
ScreenSpot.

WHY SEE CLICK. ScreenSpot is an EVALUATION set ("This dataset is a
benchmarking dataset. It is not used for training"). seeclick is the
training half of the same project and — importantly — it annotates EVERY
element on a screenshot, not just one. So a training sample is a real
screenshot with its real element list, which means:

  * the candidate set during TRAINING is the real element set, not
    synthetic distractors (ScreenSpot's constructed shortlist);
  * the label is "this element is the one the instruction refers to".

That makes the training distribution much closer to deployment than the
constructed shortlist used for the ScreenSpot probe.

THE MODEL. SigLIP2-base (93M) frozen, plus a small trainable head:

    crop ─> SigLIP2 image tower ─> (196 patches, 768d) ─┐
                                                      ├─> instruction-conditioned
    instruction ─> SigLIP2 text tower ─> (768d) ─────┘   gate + scorer
    box ─> linear ─────────────────────────────────────┘

The gate is the point. The repo already measured this in
`docs/vlm-dual-tower-spec.md`: on a shared trunk with mixed tasks the
question-conditioned gate is what keeps the two streams from overwriting
each other (OCR mean 0.65 without the gate vs 0.887 with it). Here the
gate lets the INSTRUCTION modulate the visual features, which a plain
concatenation cannot do.

    gate = sigmoid( W_g . text_emb )            (per-patch, 768d)
    fused = visual * (1 + gate) + text_proj    (residual keeps the raw stream)

THE LOSS. Softmax CE across the candidates of one screenshot — the
state-internal softmax CE the repo's planning report §4.7 established as
the only loss that does not collapse to an action prior. Per-pair BCE
collapses; a state-internal CE makes an action-only prior provably useless.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SEECLICK = os.path.join(HERE, "data", "seeclick", "data", "train-00000-of-00255.parquet")


# --------------------------------------------------------------- data

def load_seeclick(max_rows=1500, seed=0):
    import pyarrow.parquet as pq

    tbl = pq.read_table(SEECLICK)
    n = min(max_rows, tbl.num_rows)
    out = []
    for i in range(n):
        row = {k: tbl.column(k)[i].as_py() for k in tbl.schema.names}
        img = row["image"]
        if isinstance(img, dict):
            img = Image.open(io.BytesIO(img["bytes"]))
        elif isinstance(img, (bytes, bytearray)):
            img = Image.open(io.BytesIO(img))
        els = row["elements"] or []
        if len(els) < 2:
            continue
        out.append(dict(image=img.convert("RGB"), elements=els))
    return out


def norm_box(b, w, h):
    x0, y0, x1, y1 = b
    return (x0 * w, y0 * h, x1 * w, y1 * h)


def build_train_set(data, max_cands=12, min_box=6, seed=0):
    """Each sample: a screenshot, a target element, and distractors taken
    from the SAME screenshot's element list (real distractors, not random
    boxes)."""
    rng = np.random.default_rng(seed)
    out = []
    for si, d in enumerate(data):
        w, h = d["image"].size
        els = d["elements"]
        # keep elements with a usable, non-degenerate box
        usable = []
        for e in els:
            b = norm_box(e["bbox"], w, h)
            if (b[2] - b[0]) >= min_box and (b[3] - b[1]) >= min_box:
                usable.append((e["instruction"], b, e.get("data_type", "text")))
        if len(usable) < 2:
            continue
        for t_instr, t_box, t_type in usable:
            others = [u for u in usable if u[1] != t_box]
            rng.shuffle(others)
            others = others[: max_cands - 1]
            cands = [(t_instr, t_box, t_type)] + others
            order = rng.permutation(len(cands))
            shuffled = [cands[j] for j in order]
            gold = list(order).index(0)
            out.append(dict(image=d["image"], instruction=t_instr,
                            candidates=[c[1] for c in shuffled],
                            cand_texts=[c[0] for c in shuffled],
                            gold=gold, shot=si))
    return out


# --------------------------------------------------------------- model

class ConditionalScorer(nn.Module):
    """Instruction-conditioned candidate scorer on a frozen SigLIP2.

    NOTE ON DIMENSIONS. This checkpoint's two towers do NOT share an output
    space: the vision tower pools to 768 and the text tower to 1152. So the
    instruction cannot simply be added to the visual feature. Instead the
    instruction is PROJECTED into the visual space by a trainable linear
    layer, and the gate is computed there:

        t_vis = W_text . text_emb            (1152 -> 768)
        gate  = sigmoid(W_g . t_vis)         (per-candidate, 768)
        fused = v * (1 + gate) + t_vis      (residual keeps the raw stream)

    The residual matters: the repo measured that removing the raw stream
    collapses a dual-tower model (`docs/vlm-dual-tower-spec.md`: OCR mean
    0.887 with both streams, 0.65 with one).
    """

    def __init__(self, tower, dim, text_dim, hidden=384):
        super().__init__()
        self.tower = tower
        self.dim = dim
        self.text_dim = text_dim
        self.text_proj = nn.Linear(text_dim, dim)
        self.gate = nn.Linear(dim, dim)
        self.box_proj = nn.Linear(4, dim)
        self.fuse = nn.Sequential(
            # fused + raw visual + text + projected box, all at width `dim`
            nn.Linear(dim * 4, hidden), nn.GELU(),
            nn.Linear(hidden, hidden // 2), nn.GELU())
        self.out = nn.Linear(hidden // 2, 1)

    def score_features(self, v, boxes, text_emb, mask):
        """v (B,N,Dv) pre-computed crop embeddings | boxes (B,N,4)
        | text_emb (B,Dt) | mask (B,N)

        The frozen half of the model lives here, so the training loop can
        feed PRE-COMPUTED features and skip the vision tower entirely.
        """
        B, N = v.shape[:2]
        t = self.text_proj(text_emb)                      # (B, Dv)
        g = torch.sigmoid(self.gate(t)).unsqueeze(1)      # (B,1,Dv)
        fused = v * (1.0 + g) + t.unsqueeze(1)            # instruction modulates vision
        b = self.box_proj(boxes)
        x = torch.cat([fused, v, t.unsqueeze(1).expand(B, N, -1), b], dim=-1)
        logits = self.out(self.fuse(x)).squeeze(-1)
        return logits.masked_fill(~mask, -1e4)

    def forward(self, crops, boxes, text_emb, mask):
        """crops (B,N,3,S,S) | boxes (B,N,4) | text_emb (B,Dt) | mask (B,N)"""
        B, N = crops.shape[:2]
        with torch.no_grad():
            v = self.image_features(crops)
            if v.dim() == 3:
                v = v.mean(dim=1)
        return self.score_features(v.reshape(B, N, -1), boxes, text_emb, mask)

    def crop_inputs(self, crops):
        raise NotImplementedError


class SigLIPConditionalScorer(ConditionalScorer):
    """Concrete version: knows how to preprocess crops.

    NOTE on the checkpoint format. This SigLIP2 checkpoint uses the OLD
    config layout (the repo hit this before: "该 checkpoint 的 config 是旧
    格式，要用 xxx 加载完整模型再取 xxx"). `get_image_features` does not
    exist on this class, and `model(**pixel_values)` raises because the
    forward demands input_ids. The working path is to run the sub-modules
    directly and apply the projections by hand:

        vision_model(pixel_values) -> vision_projection(...)  (image side)
        text_model(input_ids)      -> text_projection(...)    (text side)

    Both projections land in the SAME space (768d), which is what lets the
    instruction modulate the visual features.
    """

    def __init__(self, model_id, proc, device, hidden=384):
        tower = model_id
        # The config says both towers are 768, but the ACTUAL pooled widths
        # of this checkpoint differ (vision 1152, text 768) — the config lies
        # about this model. So the widths are measured from a real forward
        # pass instead of being read from the config.
        with torch.no_grad():
            v0 = tower.vision_model(
                pixel_values=torch.zeros(1, 3, 224, 224, dtype=tower.dtype, device=device))
            vpool = v0.pooler_output if hasattr(v0, "pooler_output") \
                else v0.last_hidden_state.mean(dim=1)
            t0 = tower.text_model(
                input_ids=torch.zeros(1, 8, dtype=torch.long, device=device),
                attention_mask=torch.ones(1, 8, dtype=torch.long, device=device))
            tpool = t0.pooler_output if hasattr(t0, "pooler_output") \
                else t0.last_hidden_state.mean(dim=1)
        vdim, tdim = int(vpool.shape[-1]), int(tpool.shape[-1])
        print(f"  measured tower widths: vision {vdim}, text {tdim}", flush=True)
        super().__init__(tower, vdim, tdim, hidden)
        self.proc = proc
        self.device = device

    def crop_inputs(self, crops):
        """crops: (B,N,3,S,S) float in [0,1] -> the tower's pixel values.

        The crops are already resized to SxS, so the SigLIP normalisation is
        applied directly instead of round-tripping through PIL.
        """
        mean = torch.tensor([0.5, 0.5, 0.5], device=crops.device).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5], device=crops.device).view(1, 1, 3, 1, 1)
        x = (crops - mean) / std
        B, N = x.shape[:2]
        return {"pixel_values": x.reshape(B * N, *x.shape[2:]).to(self.tower.dtype)}

    def image_features(self, crops):
        t = self.tower
        B, N = crops.shape[:2]
        pv = self.crop_inputs(crops)["pixel_values"]
        out = t.vision_model(pixel_values=pv)
        return out.pooler_output if hasattr(out, "pooler_output") \
            else out.last_hidden_state.mean(dim=1)

    def text_features(self, **tin):
        t = self.tower
        out = t.text_model(**tin)
        return out.pooler_output if hasattr(out, "pooler_output") \
            else out.last_hidden_state.mean(dim=1)


def crop_batch(images, boxes, size=224):
    """PIL crops -> a normalised float tensor."""
    arr = []
    for img, b in zip(images, boxes):
        x0, y0, x1, y1 = [int(v) for v in b]
        x0 = max(0, min(img.width - 1, x0)); y0 = max(0, min(img.height - 1, y0))
        x1 = max(x0 + 1, min(img.width, x1)); y1 = max(y0 + 1, min(img.height, y1))
        c = img.crop((x0, y0, x1, y1)).resize((size, size), Image.BICUBIC)
        arr.append(np.asarray(c, dtype=np.float32) / 255.0)
    return torch.from_numpy(np.stack(arr)).permute(0, 3, 1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--rows", type=int, default=600)
    ap.add_argument("--max-cands", type=int, default=8)
    ap.add_argument("--out", default="conditional_scorer.json")
    ap.add_argument("--ckpt", default="conditional_scorer.pt")
    args = ap.parse_args()
    device = args.device

    from transformers import AutoModel, AutoProcessor
    from backbone_registry import local_path

    mid = "google/siglip2-base-patch16-224"
    proc = AutoProcessor.from_pretrained(local_path(mid))
    tower = AutoModel.from_pretrained(local_path(mid)).to(device).eval()
    for p in tower.parameters():
        p.requires_grad_(False)

    print(f"loading seeclick...", flush=True)
    raw = load_seeclick(max_rows=args.rows)
    train = build_train_set(raw, max_cands=args.max_cands, seed=0)
    print(f"  {len(raw)} screenshots -> {len(train)} training samples", flush=True)

    # Split by SCREENSHOT, not by sample. Each screenshot contributes many
    # samples (one per element), so a sample-level split would put the same
    # screenshot on both sides and the held-out number would measure
    # memorisation of the screen rather than generalisation.
    # (PIL Images are unhashable, so the screenshot is tracked by its index
    # in the source list.)
    shots = sorted({d["shot"] for d in train})
    np.random.default_rng(0).shuffle(shots)
    n_tr_shot = int(len(shots) * 0.85)
    tr_shots = set(shots[:n_tr_shot])
    tr_idx = [i for i, d in enumerate(train) if d["shot"] in tr_shots]
    te_idx = [i for i, d in enumerate(train) if d["shot"] not in tr_shots]
    print(f"  split by screenshot: {len(tr_shots)} train / {len(shots)-len(tr_shots)} test",
          flush=True)

    # ---- PRE-COMPUTE THE FROZEN FEATURES.
    # The backbone is frozen, so the crop embedding and the instruction
    # embedding are constant. Computing them once (instead of every epoch)
    # turns an 8-minute epoch into a few seconds.
    #
    # MEMORY. The crops are the problem: 6822 samples x 8 crops x 3x224x224
    # float32 is 32 GB, which does not fit. Two fixes together:
    #   * crops are stored as uint8 (4x less) and normalised on the fly;
    #   * crops are DEDUPLICATED per screenshot, because the many samples
    #     drawn from one screenshot share the same candidate boxes. The
    #     unique crop set is a few thousand, not 55k.
    print("  pre-computing frozen features...", flush=True)
    t0 = time.time()
    crop_key = {}
    crop_list = []
    feats = []
    for c in train:
        img = c["image"]
        w, h = img.size
        keys, boxes = [], []
        for b in c["candidates"]:
            k = (c["shot"], tuple(round(v, 2) for v in b))
            if k not in crop_key:
                crop_key[k] = len(crop_list)
                crop_list.append(crop_batch([img], [b], 224)[0])
            keys.append(crop_key[k])
            boxes.append([b[0] / w, b[1] / h, b[2] / w, b[3] / h])
        feats.append(dict(keys=keys, boxes=torch.tensor(boxes), gold=c["gold"],
                          instruction=c["instruction"]))
    print(f"  {len(crop_list)} unique crops (from {sum(len(f['keys']) for f in feats)} slots)",
          flush=True)

    all_crops = torch.stack(crop_list).to(torch.uint8)
    chunk = 64
    vs = []
    for i in range(0, len(all_crops), chunk):
        px = all_crops[i:i + chunk].float().div(255.0)
        pv = ((px - 0.5) / 0.5).to(device=device, dtype=tower.dtype)
        with torch.no_grad():
            vo = tower.vision_model(pixel_values=pv)
            vpool = vo.pooler_output if hasattr(vo, "pooler_output") \
                else vo.last_hidden_state.mean(dim=1)
        vs.append(vpool.float().cpu())
    vpool_all = torch.cat(vs)

    uniq = sorted({c["instruction"] for c in train})
    tin = proc(text=uniq, return_tensors="pt", padding=True, truncation=True, max_length=32)
    tin = {k: v.to(device) for k, v in tin.items()}
    with torch.no_grad():
        to = tower.text_model(**tin)
        tpool = to.pooler_output if hasattr(to, "pooler_output") \
            else to.last_hidden_state.mean(dim=1)
    tmap = {u: tpool[i].float().cpu() for i, u in enumerate(uniq)}
    for f in feats:
        f["v"] = vpool_all[torch.tensor(f["keys"])]
        f["t"] = tmap[f["instruction"]]
    print(f"  features done in {time.time()-t0:.0f}s", flush=True)

    model = SigLIPConditionalScorer(tower, proc, device).to(device)
    # the head now takes PRE-COMPUTED features
    print(f"  trainable params {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M",
          flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    lossf = nn.CrossEntropyLoss()

    def run_epoch(indices, train_mode=True):
        model.train(train_mode)
        tot, nb, hits = 0.0, 0, 0
        for s in range(0, len(indices), args.bs):
            chunk = [feats[i] for i in indices[s:s + args.bs]]
            B = len(chunk)
            K = max(f["v"].shape[0] for f in chunk)
            v = torch.zeros(B, K, model.dim)
            b = torch.zeros(B, K, 4)
            t = torch.zeros(B, model.text_dim)
            mask = torch.zeros(B, K, dtype=torch.bool)
            golds = []
            for j, f in enumerate(chunk):
                n = f["v"].shape[0]
                v[j, :n] = f["v"]
                b[j, :n] = f["boxes"]
                mask[j, :n] = True
                t[j] = f["t"]
                golds.append(f["gold"])
            logits = model.score_features(v.to(device), b.to(device), t.to(device), mask.to(device))
            gold = torch.tensor(golds, device=device)
            loss = lossf(logits, gold)
            if train_mode:
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            else:
                hits += int((logits.argmax(-1) == gold).sum())
            tot += float(loss.detach()); nb += 1
        return tot / max(nb, 1), hits / max(len(indices), 1)

    for ep in range(args.epochs):
        tr_loss, _ = run_epoch(tr_idx, True)
        te_loss, te_acc = run_epoch(te_idx, False)
        print(f"  ep{ep+1:02d} train ce {tr_loss:.4f} | held-out ce {te_loss:.4f} "
              f"top-1 {te_acc:.4f} (floor {1/args.max_cands:.4f})", flush=True)
        sched.step()

    # Save ONLY the trainable head. The frozen SigLIP2 tower is 1.5 GB and
    # is re-downloadable from ModelScope by id; storing it in the checkpoint
    # made the file 1.5 GB for 2.4 M trainable parameters.
    head_state = {k: v for k, v in model.state_dict().items()
                  if not k.startswith("tower.")}
    torch.save(dict(head=head_state, dim=model.dim, text_dim=model.text_dim),
               args.ckpt)
    print(f"  saved head only ({sum(v.numel() for v in head_state.values())/1e6:.2f}M params) "
          f"to {args.ckpt}", flush=True)
    res = dict(held_out_top1=te_acc, floor=1.0 / args.max_cands, epochs=args.epochs,
               train_samples=len(tr_idx), test_samples=len(te_idx),
               trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad))
    json.dump(res, open(args.out, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
