"""Gate 1: can a per-input selector find the path signal that Gate 0 proved exists?

Gate 0 result (3 seeds, SciFact, 910B3):

    A  single best fixed path     0.3503  sd 0.0041
    B  top-8 ensemble (CV)        0.3448  sd 0.0130
    C  all-40 ensemble            0.3553  sd 0.0140      C-A = +0.0049
    D  oracle (labels)            0.7113  sd 0.0079      coverage 0.751

C-A fell inside the seed noise, so no input-blind combination extracts the
signal. But the oracle reaches 0.7113 over 75.1% coverage, which means correct
paths exist for three quarters of the rows. Gate 1 is the only remaining
question: can a selector that reads the INPUT (never the label) find them?

Design
------
  probe      mean-pooled hidden states after the first `--probe-layers`
             layers. Those layers run once no matter which path is chosen,
             so the probe is nearly free.
  experts    each candidate path is an expert that emits a 3-way softmax.
  selector   MLP(probe) -> P logits over the experts.
  training   cross-entropy toward the experts that were CORRECT on that train
             row (labels are allowed while training), rows with no correct
             expert are skipped.

Reported conditions, all on the same held-out test split:

  G1-argmax  selector picks one expert per input          (1x path cost)
  G1-soft    selector weights all experts by softmax      (P x path cost)
  G1-rand    IDENTICAL selector trained on RANDOM targets -- the control
  A / C / D  carried over from Gate 0 for reference

The G1-rand control matters: a selector has enough capacity to memorise
noise, so a gain over A only counts if it also beats a same-shaped selector
trained on shuffled targets. Without it the JM failure repeats -- a gate that
looks trained but carries no input-specific information.

Degeneracy checks (docs/memory-report.md section 6.3: the JM gate collapsed to
a constant while its weights moved 141%):

  * entropy of the selector distribution across inputs
  * fraction of inputs whose argmax differs from the modal expert
  Both are printed. A selector that always picks one expert has gained
  nothing beyond A no matter what its test F1 says.

Usage
-----
    python scripts/gate1_selector.py --model <ckpt> --seed 0 1 2 --k 8
"""

import argparse
import json
import os
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
from jaume.modeling.checkpoint import load_state_dict

from gate0_paths import (load_jsonl, macro_f1, build_candidates,
                         forward_with_path, encode_path, fit_head, predict)


def probe_features(backbone, tokenizer, texts, n_layers, device, max_length, bs=32):
    """Mean-pooled hidden states after the first `n_layers` layers."""
    feats = []
    for i in range(0, len(texts), bs):
        enc = tokenizer(texts[i:i + bs], padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt").to(device)
        h = backbone.embed_tokens(enc["input_ids"])
        for li in range(n_layers):
            h, _ = backbone.layers[li](h, enc["attention_mask"])
        m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        feats.append(pooled.float().cpu())
    return torch.cat(feats, 0)


class Selector(nn.Module):
    def __init__(self, in_dim, n_experts, hidden=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_experts),
        )

    def forward(self, x):
        return self.net(x)


def train_selector(S, Xtr, targets, weights, Xva, device, seed,
                   epochs, lr, bs, weight_decay):
    """Train on soft targets; `weights` is 0 for rows with no correct expert."""
    seed_all(device, seed)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(S.parameters(), lr=lr, weight_decay=weight_decay)
    Xtr_d, T_d, W_d = Xtr.to(device), targets.to(device), weights.to(device)
    hist = []
    for ep in range(epochs):
        S.train()
        perm = torch.randperm(len(Xtr_d), device=device)
        tot, n = 0.0, 0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            logits = S(Xtr_d[idx])
            logp = torch.log_softmax(logits, -1)
            # weighted cross-entropy against the soft target
            loss_i = -(T_d[idx] * logp).sum(-1)
            loss = (loss_i * W_d[idx]).sum() / W_d[idx].sum().clamp(min=1e-9)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            n += 1
        hist.append(tot / max(n, 1))
    return hist


@torch.no_grad()
def selector_probs(S, X, device, bs=512):
    S.eval()
    outs = []
    Xd = X.to(device)
    for i in range(0, len(Xd), bs):
        outs.append(torch.softmax(S(Xd[i:i + bs]).float(), -1).cpu())
    return torch.cat(outs, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=os.path.join(PROJECT_ROOT, "data", "scifact_blocks"))
    ap.add_argument("--seed", type=int, nargs="+", default=[0])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--probe-layers", type=int, default=4,
                    help="layers pooled for the selector probe")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--max-candidates", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--cv-epochs", type=int, default=6)
    ap.add_argument("--feature-cache", default=None,
                    help="npz path; reuse the per-path features across runs")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)
    print(f"[g1] device={device} ({describe(device)})", flush=True)

    tr = load_jsonl(os.path.join(args.data, "train.jsonl"))
    va = load_jsonl(os.path.join(args.data, "val.jsonl"))
    te = load_jsonl(os.path.join(args.data, "test.jsonl"))

    def text_of(r):
        return (r["stage1"] + " " + r["stage2"] + " " + r["claim"]).strip()

    tr_t = [text_of(r) for r in tr]
    va_t = [text_of(r) for r in va]
    te_t = [text_of(r) for r in te]
    ytr = torch.tensor([r["label"] for r in tr])
    yva = torch.tensor([r["label"] for r in va])
    yte = torch.tensor([r["label"] for r in te])
    print(f"[g1] train={len(tr)} val={len(va)} test={len(te)}", flush=True)

    from jaume.modeling.config import JaumeConfig
    from jaume.modeling.embedding import JaumeEmbeddingModel
    from jaume.data.dataset import load_tokenizer

    config = JaumeConfig.load(os.path.join(args.model, "config.json"))
    tok = os.path.join(args.model, "tokenizer")
    if not os.path.isdir(tok):
        tok = os.path.join(REPO, "jaume_tokenizer")
    tokenizer = load_tokenizer(tok)
    config.vocab_size = len(tokenizer)
    model = JaumeEmbeddingModel(config).to(device)
    model.load_state_dict(
        load_state_dict(os.path.join(args.model, "pytorch_model.bin")), strict=True)
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)

    # ---- candidate paths, same space as Gate 0 ---------------------------
    cands = build_candidates(config.num_layers, seed=0)
    names = list(cands)
    if args.max_candidates and len(names) > args.max_candidates:
        import random as _r
        rng = _r.Random(0)
        keep = ["base"] + rng.sample([n for n in names if n != "base"],
                                     args.max_candidates - 1)
        names = keep
        cands = {n: cands[n] for n in names}
    print(f"[g1] {len(names)} candidate paths, probe over {args.probe_layers} layers",
          flush=True)

    # ---- features: probe + per-path outputs (cacheable) -------------------
    t0 = time.time()
    Ptr = probe_features(model.backbone, tokenizer, tr_t, args.probe_layers,
                         device, args.max_length)
    Pva = probe_features(model.backbone, tokenizer, va_t, args.probe_layers,
                         device, args.max_length)
    Pte = probe_features(model.backbone, tokenizer, te_t, args.probe_layers,
                         device, args.max_length)
    print(f"[g1] probe features {tuple(Ptr.shape)} in {time.time()-t0:.0f}s", flush=True)

    cache = args.feature_cache
    blob = None
    if cache and os.path.exists(cache):
        blob = torch.load(cache, map_location="cpu", weights_only=False)
        print(f"[g1] reusing feature cache {cache}", flush=True)

    if blob is None:
        t0 = time.time()
        Ftr, Fva, Fte = {}, {}, {}
        for n in names:
            Ftr[n] = encode_path(model, tokenizer, tr_t, cands[n], device, args.max_length)
            Fva[n] = encode_path(model, tokenizer, va_t, cands[n], device, args.max_length)
            Fte[n] = encode_path(model, tokenizer, te_t, cands[n], device, args.max_length)
        print(f"[g1] encoded {len(names)} paths x 3 splits in {time.time()-t0:.0f}s",
              flush=True)
        if cache:
            os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
            torch.save({"names": names, "Ftr": Ftr, "Fva": Fva, "Fte": Fte,
                        "Ptr": Ptr, "Pva": Pva, "Pte": Pte}, cache)
            print(f"[g1] cached features -> {cache}", flush=True)
    else:
        names = blob["names"]
        Ftr, Fva, Fte = blob["Ftr"], blob["Fva"], blob["Fte"]
        Ptr, Pva, Pte = blob["Ptr"], blob["Pva"], blob["Pte"]

    # ---- per-path heads (one per path, reused everywhere) ---------------
    t0 = time.time()
    probs_tr, probs_va, probs_te = {}, {}, {}
    pred_te = {}
    for n in names:
        h, _ = fit_head(Ftr[n], ytr, Fva[n], yva, device, seed=args.seed[0])
        probs_tr[n] = torch.softmax(h(Ftr[n].to(device)).float(), -1).cpu()
        probs_va[n] = torch.softmax(h(Fva[n].to(device)).float(), -1).cpu()
        probs_te[n] = torch.softmax(h(Fte[n].to(device)).float(), -1).cpu()
        pred_te[n] = probs_te[n].argmax(-1)
    print(f"[g1] per-path heads fitted in {time.time()-t0:.0f}s", flush=True)

    # ---- correctness targets on train -----------------------------------
    # correct[i, e] = 1 if expert e got train row i right
    stack_tr = torch.stack([probs_tr[n] for n in names], 1)   # [N, E, 3]
    stack_te = torch.stack([probs_te[n] for n in names], 1)   # [N, E, 3]
    Ctr = torch.stack([probs_tr[n].argmax(-1) == ytr for n in names], 1).float()
    has_any = Ctr.sum(1) > 0
    soft_t = Ctr / Ctr.sum(1, keepdim=True).clamp(min=1)
    w = has_any.float()
    print(f"[g1] train rows with >=1 correct expert: {int(has_any.sum())}/{len(ytr)} "
          f"({float(has_any.float().mean())*100:.1f}%)", flush=True)

    # ---- reference conditions from Gate 0 --------------------------------
    # A: best fixed path, ranked by train CV exactly as Gate 0 does
    n_folds = max(2, args.cv_folds)
    g = torch.Generator().manual_seed(0)
    fold_of = torch.randperm(len(ytr), generator=g) % n_folds
    cv_score = {}
    for n in names:
        f1s = []
        for f in range(n_folds):
            tri = (fold_of != f).nonzero().flatten()
            vai = (fold_of == f).nonzero().flatten()
            h, _ = fit_head(Ftr[n][tri], ytr[tri], Ftr[n][vai], ytr[vai],
                            device, seed=0, epochs=args.cv_epochs)
            f1s.append(macro_f1(ytr[vai].tolist(),
                                predict(h, Ftr[n][vai], device).tolist()))
        cv_score[n] = sum(f1s) / len(f1s)
    best_fixed = max(names, key=lambda n: cv_score[n])
    print(f"[g1] best fixed path = {best_fixed} (CV {cv_score[best_fixed]:.4f})",
          flush=True)

    results = []
    for seed in args.seed:
        A = macro_f1(yte.tolist(), pred_te[best_fixed].tolist())
        probs_c = torch.stack([probs_te[n] for n in names]).mean(0)
        C = macro_f1(yte.tolist(), probs_c.argmax(-1).tolist())

        per_path_pred = torch.stack([pred_te[n] for n in names], 0)   # [E, N]
        corr = (per_path_pred == yte.unsqueeze(0))
        any_ok = corr.any(0)
        cov = float(any_ok.float().mean())
        D = macro_f1(yte.tolist(), torch.where(any_ok, yte, per_path_pred[0]).tolist())

        # --- real selector ---
        S = Selector(Ptr.shape[1], len(names), args.hidden, args.dropout).to(device)
        hist = train_selector(S, Ptr, soft_t, w, Pva, device, seed,
                              args.epochs, args.lr, args.batch_size, args.weight_decay)
        sel_te = selector_probs(S, Pte, device)          # [N, E]
        sel_tr = selector_probs(S, Ptr, device)
        sel_va = selector_probs(S, Pva, device)

        # argmax: pick one expert per row
        pick = sel_te.argmax(1)
        pred_argmax = torch.stack([pred_te[n] for n in names], 1)[
            torch.arange(len(yte)), pick]
        g1_argmax = macro_f1(yte.tolist(), pred_argmax.tolist())

        # soft: weight every expert's class distribution
        blend = torch.einsum("ne,nec->nc", sel_te, stack_te)
        g1_soft = macro_f1(yte.tolist(), blend.argmax(-1).tolist())

        # --- control: same selector, RANDOM targets ---
        torch.manual_seed(seed + 7777)
        rnd_t = torch.rand(len(ytr), len(names))
        rnd_t = rnd_t / rnd_t.sum(1, keepdim=True)
        S2 = Selector(Ptr.shape[1], len(names), args.hidden, args.dropout).to(device)
        train_selector(S2, Ptr, rnd_t, torch.ones(len(ytr)), Pva, device, seed + 1,
                       args.epochs, args.lr, args.batch_size, args.weight_decay)
        sel2_te = selector_probs(S2, Pte, device)
        pick2 = sel2_te.argmax(1)
        pred2 = torch.stack([pred_te[n] for n in names], 1)[
            torch.arange(len(yte)), pick2]
        g1_rand = macro_f1(yte.tolist(), pred2.tolist())

        # --- degeneracy diagnostics ---
        ent = float(-(sel_tr * (sel_tr + 1e-9).log()).sum(1).mean())
        ent_max = torch.log(torch.tensor(float(len(names)))).item()
        modal = int(sel_tr.argmax(1).bincount().argmax())
        frac_modal = float((sel_tr.argmax(1) == modal).float().mean())
        n_distinct = int(sel_tr.argmax(1).unique().numel())

        results.append({
            "seed": seed,
            "G1_argmax": g1_argmax, "G1_soft": g1_soft, "G1_rand": g1_rand,
            "A": A, "C": C, "D": D, "coverage": cov,
            "sel_entropy": ent, "sel_entropy_max": ent_max,
            "frac_picking_modal": frac_modal, "n_distinct_experts": n_distinct,
            "train_loss_first": hist[0] if hist else None,
            "train_loss_last": hist[-1] if hist else None,
        })
        print(f"[g1] seed {seed}: A={A:.4f} C={C:.4f} | "
              f"G1-argmax={g1_argmax:.4f} G1-soft={g1_soft:.4f} "
              f"G1-rand={g1_rand:.4f} | D={D:.4f} cov={cov:.3f}", flush=True)
        print(f"[g1]        selector: entropy={ent:.3f}/{ent_max:.3f}, "
              f"modal-frac={frac_modal:.3f}, distinct={n_distinct}/{len(names)}, "
              f"loss {hist[0]:.3f}->{hist[-1]:.3f}", flush=True)

    def _mean(k):
        return sum(r[k] for r in results) / len(results)

    def _sd(k):
        if len(results) < 2:
            return 0.0
        m = _mean(k)
        return (sum((r[k] - m) ** 2 for r in results) / (len(results) - 1)) ** 0.5

    print("", flush=True)
    print(f"[g1] over {len(results)} seed(s):", flush=True)
    for k, label in (("A", "A single best fixed "), ("C", "C all-path ensemble"),
                     ("G1_argmax", "G1 argmax (1 path)  "),
                     ("G1_soft", "G1 soft (all paths) "),
                     ("G1_rand", "G1 random-target    "),
                     ("D", "D oracle (labelled) ")):
        print(f"[g1]   {label} = {_mean(k):.4f}  sd={_sd(k):.4f}", flush=True)

    g1a = _mean("G1_argmax") - _mean("A")
    g1c = _mean("G1_argmax") - _mean("C")
    g1r = _mean("G1_argmax") - _mean("G1_rand")
    spread = max(_sd("G1_argmax"), _sd("A"), 1e-9)
    print("", flush=True)
    print(f"[g1]   G1 - A    = {g1a:+.4f}   (pass line)", flush=True)
    print(f"[g1]   G1 - C    = {g1c:+.4f}   (real-effect line)", flush=True)
    print(f"[g1]   G1 - rand = {g1r:+.4f}   (must be positive: gain is not noise)",
          flush=True)
    print(f"[g1]   seed sd   = {spread:.4f}", flush=True)
    gate1a = g1a > spread
    gate1c = g1c > spread
    gate1r = g1r > spread
    print(f"[g1] Gate 1a (selector > A, beyond seed noise): "
          f"{'PASS' if gate1a else 'FAIL'}", flush=True)
    print(f"[g1] Gate 1c (selector > C, beyond seed noise): "
          f"{'PASS' if gate1c else 'FAIL'}", flush=True)
    print(f"[g1] Gate 1r (selector > random-target control): "
          f"{'PASS' if gate1r else 'FAIL'}", flush=True)
    verdict = "PASS" if (gate1a and gate1r) else "FAIL"
    print(f"[g1] VERDICT: {verdict}", flush=True)

    payload = {"model": args.model, "k": args.k, "n_paths": len(names),
               "probe_layers": args.probe_layers, "epochs": args.epochs,
               "best_fixed_path": best_fixed, "results": results,
               "mean": {k: _mean(k) for k in
                        ("A", "C", "G1_argmax", "G1_soft", "G1_rand", "D")},
               "g1_minus_a": g1a, "g1_minus_c": g1c, "g1_minus_rand": g1r,
               "gate1a": gate1a, "gate1c": gate1c, "gate1r": gate1r,
               "verdict": verdict}
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[g1] wrote {args.out}", flush=True)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
