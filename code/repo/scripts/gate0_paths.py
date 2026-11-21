"""Gate 0: is a layer-execution-path gain problem-specific or generic?

This is the question the whole TTC direction rests on. A model can be made to
improve by perturbing which layers run, but arXiv:2610.10368 showed that on
multiple-choice evaluation an INPUT-BLIND fixed program -- chosen without
looking at the question -- beat programs selected per question. If the gain
comes from a generic perturbation rather than from matching the path to the
input, then distilling it teaches nothing conditional, and the direction is
empty.

Four conditions, all on the same frozen backbone, same head, same split:

  A  best single FIXED path chosen on train            (baseline)
  B  INPUT-BLIND ensemble of K paths, never per-input  (PLACEBO, mandatory)
  C  ensemble of K randomly drawn paths                (random baseline)
  D  ORACLE: per-sample best path, using labels        (upper bound only)

Verdict rules, frozen in docs/ttc-spec.md §5:

  Gate 0a  D > C   -- otherwise per-input selection has nothing to select
  Gate 0b  D - B   -- the CORE test. If the input-blind placebo is as good,
                      the gain is not problem-specific.

D uses labels and is therefore NEVER part of any training flow. It exists to
bound how much conditional information is available at all.

Usage:
    python scripts/gate0_paths.py --model <ckpt> --seed 0 1 2 --k 8
"""

import argparse
import itertools
import json
import os
import random
import sys

import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
REPO = os.path.join(PROJECT_ROOT, "code", "repo")
sys.path.insert(0, REPO)

from jaume.device import resolve_device, seed_all, describe
from jaume.modeling.checkpoint import load_state_dict


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def macro_f1(y_true, y_pred, n_cls=3):
    f1s = []
    for c in range(n_cls):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s)


def build_candidates(n_layers, seed=0):
    """The candidate path space, as layer-index sequences.

    BASE       [0 .. L-1]
    REPEAT     BASE with the contiguous span [i..j] executed a second time
    SKIP       BASE without layer i
    ITERATE2   BASE twice
    REORDER    a full permutation of BASE (new)
    BLOCKSWAP  BASE with two contiguous blocks exchanged (new)
    REVERSE    a contiguous span executed backwards (new)

    REORDER / BLOCKSWAP / REVERSE exist because the first Gate 0 run found
    that every candidate was a REPEAT variant: 39 of 40 paths re-executed the
    same span structure, so their errors were highly correlated and an
    ensemble of them could not beat a single path. Diversity has to come from
    structurally different paths, not from more repeats.
    """
    import random as _random

    base = list(range(n_layers))
    cands = {"base": base}

    # repeats: quadratic, the original family
    for i in range(n_layers):
        for j in range(i, n_layers):
            span = base[i:j + 1]
            cands["repeat_%d_%d" % (i, j)] = base[:j + 1] + span + base[j + 1:]

    # skips: single-layer removal
    for i in range(n_layers):
        cands["skip_%d" % i] = [l for l in base if l != i]

    cands["iterate2"] = base + base

    # ---- structurally different families -------------------------------
    # reorders: fixed-seed permutations, never the identity
    rng = _random.Random(seed)
    seen = {tuple(base)}
    for k in range(8):
        perm = base[:]
        for _ in range(100):
            rng.shuffle(perm)
            if tuple(perm) not in seen:
                break
        seen.add(tuple(perm))
        cands["reorder_%d" % k] = list(perm)

    # block swaps: exchange two contiguous blocks, keeping the middle
    for (a, b, c) in ((0, 4, 8), (4, 8, 12), (0, 8, 12), (8, 12, 16),
                      (2, 6, 10), (6, 10, 14)):
        if c <= n_layers:
            cands["blockswap_%d_%d_%d" % (a, b, c)] = (
                base[:a] + base[b:c] + base[a:b] + base[c:])

    # reversals: a span executed backwards
    for (a, b) in ((0, 8), (4, 12), (8, 16), (0, 16), (2, 10), (6, 14)):
        if b <= n_layers:
            cands["reverse_%d_%d" % (a, b)] = base[:a] + base[a:b][::-1] + base[b:]

    return cands


def forward_with_path(backbone, input_ids, attention_mask, path):
    """Run the backbone along an explicit layer-index sequence."""
    h = backbone.embed_tokens(input_ids)
    aux_total = torch.zeros((), device=input_ids.device)
    for li in path:
        h, aux = backbone.layers[li](h, attention_mask)
        if aux is not None:
            aux_total = aux_total + aux
    return backbone.norm(h), aux_total


@torch.no_grad()
def encode_path(model, tokenizer, texts, path, device, max_length, bs=32):
    feats = []
    for i in range(0, len(texts), bs):
        enc = tokenizer(texts[i:i + bs], padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt").to(device)
        h, _ = forward_with_path(model.backbone, enc["input_ids"], enc["attention_mask"], path)
        m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        feats.append(((h * m).sum(1) / m.sum(1).clamp(min=1e-9)).float().cpu())
    return torch.cat(feats, 0)


def fit_head(Xtr, ytr, Xva, yva, device, seed, epochs=8, lr=1e-2, bs=32):
    seed_all(device, seed)
    torch.manual_seed(seed)
    head = nn.Linear(Xtr.shape[1], 3).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    Xtr_d, ytr_d = Xtr.to(device), ytr.to(device)
    Xva_d, yva_d = Xva.to(device), yva.to(device)
    best, best_state = -1.0, None
    for _ in range(epochs):
        head.train()
        perm = torch.randperm(len(Xtr_d), device=device)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            loss = nn.functional.cross_entropy(head(Xtr_d[idx]), ytr_d[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            f1 = macro_f1(yva.tolist(), head(Xva_d).argmax(-1).cpu().tolist())
        if f1 > best:
            best = f1
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    head.eval()
    return head, best


def predict(head, X, device, bs=512):
    head.eval()
    outs = []
    with torch.no_grad():
        Xd = X.to(device)
        for i in range(0, len(Xd), bs):
            outs.append(head(Xd[i:i + bs]).argmax(-1).cpu())
    return torch.cat(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=os.path.join(PROJECT_ROOT, "data", "scifact_blocks"))
    ap.add_argument("--seed", type=int, nargs="+", default=[0])
    ap.add_argument("--k", type=int, default=8, help="paths in the B/C ensembles")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--path-seed", type=int, default=0,
                    help="seed for the REORDER path family")
    ap.add_argument("--cv-folds", type=int, default=5,
                    help="folds for ranking paths inside the TRAIN split")
    ap.add_argument("--cv-epochs", type=int, default=6,
                    help="epochs for the cheap CV ranking heads")
    ap.add_argument("--max-candidates", type=int, default=0,
                    help="subsample the candidate space (0 = all)")
    args = ap.parse_args()

    device = resolve_device(args.device)
    print(f"[g0] device={device} ({describe(device)})", flush=True)

    tr = load_jsonl(os.path.join(args.data, "train.jsonl"))
    va = load_jsonl(os.path.join(args.data, "val.jsonl"))
    te = load_jsonl(os.path.join(args.data, "test.jsonl"))
    def text_of(r):
        return (r["stage1"] + " " + r["stage2"] + " " + r["claim"]).strip()
    tr_t, va_t, te_t = [text_of(r) for r in tr], [text_of(r) for r in va], [text_of(r) for r in te]
    ytr = torch.tensor([r["label"] for r in tr])
    yva = torch.tensor([r["label"] for r in va])
    yte = torch.tensor([r["label"] for r in te])
    print(f"[g0] train={len(tr)} val={len(va)} test={len(te)}", flush=True)

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
    model.load_state_dict(load_state_dict(os.path.join(args.model, "pytorch_model.bin")), strict=True)
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)

    cands = build_candidates(config.num_layers, seed=args.path_seed)
    names = list(cands)
    if args.max_candidates and len(names) > args.max_candidates:
        rng = random.Random(0)
        keep = ["base"] + rng.sample([n for n in names if n != "base"],
                                     args.max_candidates - 1)
        names = keep
        cands = {n: cands[n] for n in names}
    print(f"[g0] candidate paths: {len(names)}", flush=True)

    # ---- per-path features, encoded ONCE on all three splits --------------
    import time
    t0 = time.time()
    Ftr, Fva, Fte = {}, {}, {}
    for n in names:
        Ftr[n] = encode_path(model, tokenizer, tr_t, cands[n], device, args.max_length)
        Fva[n] = encode_path(model, tokenizer, va_t, cands[n], device, args.max_length)
        Fte[n] = encode_path(model, tokenizer, te_t, cands[n], device, args.max_length)
    print(f"[g0] encoded {len(names)} paths x 3 splits in {time.time()-t0:.0f}s", flush=True)

    # ---- honest path ranking: k-fold CV inside TRAIN ---------------------
    #
    # The previous version ranked paths by their score on val (164 rows) and
    # then reported a test number. That selected for val overfitting: the
    # chosen "top-8" ensemble scored 0.3179 on test while a RANDOM 8-path
    # ensemble scored 0.3580. A good ensemble losing to a random one is the
    # signature of selection noise, not a real effect.
    #
    # Ranking by cross-validation inside train (716 rows) uses every row for
    # scoring without ever touching val or test.
    n_folds = max(2, args.cv_folds)
    g = torch.Generator().manual_seed(0)
    fold_of = torch.randperm(len(ytr), generator=g) % n_folds

    cv_score = {}
    t0 = time.time()
    for n in names:
        f1s = []
        for f in range(n_folds):
            tr_idx = (fold_of != f).nonzero().flatten()
            va_idx = (fold_of == f).nonzero().flatten()
            h, _ = fit_head(Ftr[n][tr_idx], ytr[tr_idx],
                            Ftr[n][va_idx], ytr[va_idx],
                            device, seed=0, epochs=args.cv_epochs)
            f1s.append(macro_f1(ytr[va_idx].tolist(),
                                predict(h, Ftr[n][va_idx], device).tolist()))
        cv_score[n] = sum(f1s) / len(f1s)
    print(f"[g0] {n_folds}x CV ranking over {len(names)} paths in "
          f"{time.time()-t0:.0f}s", flush=True)

    ranked = sorted(names, key=lambda n: cv_score[n], reverse=True)
    best_fixed = ranked[0]
    topk_paths = ranked[:min(args.k, len(ranked))]
    print(f"[g0] best single path = {best_fixed}  (CV {cv_score[best_fixed]:.4f})",
          flush=True)
    print("[g0] top-%d by CV: %s" % (
        len(topk_paths),
        ", ".join("%s(%.3f)" % (n, cv_score[n]) for n in topk_paths)), flush=True)

    results = []
    print(f"[g0] fitting final heads: {len(names)} paths x {len(args.seed)} seeds",
          flush=True)

    for seed in args.seed:
        # One head per path, trained on the FULL train split, reused by A-D.
        probs_by_path, preds_by_path = {}, {}
        for n in names:
            h, _ = fit_head(Ftr[n], ytr, Fva[n], yva, device, seed=seed)
            probs_by_path[n] = torch.softmax(h(Fte[n].to(device)).float(), -1).cpu()
            preds_by_path[n] = probs_by_path[n].argmax(-1)

        # A: single best fixed path, chosen by CV (never by val/test)
        A = macro_f1(yte.tolist(), preds_by_path[best_fixed].tolist())

        # B: input-blind ensemble of the top-K paths by CV (the PLACEBO)
        probs_b = torch.stack([probs_by_path[n] for n in topk_paths]).mean(0)
        B = macro_f1(yte.tolist(), probs_b.argmax(-1).tolist())

        # C: ensemble of EVERY candidate path -- zero selection of any kind,
        #    so it cannot overfit val or CV.
        probs_c = torch.stack([probs_by_path[n] for n in names]).mean(0)
        C = macro_f1(yte.tolist(), probs_c.argmax(-1).tolist())

        # D: ORACLE -- per sample, take a path that was correct. Uses the
        #    label and is UNREACHABLE; it only bounds how much conditional
        #    information exists at all.
        preds = torch.stack([preds_by_path[n] for n in names], 0)   # [P, N]
        correct = (preds == yte.unsqueeze(0))
        any_ok = correct.any(0)
        cov = float(any_ok.float().mean())
        D = macro_f1(yte.tolist(), torch.where(any_ok, yte, preds[0]).tolist())

        results.append({"seed": seed, "A_fixed": A, "B_placebo": B,
                        "C_all_paths": C, "D_oracle": D,
                        "oracle_coverage": cov,
                        "best_fixed_path": best_fixed,
                        "placebo_paths": list(topk_paths),
                        "cv_score_best": cv_score[best_fixed]})
        print(f"[g0] seed {seed}: A={A:.4f}  B(top{len(topk_paths)})={B:.4f}  "
              f"C(all {len(names)})={C:.4f}  D(oracle)={D:.4f}  cov={cov:.3f}",
              flush=True)

    d_minus_b = [r["D_oracle"] - r["B_placebo"] for r in results]
    d_minus_c = [r["D_oracle"] - r["C_all_paths"] for r in results]
    a_mean = sum(r["A_fixed"] for r in results) / len(results)
    b_mean = sum(r["B_placebo"] for r in results) / len(results)
    c_mean = sum(r["C_all_paths"] for r in results) / len(results)
    d_mean = sum(r["D_oracle"] for r in results) / len(results)
    cov_mean = sum(r["oracle_coverage"] for r in results) / len(results)

    def _spread(key):
        vals = [r[key] for r in results]
        if len(vals) < 2:
            return 0.0, 0.0
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
        return (var ** 0.5), max(vals) - min(vals)

    a_sd, a_rg = _spread("A_fixed")
    b_sd, b_rg = _spread("B_placebo")
    c_sd, c_rg = _spread("C_all_paths")
    d_sd, d_rg = _spread("D_oracle")

    print("", flush=True)
    print(f"[g0] over {len(results)} seed(s):", flush=True)
    print(f"[g0]   A single best fixed   = {a_mean:.4f}  sd={a_sd:.4f} range={a_rg:.4f}",
          flush=True)
    print(f"[g0]   B top-K ensemble      = {b_mean:.4f}  sd={b_sd:.4f} range={b_rg:.4f}"
          f"   (input-blind, CV-selected)", flush=True)
    print(f"[g0]   C all-path ensemble   = {c_mean:.4f}  sd={c_sd:.4f} range={c_rg:.4f}"
          f"   (zero selection)", flush=True)
    print(f"[g0]   D oracle (unreachable) = {d_mean:.4f}  sd={d_sd:.4f} range={d_rg:.4f}"
          f"   coverage={cov_mean:.3f}", flush=True)

    # The question 'is an ensemble actually better than one path?' needs a
    # noise yardstick. Compare the C-A gap against the seed-to-seed spread:
    # if the gap is smaller than the spread, the difference is not resolved.
    ca_gap = c_mean - a_mean
    spread = max(a_sd, c_sd, 1e-9)
    resolved = abs(ca_gap) > spread
    print(f"[g0]   C-A = {ca_gap:+.4f}   seed sd ~{spread:.4f}   "
          f"=> {'RESOLVED' if resolved else 'NOT RESOLVED (gap within seed noise)'}",
          flush=True)
    print(f"[g0]   D-B = {sum(d_minus_b)/len(d_minus_b):+.4f}   "
          f"D-C = {sum(d_minus_c)/len(d_minus_c):+.4f}", flush=True)
    gate0a = all(x > 0 for x in d_minus_c)
    gate0b = all(x > 0 for x in d_minus_b)
    print(f"[g0] Gate 0a (D > C, oracle beats the zero-selection ensemble): "
          f"{'PASS' if gate0a else 'FAIL'}", flush=True)
    print(f"[g0] Gate 0b (D > B, oracle beats the CV-selected ensemble): "
          f"{'PASS' if gate0b else 'FAIL'}", flush=True)
    verdict = "PASS" if (gate0a and gate0b) else "FAIL"
    print(f"[g0] VERDICT: {verdict}", flush=True)
    print("", flush=True)
    print(f"[g0] READ THIS BEFORE QUOTING D: D picks, per sample, a path that "
          f"happened to be correct, using the label. It is an upper bound and "
          f"CANNOT be implemented at inference. Coverage {cov_mean:.3f} means "
          f"{1-cov_mean:.1%} of test rows had no correct path at all, so D is "
          f"capped there regardless of selector quality.", flush=True)

    payload = {"model": args.model, "k": args.k, "n_candidates": len(names),
               "cv_folds": n_folds, "best_fixed_path": best_fixed,
               "cv_scores": cv_score, "results": results,
               "spread": {"A_sd": a_sd, "B_sd": b_sd, "C_sd": c_sd, "D_sd": d_sd},
               "c_minus_a": ca_gap, "c_minus_a_resolved": resolved,
               "gate0a_d_gt_c": gate0a, "gate0b_d_gt_b": gate0b, "verdict": verdict}
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[g0] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
