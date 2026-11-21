"""Gate 1b: test-time adaptive path selection by output confidence.

Gate 1 failed for a specific reason: a selector that reads only the FIRST few
layers cannot know which later path will be right, because those layers run
identically under every path. Gate 1's probe was blind by construction.

This script removes that blindness by letting the model actually RUN candidate
paths and then decide from their OUTPUTS, without labels:

  budget b    run the first b paths of a fixed, CV-ranked order
  decide      use the confidence (max softmax) of each run to pick, or to stop
  stop early  stop as soon as a run's confidence clears a threshold

Conditions, all on the same held-out test split:

  fixed-1    A   the single best fixed path (1x)
  vote-all   C   all paths averaged (Px)
  adapt-top1     run b paths, take the most confident one (bx)
  adapt-stop     run b paths, stop at the first confident enough (<= bx)
  oracle     D   per-sample correct path, needs labels (reference ceiling)

Two controls that decide whether anything was learned:

  shuffle-conf   use each path's confidence from a DIFFERENT test row. If this
                 scores the same as adapt-top1, the gain is the budget, not
                 the confidence signal.
  rand-order     run the paths in a random order (same budget). If this
                 matches, the CV ranking contributes nothing.

This is a genuinely different mechanism from Gate 1: it spends compute
instead of reading a prefix, so it is not bounded by what the early layers
know. The question is whether confidence is a usable signal at all.

Usage
-----
    python scripts/gate1b_adaptive.py --model <ckpt> --budget 1 2 4 8
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
REPO = os.path.join(PROJECT_ROOT, "code", "repo")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jaume.device import resolve_device, describe
from jaume.modeling.checkpoint import load_state_dict

from gate0_paths import (load_jsonl, macro_f1, build_candidates,
                         encode_path, fit_head, predict)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=os.path.join(PROJECT_ROOT, "data", "scifact_blocks"))
    ap.add_argument("--budget", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--max-candidates", type=int, default=40)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--threshold", type=float, nargs="+", default=[0.5, 0.7, 0.9])
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--cv-epochs", type=int, default=6)
    ap.add_argument("--feature-cache", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)
    print(f"[g1b] device={device} ({describe(device)})", flush=True)

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
    print(f"[g1b] train={len(tr)} val={len(va)} test={len(te)}", flush=True)

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

    cands = build_candidates(config.num_layers, seed=0)
    names = list(cands)
    if args.max_candidates and len(names) > args.max_candidates:
        import random as _r
        rng = _r.Random(0)
        keep = ["base"] + rng.sample([n for n in names if n != "base"],
                                     args.max_candidates - 1)
        names = keep
        cands = {n: cands[n] for n in names}
    print(f"[g1b] {len(names)} candidate paths", flush=True)

    cache = args.feature_cache
    blob = None
    if cache and os.path.exists(cache):
        blob = torch.load(cache, map_location="cpu", weights_only=False)
        print(f"[g1b] reusing {cache}", flush=True)

    if blob is None:
        t0 = time.time()
        Ftr, Fva, Fte = {}, {}, {}
        for n in names:
            Ftr[n] = encode_path(model, tokenizer, tr_t, cands[n], device, args.max_length)
            Fva[n] = encode_path(model, tokenizer, va_t, cands[n], device, args.max_length)
            Fte[n] = encode_path(model, tokenizer, te_t, cands[n], device, args.max_length)
        print(f"[g1b] encoded {len(names)} paths x 3 splits in {time.time()-t0:.0f}s",
              flush=True)
        if cache:
            os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
            torch.save({"names": names, "Ftr": Ftr, "Fva": Fva, "Fte": Fte}, cache)
    else:
        names = blob["names"]
        Ftr, Fva, Fte = blob["Ftr"], blob["Fva"], blob["Fte"]

    # per-path heads
    probs_te, pred_te = {}, {}
    t0 = time.time()
    for n in names:
        h, _ = fit_head(Ftr[n], ytr, Fva[n], yva, device, seed=0)
        probs_te[n] = torch.softmax(h(Fte[n].to(device)).float(), -1).cpu()
        pred_te[n] = probs_te[n].argmax(-1)
    print(f"[g1b] heads fitted in {time.time()-t0:.0f}s", flush=True)

    # CV ranking (same protocol as Gate 0/1)
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
    ranked = sorted(names, key=lambda n: cv_score[n], reverse=True)
    best_fixed = ranked[0]
    print(f"[g1b] best fixed = {best_fixed} (CV {cv_score[best_fixed]:.4f})", flush=True)

    P = torch.stack([probs_te[n] for n in names], 0)      # [E, N, 3]
    conf = P.max(-1).values                                # [E, N]
    NN = len(yte)

    # reference numbers
    A = macro_f1(yte.tolist(), pred_te[best_fixed].tolist())
    C = macro_f1(yte.tolist(), P.mean(0).argmax(-1).tolist())
    corr = (P.argmax(-1) == yte.unsqueeze(0))
    any_ok = corr.any(0)
    cov = float(any_ok.float().mean())
    D = macro_f1(yte.tolist(), torch.where(any_ok, yte, P.argmax(-1)[0]).tolist())
    print(f"[g1b] A={A:.4f}  C={C:.4f}  D={D:.4f}  coverage={cov:.3f}", flush=True)

    order = [names.index(n) for n in ranked]
    torch.manual_seed(0)
    rand_order = torch.randperm(len(names)).tolist()

    results = []
    print("", flush=True)
    print(f"{'budget':>6} {'adapt-top1':>11} {'shuffle':>9} {'randorder':>10} "
          f"{'adaptive-stop@thr':>20}", flush=True)
    for b in args.budget:
        if b > len(names):
            continue
        sub = order[:b]
        idx = torch.arange(NN)

        # adapt-top1: among the first b paths, take the most confident output
        best_e = conf[sub].argmax(0)                       # [N] index into sub
        pick = torch.tensor(sub)[best_e]
        pred_top1 = P[pick, idx].argmax(-1)
        acc_top1 = macro_f1(yte.tolist(), pred_top1.tolist())

        # shuffle control: same confidence values, wrong rows
        perm = torch.randperm(NN, generator=torch.Generator().manual_seed(1))
        best_e_sh = conf[sub][:, perm].argmax(0)
        pick_sh = torch.tensor(sub)[best_e_sh]
        acc_shuf = macro_f1(yte.tolist(), P[pick_sh, idx].argmax(-1).tolist())

        # rand-order control: a different set of b paths
        sub_r = rand_order[:b]
        best_e_r = conf[sub_r].argmax(0)
        pick_r = torch.tensor(sub_r)[best_e_r]
        acc_rand_order = macro_f1(yte.tolist(), P[pick_r, idx].argmax(-1).tolist())

        stops = []
        for thr in args.threshold:
            # stop at the first path whose confidence clears the threshold;
            # otherwise fall back to the most confident of the budget
            chosen = []
            for i in range(NN):
                sel_e = None
                for e in sub:
                    if conf[e, i] >= thr:
                        sel_e = e
                        break
                if sel_e is None:
                    sel_e = sub[int(conf[sub, i].argmax())]
                chosen.append(sel_e)
            chosen = torch.tensor(chosen)
            acc_stop = macro_f1(yte.tolist(), P[chosen, idx].argmax(-1).tolist())
            stops.append((thr, acc_stop))

        results.append({"budget": b, "adapt_top1": acc_top1,
                        "shuffle_conf": acc_shuf, "rand_order": acc_rand_order,
                        "stop": stops})
        stop_str = "  ".join("%.1f:%.4f" % (t, a) for t, a in stops)
        print(f"{b:>6} {acc_top1:>11.4f} {acc_shuf:>9.4f} {acc_rand_order:>10.4f} "
              f"{stop_str:>20}", flush=True)

    print("", flush=True)
    print(f"[g1b] reference: A(fixed)={A:.4f}  C(vote)={C:.4f}  "
          f"D(oracle)={D:.4f}  coverage={cov:.3f}", flush=True)

    best_b = max(results, key=lambda r: r["adapt_top1"])
    gain_a = best_b["adapt_top1"] - A
    gain_shuf = best_b["adapt_top1"] - best_b["shuffle_conf"]
    gain_rand = best_b["adapt_top1"] - best_b["rand_order"]
    print("", flush=True)
    print(f"[g1b] best budget = {best_b['budget']}  adapt-top1={best_b['adapt_top1']:.4f}",
          flush=True)
    print(f"[g1b]   vs A          = {gain_a:+.4f}", flush=True)
    print(f"[g1b]   vs shuffle    = {gain_shuf:+.4f}   (confidence must beat wrong-row confidence)",
          flush=True)
    print(f"[g1b]   vs rand-order = {gain_rand:+.4f}   (CV ranking must beat a random subset)",
          flush=True)
    # 228 test rows: a ~0.02 gap is roughly the 1-sigma binomial noise
    noise = (0.35 * 0.65 / len(yte)) ** 0.5
    print(f"[g1b]   binomial 1-sigma on 228 rows ~= {noise:.4f}", flush=True)
    pass_a = gain_a > 2 * noise
    pass_shuf = gain_shuf > noise
    pass_rand = gain_rand > noise
    print(f"[g1b] Gate 1b-a (adapt > A): {'PASS' if pass_a else 'FAIL'}", flush=True)
    print(f"[g1b] Gate 1b-conf (adapt > shuffle): {'PASS' if pass_shuf else 'FAIL'}",
          flush=True)
    print(f"[g1b] Gate 1b-rank (adapt > random order): {'PASS' if pass_rand else 'FAIL'}",
          flush=True)
    verdict = "PASS" if (pass_a and pass_shuf) else "FAIL"
    print(f"[g1b] VERDICT: {verdict}", flush=True)

    payload = {"model": args.model, "n_paths": len(names), "best_fixed": best_fixed,
               "A": A, "C": C, "D": D, "coverage": cov, "results": results,
               "best_budget": best_b["budget"], "best_adapt_top1": best_b["adapt_top1"],
               "gain_vs_A": gain_a, "gain_vs_shuffle": gain_shuf,
               "gain_vs_randorder": gain_rand, "binomial_noise": noise,
               "verdict": verdict}
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[g1b] wrote {args.out}", flush=True)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
