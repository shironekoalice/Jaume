"""Gate 0/1 on Qasper: does path choice matter for evidence localisation?

Why Qasper, after SciFact failed
--------------------------------
On SciFact (3-way claim verification) the oracle over layer paths reached
0.7194 while every input-blind mechanism stayed at ~0.35, and two controls
killed the selector idea outright:

  * G1-rand (0.3483) beat the real selector (0.3324);
  * with budget 16, shuffled confidences (0.3826) beat real ones (0.3470).

The working hypothesis for that failure is TASK DEPTH: a three-way claim
verification is largely decided by surface evidence matching, so reordering
layers barely changes the output and no selection signal survives.

Qasper is a different shape: 34 candidate blocks per paper, the model must
pick the ONE block holding the answer evidence. That is a 34-way search that
has to compare every block against the question, which is exactly the kind of
work where "how many layers, in what order" should matter more.

Task and metric
---------------
  score(i)  = cos(question_emb, block_i_emb)   for every block of a paper
  pred      = argmax_i score(i)
  metric    = accuracy@1 over that paper's blocks

Conditions
----------
  A     single best fixed path (CV-ranked on train accuracy@1)
  C     all-path ensemble: mean of per-path score vectors
  D     oracle: per paper, a path that ranked the gold block first (labels)
  G1    selector over probe features, pick one path per paper
  Gr    same selector trained on random targets (control)

All paths are the same 174-candidate space used for SciFact; --max-candidates
subsamples it, matching the earlier runs.

Usage
-----
    python scripts/gate_qasper.py --model <ckpt> --max-candidates 40
    python scripts/gate_qasper.py --model <ckpt> --max-candidates 40 --gate1
"""

import argparse
import json
import os
import random
import sys
import time

import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
REPO = os.path.join(PROJECT_ROOT, "code", "repo")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jaume.device import resolve_device, seed_all, describe
from jaume.modeling.checkpoint import load_state_dict

from gate0_paths import build_candidates, forward_with_path


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


@torch.no_grad()
def embed_texts(model, tokenizer, texts, path, device, max_length, bs=32):
    """Mean-pooled embedding of each text under one explicit layer path."""
    outs = []
    for i in range(0, len(texts), bs):
        enc = tokenizer(texts[i:i + bs], padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt").to(device)
        h, _ = forward_with_path(model.backbone, enc["input_ids"],
                                 enc["attention_mask"], path)
        m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        outs.append(((h * m).sum(1) / m.sum(1).clamp(min=1e-9)).float().cpu())
    return torch.cat(outs, 0)


@torch.no_grad()
def score_papers(model, tokenizer, rows, path, device, max_length,
                 max_blocks=34, batch_texts=96):
    """Per-paper score vectors over that paper's blocks.

    Batching matters here: one paper has up to 34 blocks, so looping
    paper-by-paper would issue ~35 forwards per paper per path. Instead every
    (question or block) text of a paper is packed into one flat batch, encoded
    together, and unpacked afterwards. That turns ~35 small forwards into one
    larger one, which is the difference between ~4 hours and ~20 minutes for
    40 paths x 283 papers.

    Blocks beyond `max_blocks` are dropped (Qasper's own builder caps at 34,
    but rows are re-checked here).
    """
    scores, golds = [], []
    for r in rows:
        blocks = r["blocks"][:max_blocks]
        if not (0 <= int(r["evidence_idx"]) < len(blocks)):
            continue
        texts = [r["question"]] + blocks
        parts = []
        for i in range(0, len(texts), batch_texts):
            parts.append(embed_texts(model, tokenizer, texts[i:i + batch_texts],
                                     path, device, max_length, bs=batch_texts))
        emb = torch.cat(parts, 0)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        scores.append(emb[1:] @ emb[0])
        golds.append(int(r["evidence_idx"]))
    return scores, golds


def acc_at_1(scores, golds):
    hit = sum(1 for s, g in zip(scores, golds) if int(s.argmax()) == g)
    return hit / max(len(golds), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=os.path.join(PROJECT_ROOT, "data", "qasper_blocks"))
    ap.add_argument("--max-candidates", type=int, default=40)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--max-rows", type=int, default=0,
                    help="subsample rows per split (0 = all)")
    ap.add_argument("--gate1", action="store_true",
                    help="also train the per-input selector and its control")
    ap.add_argument("--probe-layers", type=int, default=16)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--device", default=None)
    ap.add_argument("--feature-cache", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)
    print(f"[q] device={device} ({describe(device)})", flush=True)

    tr = load_jsonl(os.path.join(args.data, "train.jsonl"))
    va = load_jsonl(os.path.join(args.data, "val.jsonl"))
    te = load_jsonl(os.path.join(args.data, "test.jsonl"))
    if args.max_rows:
        tr, va, te = tr[:args.max_rows], va[:args.max_rows], te[:args.max_rows]
    print(f"[q] papers train={len(tr)} val={len(va)} test={len(te)}", flush=True)
    print(f"[q] mean blocks: {sum(r['n_blocks'] for r in te)/len(te):.1f}", flush=True)

    from jaume.modeling.config import JaumeConfig
    from jaume.modeling.embedding import JaumeEmbeddingModel
    from jaume.data.dataset import load_tokenizer

    config = JaumeConfig.load(os.path.join(args.model, "config.json"))
    tokdir = os.path.join(args.model, "tokenizer")
    if not os.path.isdir(tokdir):
        tokdir = os.path.join(REPO, "jaume_tokenizer")
    tokenizer = load_tokenizer(tokdir)
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
        rng = random.Random(0)
        keep = ["base"] + rng.sample([n for n in names if n != "base"],
                                     args.max_candidates - 1)
        names = keep
        cands = {n: cands[n] for n in names}
    print(f"[q] {len(names)} candidate paths", flush=True)

    cache = args.feature_cache
    blob = None
    if cache and os.path.exists(cache):
        blob = torch.load(cache, map_location="cpu", weights_only=False)
        print(f"[q] reusing cache {cache}", flush=True)

    if blob is None:
        t0 = time.time()
        Str, Sva, Ste = {}, {}, {}
        for n in names:
            Str[n], _ = score_papers(model, tokenizer, tr, cands[n], device, args.max_length)
            Sva[n], _ = score_papers(model, tokenizer, va, cands[n], device, args.max_length)
            Ste[n], _ = score_papers(model, tokenizer, te, cands[n], device, args.max_length)
        print(f"[q] scored {len(names)} paths x 3 splits in {time.time()-t0:.0f}s", flush=True)
        golds_tr = [int(r["evidence_idx"]) for r in tr]
        golds_te = [int(r["evidence_idx"]) for r in te]
        blob = {"names": names, "Str": Str, "Sva": Sva, "Ste": Ste,
                "golds_tr": golds_tr, "golds_te": golds_te}
        if cache:
            os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
            torch.save(blob, cache)
            print(f"[q] cached -> {cache}", flush=True)
    else:
        names = blob["names"]
        Str, Sva, Ste = blob["Str"], blob["Sva"], blob["Ste"]
    golds_tr = blob["golds_tr"]
    golds_te = blob["golds_te"]

    # ---- reference numbers ----
    A_per_path = {n: acc_at_1(Ste[n], golds_te) for n in names}
    A_path = max(names, key=lambda n: A_per_path[n])
    A = A_per_path[A_path]

    # C: mean score vector across all paths, per paper
    def ensemble_acc(sel):
        hit = 0
        for i in range(len(golds_te)):
            vs = torch.stack([Ste[n][i] for n in sel]).mean(0)
            hit += int(int(vs.argmax()) == golds_te[i])
        return hit / max(len(golds_te), 1)
    C = ensemble_acc(names)

    # D: oracle -- a path that gets this paper right
    oracle_hit = 0
    coverage = 0
    for i in range(len(golds_te)):
        ok = False
        for n in names:
            if int(Ste[n][i].argmax()) == golds_te[i]:
                ok = True
                break
        coverage += int(ok)
        oracle_hit += int(ok)
    D = oracle_hit / max(len(golds_te), 1)
    coverage /= max(len(golds_te), 1)

    print("", flush=True)
    print(f"[q] A best-fixed ({A_path}) = {A:.4f}", flush=True)
    print(f"[q] C all-path ensemble     = {C:.4f}", flush=True)
    print(f"[q] D oracle (labels)       = {D:.4f}   coverage={coverage:.3f}", flush=True)

    result = {"model": args.model, "n_paths": len(names),
              "n_test_papers": len(te), "best_fixed": A_path,
              "A": A, "C": C, "D": D, "coverage": coverage,
              "A_per_path": A_per_path}

    # ---- optional Gate 1 selector ----
    if args.gate1:
        def probe_features(rows):
            feats = []
            for r in rows:
                emb = embed_texts(model, tokenizer, [r["question"]],
                                  list(range(config.num_layers)), device, args.max_length)
                feats.append(emb[0])
            return torch.stack(feats, 0)

        t0 = time.time()
        Ptr = probe_features(tr)
        Pte = probe_features(te)
        print(f"[q] probe features {tuple(Ptr.shape)} in {time.time()-t0:.0f}s", flush=True)

        E = len(names)
        # target: which paths got this train paper right
        Ctr = torch.zeros(len(tr), E)
        for j, n in enumerate(names):
            for i in range(len(tr)):
                Ctr[i, j] = float(int(Str[n][i].argmax()) == golds_tr[i])
        has_any = Ctr.sum(1) > 0
        soft_t = Ctr / Ctr.sum(1, keepdim=True).clamp(min=1)
        w = has_any.float()
        print(f"[q] train papers with >=1 correct path: {int(has_any.sum())}/{len(tr)}",
              flush=True)

        class Sel(nn.Module):
            def __init__(self, d, e, h=128, dp=0.2):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(dp),
                                         nn.Linear(h, h), nn.GELU(), nn.Dropout(dp),
                                         nn.Linear(h, e))
            def forward(self, x):
                return self.net(x)

        def train_sel(S, X, T, W, seed):
            seed_all(device, seed); torch.manual_seed(seed)
            opt = torch.optim.AdamW(S.parameters(), lr=3e-3, weight_decay=1e-4)
            Xd, Td, Wd = X.to(device), T.to(device), W.to(device)
            for _ in range(args.epochs):
                S.train()
                perm = torch.randperm(len(Xd), device=device)
                for i in range(0, len(perm), 64):
                    idx = perm[i:i+64]
                    logp = torch.log_softmax(S(Xd[idx]), -1)
                    li = -(Td[idx] * logp).sum(-1)
                    loss = (li * Wd[idx]).sum() / Wd[idx].sum().clamp(min=1e-9)
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            return S

        @torch.no_grad()
        def sel_probs(S, X):
            S.eval(); return torch.softmax(S(X.to(device)).float(), -1).cpu()

        def eval_selector(S, Xte):
            sp = sel_probs(S, Xte)
            hit = 0
            for i in range(len(golds_te)):
                e = int(sp[i].argmax())
                if int(Ste[names[e]][i].argmax()) == golds_te[i]:
                    hit += 1
            return hit / max(len(golds_te), 1), sp

        g1_runs = []
        for seed in args.seeds:
            S = Sel(Ptr.shape[1], E).to(device)
            train_sel(S, Ptr, soft_t, w, seed)
            g1, sp = eval_selector(S, Pte)
            # random-target control
            torch.manual_seed(seed + 7777)
            rnd = torch.rand(len(tr), E); rnd = rnd / rnd.sum(1, keepdim=True)
            S2 = Sel(Ptr.shape[1], E).to(device)
            train_sel(S2, Ptr, rnd, torch.ones(len(tr)), seed + 1)
            gr, _ = eval_selector(S2, Pte)
            ent = float(-(sp * (sp + 1e-9).log()).sum(1).mean())
            modal = int(sp.argmax(1).bincount().argmax())
            frac_modal = float((sp.argmax(1) == modal).float().mean())
            g1_runs.append({"seed": seed, "g1": g1, "g1_rand": gr,
                            "entropy": ent, "entropy_max": float(torch.log(torch.tensor(float(E)))),
                            "modal_frac": frac_modal})
            print(f"[q] seed {seed}: G1={g1:.4f}  G1-rand={gr:.4f}  "
                  f"ent={ent:.2f}/{float(torch.log(torch.tensor(float(E)))):.2f}  "
                  f"modal={frac_modal:.3f}", flush=True)

        m1 = sum(r["g1"] for r in g1_runs) / len(g1_runs)
        mr = sum(r["g1_rand"] for r in g1_runs) / len(g1_runs)
        print("", flush=True)
        print(f"[q] G1={m1:.4f}  G1-rand={mr:.4f}  G1-A={m1-A:+.4f}  "
              f"G1-rand_gap={m1-mr:+.4f}", flush=True)
        noise = (A * (1 - A) / max(len(te), 1)) ** 0.5
        print(f"[q] binomial 1-sigma on {len(te)} papers ~= {noise:.4f}", flush=True)
        pa = (m1 - A) > 2 * noise
        pc = (m1 - mr) > noise
        print(f"[q] Gate: G1>A(2sigma) {'PASS' if pa else 'FAIL'}   "
              f"G1>rand(1sigma) {'PASS' if pc else 'FAIL'}", flush=True)
        result.update({"gate1_runs": g1_runs, "G1_mean": m1, "G1_rand_mean": mr,
                       "gate_pass_vs_A": pa, "gate_pass_vs_rand": pc,
                       "binomial_noise": noise})

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        print(f"[q] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
