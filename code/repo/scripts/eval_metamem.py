"""Meta-memory write-selection evaluation (docs/memory-metamem-spec.md).

Frozen-backbone heads vs lexical/trivial baselines on the question-evidence
proxy label: can the model rank blocks by "worth remembering" WITHOUT seeing
the query? All judgement criteria are frozen in the spec; this script only
implements them.

Usage:
    python scripts/make_metamem_data.py
    python scripts/eval_metamem.py --checkpoint D:/项目/Jaume/350m-mlm --seeds 1 2 3
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jaume.device import resolve_device, describe  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.abspath(os.path.join(REPO, "..", ".."))

SECTION_KEEP = 12
SPLIT_ORDER = ("train", "val", "test")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.path.join(PROJECT_ROOT, "350m-mlm"))
    ap.add_argument("--data", default=os.path.join(PROJECT_ROOT, "data", "metamem", "papers.jsonl"))
    ap.add_argument("--output", default=os.path.join(PROJECT_ROOT, "output", "metamem"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--budget", type=float, default=0.25, help="keep fraction for U1/U2")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--encode-batch", type=int, default=64)
    return ap.parse_args()


# ---------------------------------------------------------------- data loading

def load_papers(path):
    papers = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            papers.append(json.loads(line))
    papers.sort(key=lambda p: (SPLIT_ORDER.index(p["split"]), p["paper_id"]))
    return papers


def encode_blocks(papers, args, device):
    """Encode every block once with the frozen encoder; cache to npz."""
    cache = os.path.join(args.output, "block_emb.npz")
    n_blocks = sum(len(p["blocks"]) for p in papers)
    if os.path.exists(cache):
        z = np.load(cache)
        if z["emb"].shape[0] == n_blocks:
            print(f"[metamem] using cached embeddings {z['emb'].shape} from {cache}")
            return z["emb"]
        print(f"[metamem] cache size mismatch ({z['emb'].shape[0]} != {n_blocks}), re-encoding")

    from jaume.inference.encode import JaumeEmbedding

    enc = JaumeEmbedding.from_pretrained(args.checkpoint, device=device)
    texts = [b for p in papers for b in p["blocks"]]
    t0 = time.time()
    emb = enc.encode(texts, batch_size=args.encode_batch).numpy()
    dt = time.time() - t0
    print(f"[metamem] encoded {len(texts)} blocks in {dt:.0f}s "
          f"({len(texts) / dt:.1f} blocks/s) dim={emb.shape[1]}")
    np.savez_compressed(cache, emb=emb)
    return emb


# ---------------------------------------------------------------- features

TOKEN_RE = re.compile(r"[a-z0-9]+")


def section_group(section):
    return (section or "").split(":::")[0].strip().lower() or "(none)"


def build_trivial(papers):
    """Handcrafted per-block features: position/length/section/has-digits.

    Fitted on train papers only; returned as aligned float matrices.
    """
    groups = {}
    for p in papers:
        if p["split"] != "train":
            continue
        for s in p["sections"]:
            g = section_group(s[0] if isinstance(s, (list, tuple)) else s)
            groups[g] = groups.get(g, 0) + 1
    top = sorted(groups, key=groups.get, reverse=True)[:SECTION_KEEP]
    g2i = {g: i for i, g in enumerate(top)}

    rows = []
    for p in papers:
        n = len(p["blocks"])
        for i, b in enumerate(p["blocks"]):
            sec = p["sections"][i] if i < len(p["sections"]) else None
            g = section_group(sec[0] if isinstance(sec, (list, tuple)) else sec)
            onehot = [0.0] * (len(top) + 1)
            onehot[g2i.get(g, len(top))] = 1.0
            rows.append([
                i, i / max(1, n - 1), float(len(b)), 1.0 if i == 0 else 0.0,
                1.0 if i == n - 1 else 0.0, 1.0 if any(c.isdigit() for c in b) else 0.0,
            ] + onehot)
    X = np.asarray(rows, dtype=np.float32)
    return X


def build_labels(papers):
    ys, qidx = [], []
    for p in papers:
        ev = set(p["evidence_union"])
        ys.extend([1 if i in ev else 0 for i in range(len(p["blocks"]))])
        qidx.extend([len(ys) - len(p["blocks"])] * len(p["blocks"]))
    return np.asarray(ys, dtype=np.float32), np.asarray(qidx)


def doc_slices(papers):
    out, start = [], 0
    for p in papers:
        out.append((p, start, start + len(p["blocks"])))
        start += len(p["blocks"])
    return out


# ---------------------------------------------------------------- torch heads

class LinearHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, 1)

    def forward(self, x):
        return self.lin(x).squeeze(-1)


class MlpHead(nn.Module):
    def __init__(self, d, h=256, p=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def per_doc_auroc(scores, ys, slices):
    from sklearn.metrics import roc_auc_score, average_precision_score

    aurocs, aprs, skipped = [], [], 0
    for _, a, b in slices:
        y, s = ys[a:b], scores[a:b]
        if y.min() == y.max():
            skipped += 1
            continue
        aurocs.append(roc_auc_score(y, s))
        aprs.append(average_precision_score(y, s))
    return float(np.mean(aurocs)), float(np.mean(aprs)), skipped


def train_head(Xtr, ytr, paper_slices_tr, Xva, yva, slices_va, pos_weight, seed,
               kind="mlp", lr=1e-3, wd=1e-4, epochs=50, patience=5, batch_papers=32,
               device="cpu"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = (LinearHead(Xtr.shape[1]) if kind == "linear" else MlpHead(Xtr.shape[1])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    Xtr_t = torch.from_numpy(Xtr).to(device)
    ytr_t = torch.from_numpy(ytr).to(device)
    Xva_t = torch.from_numpy(Xva).to(device)

    starts = [a for _, a, _ in paper_slices_tr]
    best, best_state, bad = -1.0, None, 0
    for epoch in range(epochs):
        model.train()
        order = np.random.permutation(len(starts))
        for i in range(0, len(order), batch_papers):
            idx = order[i:i + batch_papers]
            sel = np.concatenate([np.arange(paper_slices_tr[j][1], paper_slices_tr[j][2]) for j in idx])
            sel_t = torch.from_numpy(sel).to(device)
            logits = model(Xtr_t[sel_t])
            loss = lossf(logits, ytr_t[sel_t])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            va = model(Xva_t).cpu().numpy()
        a, _, _ = per_doc_auroc(va, yva, slices_va)
        if a > best + 1e-4:
            best, best_state, bad = a, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def bm25_scores(blocks, query, k1=1.5, b=0.75):
    toks = [TOKEN_RE.findall(b.lower()) for b in blocks]
    q = TOKEN_RE.findall(query.lower())
    n = len(blocks)
    avgdl = sum(len(t) for t in toks) / max(1, n)
    df = {}
    for t in toks:
        for w in set(t):
            df[w] = df.get(w, 0) + 1
    out = []
    for t in toks:
        tf = {}
        for w in t:
            tf[w] = tf.get(w, 0) + 1
        score = 0.0
        for w in q:
            if w not in tf:
                continue
            idf = np.log(1 + (n - df.get(w, 0) + 0.5) / (df.get(w, 0) + 0.5))
            score += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(t) / max(1e-6, avgdl)))
        out.append(score)
    return np.asarray(out)


# ---------------------------------------------------------------- selection utility

def selection_metrics(scores, ys, slices, budget, queries_per_paper):
    """U1: union-evidence coverage@budget; U2: questions with evidence fully kept."""
    u1, u2n, u2d = [], 0, 0
    for (p, a, b) in slices:
        n = b - a
        k = max(1, int(n * budget))
        kept = set(np.argsort(-scores[a:b])[:k] + a)
        ev = set(a + i for i in p["evidence_union"])
        if ev:
            u1.append(len(kept & ev) / len(ev))
        for q in queries_per_paper[p["paper_id"]]:
            qe = set(a + i for i in q["evidence"])
            if not qe:
                continue
            u2d += 1
            u2n += int(qe <= kept)
    return float(np.mean(u1)) if u1 else float("nan"), (u2n / u2d if u2d else float("nan"))


# ---------------------------------------------------------------- main

def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    device = resolve_device()
    print(f"[metamem] device={device} ({describe(device)})  seeds={args.seeds}")

    papers = load_papers(args.data)
    slices_all = doc_slices(papers)
    split_arr = np.repeat(
        [SPLIT_ORDER.index(p["split"]) for p, _, _ in slices_all],
        [b - a for _, a, b in slices_all],
    )
    split_mask = {s: split_arr == SPLIT_ORDER.index(s) for s in SPLIT_ORDER}
    emb = encode_blocks(papers, args, device)
    trivial = build_trivial(papers)
    ys, _ = build_labels(papers)
    queries = {p["paper_id"]: p["questions"] for p, _, _ in slices_all}

    # global pos_weight from train
    ytr_all = ys[split_mask["train"]]
    pos_weight = float((ytr_all == 0).sum() / max(1, (ytr_all == 1).sum()))
    print(f"[metamem] train pos_weight={pos_weight:.2f} "
          f"(pos {int(ytr_all.sum())} / {len(ytr_all)})")

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def rows_of(mask):
        return np.where(mask)[0]

    tr_rows, va_rows, te_rows = rows_of(split_mask["train"]), rows_of(split_mask["val"]), rows_of(split_mask["test"])

    # subset scores are aligned to the split-local row ranges, so all per-doc
    # slices used against them are re-based to the same local space
    slices_tr = [(p, a - tr_rows[0], b - tr_rows[0]) for p, a, b in slices_all if p["split"] == "train"]
    slices_va = [(p, a - va_rows[0], b - va_rows[0]) for p, a, b in slices_all if p["split"] == "val"]
    slices_te = [(p, a - te_rows[0], b - te_rows[0]) for p, a, b in slices_all if p["split"] == "test"]

    # ---- lexical / trivial baselines (sklearn)
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)
    texts = [b for p in papers for b in p["blocks"]]
    Xtf = tfidf.fit_transform([texts[i] for i in tr_rows])
    Xtf_va, Xtf_te = tfidf.transform([texts[i] for i in va_rows]), tfidf.transform([texts[i] for i in te_rows])
    ytr_s, yva_s, yte_s = ys[tr_rows], ys[va_rows], ys[te_rows]

    scaler = StandardScaler().fit(trivial[tr_rows])
    Xtv_tr, Xtv_va, Xtv_te = scaler.transform(trivial[tr_rows]), scaler.transform(trivial[va_rows]), scaler.transform(trivial[te_rows])

    def fit_lr(Xtr_m, ytr_m, Xva_m, yva_m):
        best, best_c = -1.0, 1.0
        for c in (0.5, 1.0, 4.0):
            m = LogisticRegression(C=c, max_iter=2000)
            m.fit(Xtr_m, ytr_m)
            a, _, _ = per_doc_auroc(m.predict_proba(Xva_m)[:, 1], yva_m, slices_va)
            if a > best:
                best, best_c = a, c
        m = LogisticRegression(C=best_c, max_iter=2000).fit(Xtr_m, ytr_m)
        return m, best_c

    print("[metamem] fitting B1 (tfidf LR) / B2 (trivial LR) ...")
    b1_model, b1_c = fit_lr(Xtf, ytr_s, Xtf_va, yva_s)
    b2_model, b2_c = fit_lr(Xtv_tr, ytr_s, Xtv_va, yva_s)
    b1_te = b1_model.predict_proba(Xtf_te)[:, 1]
    b2_te = b2_model.predict_proba(Xtv_te)[:, 1]
    b2_va = b2_model.predict_proba(Xtv_va)[:, 1]

    # ---- neural heads, 3 seeds
    results = {"config": vars(args), "b1_C": b1_c, "b2_C": b2_c, "pos_weight": pos_weight,
               "arms": {}, "seeds": {}}
    neural_specs = {
        "H1_linear": ("linear", emb),
        "H2_mlp": ("mlp", emb),
        "H3_mlp_trivial": ("mlp", np.concatenate([emb, trivial], axis=1)),
    }
    for seed in args.seeds:
        seed_res = {}
        for name, (kind, X) in neural_specs.items():
            model = train_head(
                X[tr_rows], ytr_s, slices_tr, X[va_rows], yva_s, slices_va,
                pos_weight, seed, kind=kind, device="cpu",
            )
            model.eval()
            with torch.no_grad():
                te = model(torch.from_numpy(X[te_rows])).numpy().astype(np.float64)
                va = model(torch.from_numpy(X[va_rows])).numpy().astype(np.float64)
            seed_res[name] = {"test_scores": te, "val_scores": va}
            a, ap, sk = per_doc_auroc(te, yte_s, slices_te)
            print(f"[metamem] seed {seed} {name}: test AUROC {a:.4f} AUPRC {ap:.4f} (skipped {sk})")
        rng = np.random.default_rng(seed)
        seed_res["R0_random"] = {"test_scores": rng.random(len(te_rows))}
        results["seeds"][seed] = seed_res

    # ---- arm selection on val (mean over seeds), then test stats
    val_auroc = {name: np.mean([per_doc_auroc(r[name]["val_scores"], yva_s, slices_va)[0]
                                for r in results["seeds"].values()])
                 for name in neural_specs}
    best_neural = max(val_auroc, key=val_auroc.get)
    print(f"[metamem] val AUROC: " + ", ".join(f"{k}={v:.4f}" for k, v in val_auroc.items())
          + f"  -> best={best_neural}")

    def eval_arm(scores, seed):
        a, ap, sk = per_doc_auroc(scores, yte_s, slices_te)
        u1, u2 = selection_metrics(scores, yte_s, slices_te, args.budget, queries)
        return {"auroc": a, "auprc": ap, "skipped_docs": sk, "U1": u1, "U2": u2}

    fixed = {
        "B1_lexical_lr": {s: b1_te for s in args.seeds},
        "B2_trivial_lr": {s: b2_te for s in args.seeds},
        "R0_random": {s: results["seeds"][s]["R0_random"]["test_scores"] for s in args.seeds},
        f"NEURAL_{best_neural}": {s: results["seeds"][s][best_neural]["test_scores"] for s in args.seeds},
    }
    results["arms"] = {}
    for name, per_seed in fixed.items():
        results["arms"][name] = {}
        for s in args.seeds:
            results["arms"][name][s] = eval_arm(per_seed[s], s)

    # U0 reference: BM25 with the test query (not a judged arm)
    u0 = {"U1": [], "U2": []}
    for (p, a, b) in slices_te:
        for q in queries[p["paper_id"]]:
            s = bm25_scores(p["blocks"], q["question"])
            u1, u2 = None, None
            # per-paper budget on this query's scores
            n = b - a
            k = max(1, int(n * args.budget))
            kept = set(np.argsort(-s)[:k] + a)
            ev = set(a + i for i in p["evidence_union"])
            if ev:
                u0["U1"].append(len(kept & ev) / len(ev))
            qe = set(a + i for i in q["evidence"])
            if qe:
                u0["_u2n"] = u0.get("_u2n", 0) + int(qe <= kept)
                u0["_u2d"] = u0.get("_u2d", 0) + 1
    results["arms"]["U0_bm25_with_query"] = {
        "reference_only": True,
        "U1": float(np.mean(u0["U1"])), "U2": u0.get("_u2n", 0) / max(1, u0.get("_u2d", 1)),
    }

    # ---- frozen criteria: paired diffs neural - B1
    from sklearn.metrics import roc_auc_score

    b1_by_doc = {}
    for (p, a, b) in slices_te:
        y, s1 = yte_s[a:b], b1_te[a:b]
        b1_by_doc[p["paper_id"]] = roc_auc_score(y, s1) if y.min() != y.max() else None

    diffs = []  # [seed][doc]
    for s in args.seeds:
        s_by_doc = {}
        neural_scores = fixed[f"NEURAL_{best_neural}"][s]
        for (p, a, b) in slices_te:
            y, sn = yte_s[a:b], neural_scores[a:b]
            s_by_doc[p["paper_id"]] = roc_auc_score(y, sn) if y.min() != y.max() else None
        diffs.append([s_by_doc[pid] - b1_by_doc[pid]
                      for pid in sorted(b1_by_doc)
                      if b1_by_doc[pid] is not None and s_by_doc[pid] is not None])
    diffs = np.asarray(diffs)  # [n_seeds, n_docs]
    seed_means = diffs.mean(axis=1)
    # directional: the preregistered question is whether NEURAL beats lexical,
    # so "同号" is read as all-positive (a significant negative is a failed M1)
    signs_positive = bool(np.all(seed_means > 0))
    rng = np.random.default_rng(0)
    boots = np.empty(args.bootstrap)
    n_docs = diffs.shape[1]
    for i in range(args.bootstrap):
        idx = rng.integers(0, n_docs, n_docs)
        boots[i] = diffs[:, idx].mean()
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    m1_pass = signs_positive and ci[0] > 0

    # U1 paired diffs
    u1_diffs = []
    for s in args.seeds:
        n_scores = fixed[f"NEURAL_{best_neural}"][s]
        d = []
        for (p, a, b) in slices_te:
            n = b - a
            k = max(1, int(n * args.budget))
            ev = set(a + i for i in p["evidence_union"])
            if not ev:
                continue
            kept_n = set(np.argsort(-n_scores[a:b])[:k] + a)
            kept_b = set(np.argsort(-b1_te[a:b])[:k] + a)
            d.append(len(kept_n & ev) / len(ev) - len(kept_b & ev) / len(ev))
        u1_diffs.append(d)
    u1_diffs = np.asarray(u1_diffs)
    u1_seed_means = u1_diffs.mean(axis=1)
    u1_signs_positive = bool(np.all(u1_seed_means > 0))
    boots_u1 = np.empty(args.bootstrap)
    for i in range(args.bootstrap):
        idx = rng.integers(0, u1_diffs.shape[1], u1_diffs.shape[1])
        boots_u1[i] = u1_diffs[:, idx].mean()
    ci_u1 = (float(np.percentile(boots_u1, 2.5)), float(np.percentile(boots_u1, 97.5)))
    m2_pass = u1_signs_positive and ci_u1[0] > 0

    b2_auroc = np.mean([results["arms"]["B2_trivial_lr"][s]["auroc"] for s in args.seeds])
    neural_auroc = np.mean([results["arms"][f"NEURAL_{best_neural}"][s]["auroc"] for s in args.seeds])
    g1_violated = bool(b2_auroc >= neural_auroc)

    results["criteria"] = {
        "best_neural": best_neural,
        "M1_auroc_diff_seed_means": seed_means.tolist(),
        "M1_bootstrap_ci": ci, "M1_pass": bool(m1_pass),
        "M2_U1_diff_seed_means": u1_seed_means.tolist(),
        "M2_bootstrap_ci": ci_u1, "M2_pass": bool(m2_pass),
        "G1_trivial_vs_neural": {"B2": b2_auroc, "neural": neural_auroc, "trivial_wins": g1_violated},
    }

    # raw score arrays stay in memory only; the json report keeps metrics
    results["seeds"] = {s: {k: {"n_scores": int(np.asarray(v["test_scores"]).shape[0])}
                            for k, v in r.items()} for s, r in results["seeds"].items()}

    with open(os.path.join(args.output, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=float)

    print("\n=== metamem test results (macro over docs) ===")
    for name in results["arms"]:
        per = results["arms"][name]
        if name == "U0_bm25_with_query":
            print(f"  {name:22s} [reference] U1={per['U1']:.3f} U2={per['U2']:.3f}")
            continue
        a = np.mean([per[s]["auroc"] for s in args.seeds])
        ap = np.mean([per[s]["auprc"] for s in args.seeds])
        u1 = np.mean([per[s]["U1"] for s in args.seeds])
        u2 = np.mean([per[s]["U2"] for s in args.seeds])
        print(f"  {name:22s} AUROC {a:.4f}  AUPRC {ap:.4f}  U1@{args.budget:.0%} {u1:.3f}  U2 {u2:.3f}")
    c = results["criteria"]
    print(f"\nM1 (neural−lexical AUROC): seed means {['%.4f' % v for v in c['M1_auroc_diff_seed_means']]}"
          f" CI [{c['M1_bootstrap_ci'][0]:.4f}, {c['M1_bootstrap_ci'][1]:.4f}]  pass={c['M1_pass']}")
    print(f"M2 (neural−lexical U1):    seed means {['%.4f' % v for v in c['M2_U1_diff_seed_means']]}"
          f" CI [{c['M2_bootstrap_ci'][0]:.4f}, {c['M2_bootstrap_ci'][1]:.4f}]  pass={c['M2_pass']}")
    print(f"G1 trivial-vs-neural:      B2 {c['G1_trivial_vs_neural']['B2']:.4f}"
          f" vs neural {c['G1_trivial_vs_neural']['neural']:.4f}  trivial_wins={g1_violated}")


if __name__ == "__main__":
    main()
