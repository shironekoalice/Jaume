"""Independent-holdout evaluation for Jaume-Memory (plan §2 / §5).

Implements the pre-registered protocol:
  * holdout sets: SICK-R (EN STS), BEIR SciFact (retrieval), C-MTEB STS (ZH);
    dev sets: STS-B / STS22 (never used for decisions, only for debugging);
  * Spearman for STS (plan §5.A), Recall@k / MRR@10 / nDCG@10 for retrieval;
  * bootstrap 1000 resamples -> 95% CI (plan §2 rule 3);
  * E9: the same run reports every Matryoshka dim.

Usage:
    python scripts/evaluate_holdout.py --model out/B3/final \
        --tasks sickr --bootstrap 1000
    python scripts/evaluate_holdout.py --model out/B3/final --tasks sickr,scifact
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jaume.inference.encode import JaumeEmbedding

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "..", "data")


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def _rank(v):
    order = torch.argsort(v)
    ranks = torch.empty_like(v)
    ranks[order] = torch.arange(len(v), dtype=torch.float32)
    return ranks


def spearman(x, y):
    rx, ry = _rank(x), _rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    return (rx * ry).sum() / (rx.norm() * ry.norm() + 1e-12)


def bootstrap_ci(pred, gold, fn, n=1000, seed=0):
    """Percentile bootstrap over pairs; returns (point, lo, hi)."""
    g = torch.Generator().manual_seed(seed)
    n_obs = len(gold)
    point = fn(pred, gold).item()
    vals = []
    for _ in range(n):
        idx = torch.randint(0, n_obs, (n_obs,), generator=g)
        if len(torch.unique(idx)) < 2:
            continue
        vals.append(fn(pred[idx], gold[idx]).item())
    v = torch.tensor(vals)
    return point, float(torch.quantile(v, 0.025)), float(torch.quantile(v, 0.975))


def recall_at_k(sims, gold_idx, k):
    topk = torch.topk(sims, k, dim=-1).indices
    return float((topk == gold_idx.unsqueeze(-1)).any(dim=-1).float().mean())


def mrr_at_k(sims, gold_idx, k=10):
    topk = torch.topk(sims, k, dim=-1).indices
    hit = (topk == gold_idx.unsqueeze(-1))
    ranks = torch.where(hit, torch.arange(1, k + 1, device=sims.device).float(), torch.zeros_like(sims[:, :k]))
    has = hit.any(dim=-1).float()
    return float((has / ranks.clamp(min=1e-9).where(hit, torch.full_like(ranks, 1e9)).min(dim=-1).values).mean())


def ndcg_at_k(sims, gold_idx, k=10):
    topk = torch.topk(sims, k, dim=-1).indices
    hit = (topk == gold_idx.unsqueeze(-1)).float()
    dcg = (hit / torch.log2(torch.arange(2, k + 2, device=sims.device).float())).sum(dim=-1)
    return float((dcg / 1.0).mean())


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------
def load_sts(name):
    paths = {
        "sickr": ("sickr-test.jsonl", "test"),
        "stsb": ("stsbenchmark_sts.jsonl", "dev"),
        "sts22": ("sts22-crosslingual-sts.jsonl", "dev"),
    }
    fname, role = paths[name]
    path = os.path.join(DATA_DIR, fname)
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    s1 = [r["sentence1"] for r in rows]
    s2 = [r["sentence2"] for r in rows]
    gold = torch.tensor([float(r["score"]) for r in rows])
    return s1, s2, gold, role


def load_retrieval(name):
    """BEIR-style retrieval. Prefers the real corpus + qrels files written by
    prepare_holdout.py; falls back to the {query, positive} pair file."""
    qrels_path = os.path.join(DATA_DIR, f"{name}_qrels.jsonl")
    corpus_path = os.path.join(DATA_DIR, f"{name}_corpus.jsonl")
    if os.path.exists(qrels_path) and os.path.exists(corpus_path):
        corpus, cids = [], []
        for l in open(corpus_path, encoding="utf-8"):
            r = json.loads(l)
            corpus.append(r["text"])
            cids.append(str(r["id"]))
        pos = {str(i): t for i, t in enumerate(corpus)}
        idx_of = {c: i for i, c in enumerate(cids)}
        queries, gold = [], []
        for l in open(qrels_path, encoding="utf-8"):
            r = json.loads(l)
            qid = str(r["qid"])
            qt = QUERY_TEXT.get(qid)
            if qt is None:
                continue
            for did in r["dids"]:
                j = idx_of.get(str(did))
                if j is not None:
                    queries.append(qt)
                    gold.append(j)
        return corpus, queries, torch.tensor(gold), "holdout"

    path = os.path.join(DATA_DIR, f"{name}.jsonl")
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    corpus, queries, gold = [], [], []
    for r in rows:
        corpus.append(r["positive"])
        queries.append(r["query"])
        gold.append(len(corpus) - 1)
    return corpus, queries, torch.tensor(gold), "holdout"


QUERY_TEXT = {}


def _load_query_text(name):
    path = os.path.join(DATA_DIR, f"{name}_queries.jsonl")
    if os.path.exists(path):
        for l in open(path, encoding="utf-8"):
            r = json.loads(l)
            QUERY_TEXT[str(r["id"])] = r["text"]


# --------------------------------------------------------------------------
def encode_all(model, texts, dims=None, batch_size=64, max_length=256):
    outs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            try:
                enc = model.tokenizer(
                    chunk, padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt",
                ).to(model.device)
            except Exception:
                # a few SciFact abstracts exceed the tokenizer model limit
                enc = model.tokenizer(
                    [t[:4000] for t in chunk], padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt",
                ).to(model.device)
            if dims is None:
                emb, _ = model.model(enc["input_ids"], enc["attention_mask"])
            else:
                out, _ = model.model.forward_matryoshka(enc["input_ids"], enc["attention_mask"])
                emb = out[dims]
            outs.append(emb.float().cpu())
    return torch.cat(outs, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", default="sickr")
    ap.add_argument("--dims", default="", help="comma list for E9, e.g. 128,256,512")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--out", default=None, help="write results json here")
    args = ap.parse_args()

    model = JaumeEmbedding.from_pretrained(args.model)
    print(f"[holdout] {args.model}  device={model.device}")

    dims_list = [int(d) for d in args.dims.split(",") if d.strip()]
    results = {"model": args.model, "tasks": {}}

    for task in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        if task in ("sickr", "stsb", "sts22"):
            s1, s2, gold, role = load_sts(task)
            if args.max_samples:
                s1, s2, gold = s1[: args.max_samples], s2[: args.max_samples], gold[: args.max_samples]
            for dims in (dims_list or [None]):
                e1 = encode_all(model, s1, dims, args.batch_size)
                e2 = encode_all(model, s2, dims, args.batch_size)
                sims = (e1 * e2).sum(dim=-1)
                point, lo, hi = bootstrap_ci(sims, gold, spearman, n=args.bootstrap)
                key = task + (f"@{dims}D" if dims else "")
                results["tasks"][key] = {
                    "metric": "spearman", "role": role, "n": len(gold),
                    "value": round(point, 4), "ci95": [round(lo, 4), round(hi, 4)],
                }
                print(f"[holdout] {key:>14} spearman {point:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  n={len(gold)} ({role})")
        else:
            _load_query_text(task)
            corpus, queries, gold_idx, role = load_retrieval(task)
            if args.max_samples:
                corpus = corpus[: args.max_samples]
                queries = queries[: args.max_samples]
                gold_idx = gold_idx[: args.max_samples]
            for dims in (dims_list or [None]):
                ce = encode_all(model, corpus, dims, args.batch_size)
                qe = encode_all(model, queries, dims, args.batch_size)
                sims = qe @ ce.T
                key = task + (f"@{dims}D" if dims else "")
                results["tasks"][key] = {
                    "role": role, "n": len(queries), "corpus": len(corpus),
                    "recall@10": round(recall_at_k(sims, gold_idx, 10), 4),
                    "recall@100": round(recall_at_k(sims, gold_idx, 100), 4),
                    "mrr@10": round(mrr_at_k(sims, gold_idx, 10), 4),
                    "ndcg@10": round(ndcg_at_k(sims, gold_idx, 10), 4),
                }
                r = results["tasks"][key]
                print(f"[holdout] {key:>14} R@10 {r['recall@10']:.4f} R@100 {r['recall@100']:.4f} "
                      f"MRR@10 {r['mrr@10']:.4f} nDCG@10 {r['ndcg@10']:.4f}  n={len(queries)} ({role})")

    out_path = args.out or os.path.join(args.model, "holdout_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        # merge instead of overwrite: a later run may add tasks (or dims) to an
        # existing report, and silently dropping earlier results corrupts the
        # multi-task comparison.
        try:
            prev = json.load(open(out_path, encoding="utf-8"))
            prev.setdefault("tasks", {}).update(results["tasks"])
            results = prev
        except Exception:
            pass
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[holdout] wrote {out_path}")


if __name__ == "__main__":
    main()
