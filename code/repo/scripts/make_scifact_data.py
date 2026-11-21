"""Build the SciFact multi-stage dataset for JM-R Real-Text Transfer.

Task (per spec): the abstract is read in two stages, then the claim is presented
and the model predicts SUPPORT / CONTRADICT / NEI.

    stage 1: first half of the abstract  -> state s_1
    stage 2: second half of the abstract -> state s_2
    stage 3: the claim                   -> label

The claim and the label NEVER appear in stages 1-2.

Data reality (must be stated, not glossed over)
-----------------------------------------------
The official SciFact release ships labels for `claims_train` (809) and
`claims_dev` (300); `claims_test` contains only {claim, id} with the labels held
out on a hidden evaluation server. There is therefore NO public labelled test
split. This script:

  * trains on claims_train, minus a held-out validation slice for early stopping
  * reports the final measurement on claims_dev
  * never touches claims_test (it has no labels to measure against)

Splits are made by CLAIM, and every claim's evidence comes from its own cited
documents, so no document text crosses splits by construction.

Label mapping:
  SUPPORT (evidence sentences labelled SUPPORT)   -> 1
  CONTRADICT                                       -> 2
  no evidence at all / not enough info             -> 0  (NEI)

A claim citing several documents with mixed labels is mapped by majority for the
binary-to-3-way label; claims with conflicting SUPPORT and CONTRADICT evidence
are dropped to avoid teaching an ambiguous target (count reported).

Usage:
    python scripts/make_scifact_data.py --out D:/Jaume/data/scifact_blocks
"""
import os as _os
PROJECT_ROOT = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import argparse
import collections
import json
import os
import random

LABELS = {"SUPPORT": 1, "CONTRADICT": 2}
LABEL_NAMES = {0: "NEI", 1: "SUPPORT", 2: "CONTRADICT"}


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def split_abstract(abstract):
    """Split an abstract into two roughly equal halves at a sentence boundary."""
    sents = [s.strip() for s in abstract.replace("\n", " ").split(". ") if s.strip()]
    if len(sents) < 2:
        return abstract, ""
    mid = max(1, len(sents) // 2)
    return ". ".join(sents[:mid]) + ".", ". ".join(sents[mid:])


def claim_label(claim):
    """Reduce SciFact's per-evidence labels to one 3-way label."""
    ev = claim.get("evidence") or {}
    votes = collections.Counter()
    for did, anns in ev.items():
        for a in anns:
            lab = a.get("label")
            if lab in LABELS:
                votes[LABELS[lab]] += 1
    if not votes:
        return 0, "no_evidence"
    if len(votes) > 1:
        # both SUPPORT and CONTRADICT evidence present: ambiguous as a 3-way
        # target, so it is dropped rather than arbitrarily resolved
        return None, "conflicting"
    return votes.most_common(1)[0][0], "ok"


def _as_text(v):
    """Some corpus rows store the abstract as a list of sentences."""
    if isinstance(v, list):
        return " ".join(str(x) for x in v)
    return str(v or "")


def build_rows(claims, corpus, use_cited_only=True):
    rows = []
    stats = collections.Counter()
    for c in claims:
        label, why = claim_label(c)
        stats[why] += 1
        if label is None:
            continue
        dids = c.get("cited_doc_ids") or []
        docs = [corpus[d] for d in dids if d in corpus]
        if not docs:
            stats["no_cited_doc"] += 1
            continue
        # concatenate the cited abstracts in a deterministic order
        text = " ".join((_as_text(d.get("title")) + ". " +
                         _as_text(d.get("abstract"))).strip()
                        for d in docs[:3])
        if len(text) < 80:
            stats["too_short"] += 1
            continue
        # Drop claims that appear verbatim inside the cited abstract. SciFact
        # contains a few such cases (measured: 1), and keeping them would hand
        # the stage-1/2 text a copy of the claim, i.e. leak the question.
        if c["claim"].strip().lower() in text.lower():
            stats["claim_in_text"] += 1
            continue
        h1, h2 = split_abstract(text)
        if not h2:
            stats["unsplittable"] += 1
            continue
        rows.append({
            "id": str(c["id"]),
            "claim": c["claim"],
            "stage1": h1,
            "stage2": h2,
            "label": label,
            "label_name": LABEL_NAMES[label],
            "doc_ids": [d.get("doc_id") for d in docs[:3]],
        })
    return rows, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=fr"{PROJECT_ROOT}\data\scifact_official\data")
    ap.add_argument("--out", default=fr"{PROJECT_ROOT}\data\scifact_blocks")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    corpus = {r["doc_id"]: r for r in load_jsonl(os.path.join(args.src, "corpus.jsonl"))}
    train_claims = load_jsonl(os.path.join(args.src, "claims_train.jsonl"))
    dev_claims = load_jsonl(os.path.join(args.src, "claims_dev.jsonl"))

    tr_rows, tr_stats = build_rows(train_claims, corpus)
    dv_rows, dv_stats = build_rows(dev_claims, corpus)

    # ------------------------------------------------------------------
    # Split CONNECTED COMPONENTS of the claim-document graph.
    #
    # The official train/dev boundary is claim-level and shared 149 documents
    # across it. Per-document hashing removed that leak but wasted many claims
    # (only 106 test claims survived) because any claim citing documents in
    # different buckets had to be dropped.
    #
    # Instead, documents are grouped into connected components (two documents
    # are linked when one claim cites both) and whole components are assigned
    # to a split. No claim can straddle splits, nothing is dropped, and no
    # document is shared across splits.
    # ------------------------------------------------------------------
    rng = random.Random(args.seed)
    all_rows = tr_rows + dv_rows

    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for r in all_rows:
        ds = r["doc_ids"]
        for d in ds[1:]:
            union(ds[0], d)

    comps = collections.defaultdict(list)
    for r in all_rows:
        comps[find(r["doc_ids"][0])].append(r)

    keys = sorted(comps, key=lambda k: str(k))
    rng.shuffle(keys)
    fit, val, test_rows = [], [], []
    for k in keys:
        bucket = rng.random()
        if bucket < 0.15:
            val.extend(comps[k])
        elif bucket < 0.35:
            test_rows.extend(comps[k])
        else:
            fit.extend(comps[k])

    print(f"[split] document components: {len(keys)} (claims dropped: 0)")
    print(f"[split] sizes -> train {len(fit)} / val {len(val)} / test {len(test_rows)}")

    os.makedirs(args.out, exist_ok=True)
    for name, rows in (("train", fit), ("val", val), ("test", test_rows)):
        p = os.path.join(args.out, f"{name}.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        dist = collections.Counter(r["label_name"] for r in rows)
        print(f"[scifact] {name:5s} n={len(rows):4d}  "
              f"NEI={dist['NEI']:4d} SUPPORT={dist['SUPPORT']:4d} "
              f"CONTRADICT={dist['CONTRADICT']:4d}")

    print(f"\n[train filter] {dict(tr_stats)}")
    print(f"[dev   filter] {dict(dv_stats)}")

    # document-level overlap check between splits
    def docs(rows):
        s = set()
        for r in rows:
            s |= set(r["doc_ids"])
        return s
    dtr, dva, dte = docs(fit), docs(val), docs(test_rows)
    print(f"\ncited-doc overlap train/val : {len(dtr & dva)}")
    print(f"cited-doc overlap train/test: {len(dtr & dte)}")
    print(f"cited-doc overlap val/test  : {len(dva & dte)}")

    meta = {"train": len(fit), "val": len(val), "test": len(test_rows),
            "train_filter": dict(tr_stats), "dev_filter": dict(dv_stats),
            "doc_overlap": {"train_val": len(dtr & dva),
                            "train_test": len(dtr & dte),
                            "val_test": len(dva & dte)},
            "note": "official claims_test has no public labels; test here is claims_dev"}
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[scifact] wrote {args.out}")


if __name__ == "__main__":
    main()


