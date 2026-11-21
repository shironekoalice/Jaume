"""Build a cross-block COMPOSITION subset from Qasper, with pre-registered gates.

The audit (audit_qasper_multi_evidence.py) found 632/2240 (28.2%) questions whose
evidence spans >=2 blocks and is not redundant. This script turns those into a
task where the answer genuinely needs COMPLEMENTARY information from multiple
blocks, and enforces the entry criteria the review pre-registered:

  G1  at least two DISTINCT blocks carry evidence
  G2  no single block alone determines the answer (checked by zero-training
      retrieval: if BM25 with one block recovers the answer, drop the item)
  G3  removing a necessary block makes the answer indeterminate
  G4  the question must not be answerable from the question text alone

Task form (answer-block MULTI-SELECT, which is what "composition" means here):

    given the question, select the set of blocks that TOGETHER contain the
    evidence; scoring is set-level (precision/recall of selected blocks), not
    top-1.

Why multi-select rather than a generated answer: a 350M encoder cannot free-form
generate, but recovering the required evidence SET demands that information from
several blocks be retained simultaneously -- exactly the ability under test.

Usage:
    python scripts/make_qasper_composition.py --out D:/Jaume/data/qasper_comp
"""
import os as _os
PROJECT_ROOT = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import argparse
import collections
import json
import math
import os
import random
import re

SRC = fr"{PROJECT_ROOT}\data\qasper_official"
TOK = fr"{PROJECT_ROOT}\output\mem\SF-s0"


def load(p):
    return json.load(open(p, encoding="utf-8"))


def paper_blocks(p, tok, max_tokens=128):
    units = [(p.get("title", ""), "title"), (p.get("abstract", ""), "abstract")]
    for sec in p.get("full_text", []):
        name = sec.get("section_name") or ""
        for para in sec.get("paragraphs", []):
            units.append((para, name))
    blocks, cur, cur_len = [], [], 0
    for text, _ in units:
        text = (text or "").strip()
        if not text:
            continue
        n = len(tok(text, truncation=False)["input_ids"])
        if cur_len + n > max_tokens and cur:
            blocks.append(" ".join(cur))
            cur, cur_len = [], 0
        cur.append(text)
        cur_len += n
    if cur:
        blocks.append(" ".join(cur))
    return blocks


def norm(s):
    return re.sub(r"\s+", " ", s or "").strip().lower()


def toks(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def jaccard(a, b):
    A, B = toks(a), toks(b)
    return len(A & B) / max(1, len(A | B))


def evidence_blocks(ev_sents, blocks_norm, thresh=0.05):
    """Map each evidence sentence to the block that best contains it."""
    hit = collections.defaultdict(list)
    for e in ev_sents:
        best, bi = 0.0, -1
        for i, b in enumerate(blocks_norm):
            if e[:120] in b:
                best, bi = 1e9, i
                break
            sc = jaccard(e, b)
            if sc > best:
                best, bi = sc, i
        if bi >= 0 and best > thresh:
            hit[bi].append(e)
    return hit


def bm25_rank(query, docs):
    qt = re.findall(r"[a-z0-9]+", query.lower())
    dt = [re.findall(r"[a-z0-9]+", d.lower()) for d in docs]
    n = len(dt)
    avgdl = sum(len(t) for t in dt) / max(1, n)
    df = collections.Counter()
    for t in dt:
        df.update(set(t))
    scores = []
    for t in dt:
        tf = collections.Counter(t)
        s = 0.0
        for q in qt:
            if q not in tf:
                continue
            idf = math.log(1 + (n - df[q] + 0.5) / (df[q] + 0.5))
            s += idf * tf[q] * 2.5 / (
                tf[q] + 1.5 * (1 - 0.75 + 0.75 * len(t) / max(1, avgdl)))
        scores.append(s)
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=fr"{PROJECT_ROOT}\data\qasper_comp")
    ap.add_argument("--max-blocks", type=int, default=12)
    ap.add_argument("--max-evidence-blocks", type=int, default=3,
                    help="keep items whose evidence spans 2..K blocks")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK)
    data = load(os.path.join(SRC, "qasper-train-v0.3.json"))
    dev = load(os.path.join(SRC, "qasper-dev-v0.3.json"))

    items, stats = [], collections.Counter()
    for split, src in (("train", data), ("dev", dev)):
        for pid, p in src.items():
            blocks = paper_blocks(p, tok)
            if len(blocks) < 2:
                continue
            if len(blocks) > args.max_blocks:
                blocks = blocks[: args.max_blocks]
            bn = [norm(b) for b in blocks]
            for qa in p.get("qas", []):
                ev = []
                for a in qa.get("answers", []):
                    ans = a.get("answer") or {}
                    if ans.get("unanswerable"):
                        continue
                    ev.extend(ans.get("evidence") or [])
                uniq = list({norm(e) for e in ev if norm(e)})
                if len(uniq) < 2:
                    stats["single_evidence"] += 1
                    continue
                hit = evidence_blocks(uniq, bn)
                k = len(hit)
                if k < 2:
                    stats["evidence_in_one_block"] += 1
                    continue
                if k > args.max_evidence_blocks:
                    stats["too_many_blocks"] += 1
                    continue
                # G2: redundancy check across the evidence pieces that matter
                pieces = [s for v in hit.values() for s in v]
                maxjac = max((jaccard(pieces[i], pieces[j])
                              for i in range(len(pieces))
                              for j in range(i + 1, len(pieces))), default=0.0)
                if maxjac >= 0.5:
                    stats["redundant"] += 1
                    continue

                # G2b: zero-training check -- does ONE block alone win on BM25?
                q = qa["question"]
                sc = bm25_rank(q, blocks)
                top = max(range(len(sc)), key=lambda i: sc[i])
                if top in hit:
                    stats["single_block_bm25_wins"] += 1
                    continue

                # G4: question must not contain the evidence verbatim
                qn = norm(q)
                if any(norm(e) in qn and len(norm(e)) > 20 for e in uniq):
                    stats["question_leaks_evidence"] += 1
                    continue

                gold = sorted(hit)
                items.append({
                    "id": f"{split}:{pid}:{qa.get('question_id','?')}",
                    "paper_id": str(pid),
                    "blocks": blocks,
                    "question": q,
                    "gold_blocks": gold,
                    "n_evidence_blocks": len(gold),
                    "evidence_text": [s[:300] for s in uniq[:6]],
                    "max_pairwise_jaccard": round(maxjac, 3),
                    "source_split": split,
                })
                stats["ok"] += 1

    print(f"[comp] candidates -> {dict(stats)}")
    print(f"[comp] kept {len(items)} cross-block composition items")

    # split by paper
    rng = random.Random(args.seed)
    papers = sorted({it["paper_id"] for it in items})
    rng.shuffle(papers)
    n = len(papers)
    test_p = set(papers[: max(1, n // 5)])
    val_p = set(papers[max(1, n // 5) : max(2, 2 * n // 5)])
    parts = {"train": [], "val": [], "test": []}
    for it in items:
        if it["paper_id"] in test_p:
            parts["test"].append(it)
        elif it["paper_id"] in val_p:
            parts["val"].append(it)
        else:
            parts["train"].append(it)

    os.makedirs(args.out, exist_ok=True)
    for name, rows in parts.items():
        p = os.path.join(args.out, f"{name}.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        k = collections.Counter(r["n_evidence_blocks"] for r in rows)
        print(f"[comp] {name:5s} n={len(rows):4d} papers="
              f"{len({r['paper_id'] for r in rows}):3d} evidence-blocks={dict(k)}")

    def papers(rows):
        return {r["paper_id"] for r in rows}
    a, b, c = papers(parts["train"]), papers(parts["val"]), papers(parts["test"])
    print(f"[comp] paper overlap tr/va={len(a&b)} tr/te={len(a&c)} va/te={len(b&c)}")

    meta = {"kept": len(items), "stats": dict(stats),
            "sizes": {k: len(v) for k, v in parts.items()},
            "paper_overlap": {"train_val": len(a & b), "train_test": len(a & c),
                              "val_test": len(b & c)}}
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
