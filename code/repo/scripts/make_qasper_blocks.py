"""Build Qasper blocks for the long-document memory experiment.

Task (Q-A: evidence-block selection):
    the model reads all blocks of a paper through the memory mechanism, then is
    given the question and must pick WHICH BLOCK contains the answer evidence.

    accuracy@1 and MRR over the block list
    the position of the evidence block gives the "distance" used by S3

Design decisions forced by measurement
--------------------------------------
Qasper papers are long: median 5169 tokens = 21 blocks of 256. 99.7% exceed the
encoder's 512-token window, so a "concatenate everything" concat baseline would
be truncated for almost every paper and is therefore NOT a fair reference.

Per review decision, concat is instead defined as: encode every block
INDEPENDENTLY, then aggregate (mean). It sees exactly the same blocks as
`carried`; the only difference is whether the state flows between them.
That isolates memory as the single varying factor.

Distance definition for S3
--------------------------
A question is scored against every block to pick the evidence block. The
"distance" a memory mechanism must bridge is how many blocks come between the
evidence block and the END of the paper (the state is built sequentially, so
evidence near the end is cheap for any sequential reader, evidence near the
start must survive across many updates).

    d = n_blocks - 1 - evidence_idx      (0 = last block, large = far back)

S3 asks whether carried - reset grows as d grows.

Usage:
    python scripts/make_qasper_blocks.py --out D:/Jaume/data/qasper_blocks
"""
import os as _os
PROJECT_ROOT = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import argparse
import collections
import json
import os
import random
import re

MAX_BLOCK_TOKENS = 128


def load(p):
    return json.load(open(p, encoding="utf-8"))


def norm(s):
    return re.sub(r"\s+", " ", s or "").strip().lower()


def paper_blocks(p, tok, max_tokens=MAX_BLOCK_TOKENS):
    """Flatten the paper in reading order and cut it into fixed-size blocks."""
    units = []
    units.append((p.get("title", ""), "title"))
    units.append((p.get("abstract", ""), "abstract"))
    for sec in p.get("full_text", []):
        name = sec.get("section_name") or ""
        for para in sec.get("paragraphs", []):
            units.append((para, name))
    if not units:
        return []

    blocks, cur, cur_len, cur_secs = [], [], 0, set()
    for text, sec in units:
        text = (text or "").strip()
        if not text:
            continue
        n = len(tok(text, truncation=False)["input_ids"])
        if cur_len + n > max_tokens and cur:
            blocks.append({"text": " ".join(cur), "sections": sorted(cur_secs)})
            cur, cur_len, cur_secs = [], 0, set()
        cur.append(text)
        cur_len += n
        cur_secs.add(sec)
    if cur:
        blocks.append({"text": " ".join(cur), "sections": sorted(cur_secs)})
    return blocks


def answer_of(qa):
    """Pick one annotator answer that has evidence; return (text, evidence)."""
    for a in qa.get("answers", []):
        ans = a.get("answer") or {}
        if ans.get("unanswerable"):
            continue
        ev = ans.get("evidence") or []
        ffa = (ans.get("free_form_answer") or "").strip()
        spans = ans.get("extractive_spans") or []
        if ev:
            return ffa or (" ".join(spans) if spans else ""), ev
    return None, None


def build(rows, tok, min_blocks=4, max_blocks=40):
    """Return items: each has blocks, question, evidence index, distance."""
    items = []
    stats = collections.Counter()
    for pid, p in rows.items():
        blocks = paper_blocks(p, tok)
        if not (min_blocks <= len(blocks) <= max_blocks):
            stats["block_count_out_of_range"] += 1
            continue
        btexts = [norm(b["text"]) for b in blocks]
        for qa in p.get("qas", []):
            ans_text, ev = answer_of(qa)
            if not ans_text or not ev:
                stats["no_answer_or_evidence"] += 1
                continue
            # locate the evidence: the block containing the most evidence text
            hit = -1
            best = 0.0
            for i, bt in enumerate(btexts):
                for e in ev:
                    ne = norm(e)
                    if not ne:
                        continue
                    if ne[:120] in bt:
                        score = len(ne)
                    else:
                        # fall back to token overlap
                        et = set(ne.split())
                        bt_t = set(bt.split())
                        score = len(et & bt_t) / max(1, len(et)) * 50
                    if score > best:
                        best, hit = score, i
            if hit < 0 or best < 5:
                stats["evidence_not_found"] += 1
                continue
            # drop cases where the answer already appears in the question: the
            # question would hand over the answer text (measured: 1 case)
            qn = norm(qa["question"])
            ang = norm(ans_text)
            if ang and len(ang) > 12 and ang in qn:
                stats["answer_in_question"] += 1
                continue
            n = len(blocks)
            items.append({
                "id": f"{pid}:{qa.get('question_id','?')}",
                "paper_id": str(pid),
                "blocks": [b["text"] for b in blocks],
                "sections": [b["sections"] for b in blocks],
                "question": qa["question"],
                "answer": ans_text[:400],
                "evidence_idx": hit,
                "n_blocks": n,
                "distance": n - 1 - hit,
            })
            stats["ok"] += 1
    return items, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=fr"{PROJECT_ROOT}\data\qasper_official")
    ap.add_argument("--tok", default=fr"{PROJECT_ROOT}\output\mem\SF-s0")
    ap.add_argument("--out", default=fr"{PROJECT_ROOT}\data\qasper_blocks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.20)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tok)

    tr_rows = load(os.path.join(args.src, "qasper-train-v0.3.json"))
    dv_rows = load(os.path.join(args.src, "qasper-dev-v0.3.json"))

    tr_items, tr_stats = build(tr_rows, tok)
    dv_items, dv_stats = build(dv_rows, tok)
    print(f"[qasper] train items={len(tr_items)} stats={dict(tr_stats)}")
    print(f"[qasper] dev   items={len(dv_items)} stats={dict(dv_stats)}")

    # split by PAPER, pooling train+dev given the small item count
    rng = random.Random(args.seed)
    papers = sorted({it["paper_id"] for it in tr_items + dv_items})
    rng.shuffle(papers)
    n = len(papers)
    n_test = int(n * args.test_frac)
    n_val = int(n * args.val_frac)
    test_p = set(papers[:n_test])
    val_p = set(papers[n_test : n_test + n_val])

    fit = [it for it in tr_items + dv_items if it["paper_id"] not in test_p | val_p]
    val = [it for it in tr_items + dv_items if it["paper_id"] in val_p]
    test = [it for it in tr_items + dv_items if it["paper_id"] in test_p]

    os.makedirs(args.out, exist_ok=True)
    for name, items in (("train", fit), ("val", val), ("test", test)):
        p = os.path.join(args.out, f"{name}.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        if not items:
            print(f"[qasper] {name}: EMPTY")
            continue
        dist = [it["distance"] for it in items]
        nb = [it["n_blocks"] for it in items]
        print(f"[qasper] {name:5s} n={len(items):5d} papers={len({i['paper_id'] for i in items}):4d} "
              f"blocks/paper={sum(nb)/len(nb):.1f} "
              f"distance median={sorted(dist)[len(dist)//2]} max={max(dist)}")

    # document-level disjointness
    def papers_of(items):
        return {it["paper_id"] for it in items}
    a, b, c = papers_of(fit), papers_of(val), papers_of(test)
    print(f"\npaper overlap train/val={len(a&b)} train/test={len(a&c)} "
          f"val/test={len(b&c)}")

    # distance strata for S3
    print("\ndistance strata (n_blocks - 1 - evidence_idx):")
    for name, items in (("train", fit), ("test", test)):
        s = collections.Counter()
        for it in items:
            d = it["distance"]
            s["d=0" if d == 0 else ("d=1-3" if d <= 3 else
                                    ("d=4-7" if d <= 7 else "d>=8"))] += 1
        print(f"  {name}: {dict(s)}")

    meta = {"train": len(fit), "val": len(val), "test": len(test),
            "train_stats": dict(tr_stats), "dev_stats": dict(dv_stats),
            "paper_overlap": {"train_val": len(a & b), "train_test": len(a & c),
                              "val_test": len(b & c)},
            "max_block_tokens": MAX_BLOCK_TOKENS}
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[qasper] wrote {args.out}")


if __name__ == "__main__":
    main()

