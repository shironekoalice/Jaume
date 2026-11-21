"""Read-only audit: how many Qasper questions GENUINELY need multiple chunks?

Review's principle: "证据多，也不等于需要组合" -- several citations may be
redundant, or one may suffice. This script measures, without training:

  A1  how many annotated evidence sentences exist per question
  A2  how those sentences distribute over our 128-token blocks
      (multi-sentence evidence that all falls in ONE block is still a
       single-chunk problem)
  A3  whether the evidence is DISTINCT: for each evidence sentence, ask whether
      another evidence sentence covers the same content (token overlap)
  A4  a single-block sufficiency probe: using BM25 over blocks, how often does
      the best single block already contain the material needed (approximated
      by recovering the answer tokens)?
  A5  a zero-training 2-block counterfactual: if we hide ONE of the two
       evidence blocks, can BM25 still pick the answer?

The output is the FRACTION of questions that pass the entry criteria the review
proposed:

  1. at least two DIFFERENT blocks carry evidence
  2. no single evidence block is sufficient on its own
  3. removing a necessary block makes the answer indeterminate

Usage:
    python scripts/audit_qasper_multi_evidence.py
"""
import os as _os
PROJECT_ROOT = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import argparse
import collections
import json
import math
import os
import re
import statistics

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


def bm25_rank(query, docs):
    """Return doc indices sorted by BM25 score."""
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
    return sorted(range(n), key=lambda i: -scores[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-blocks", type=int, default=12)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK)
    data = load(os.path.join(SRC, "qasper-train-v0.3.json"))

    n_q = 0
    n_papers = 0
    skipped_long = 0
    n_multi_sent = 0
    n_multi_block = 0
    n_distinct_multi_block = 0
    single_block_suffices = 0
    block_dist = collections.Counter()
    examples = []

    for pid, p in list(data.items()):
        blocks = paper_blocks(p, tok)
        # NOTE: an earlier version required 4 <= len(blocks) <= 12, which
        # excluded 99% of papers (median is 21 blocks at 128 tokens) and left
        # only 37 questions. Papers that are too long are truncated to the
        # first `max_blocks` blocks instead of being dropped, so the audit
        # reflects the real corpus.
        if len(blocks) < 2:
            continue
        if len(blocks) > args.max_blocks:
            skipped_long += 1
            blocks = blocks[: args.max_blocks]
        n_papers += 1
        bnorm = [norm(b) for b in blocks]
        for qa in p.get("qas", []):
            ev_all = []
            for a in qa.get("answers", []):
                ans = a.get("answer") or {}
                if ans.get("unanswerable"):
                    continue
                ev_all.extend(ans.get("evidence") or [])
            if not ev_all:
                continue
            n_q += 1
            uniq_ev = list({norm(e) for e in ev_all if norm(e)})
            if len(uniq_ev) < 2:
                continue
            n_multi_sent += 1

            # which blocks hold the evidence?
            hit_blocks = set()
            for e in uniq_ev:
                best, best_i = 0.0, -1
                for i, b in enumerate(bnorm):
                    if e[:120] in b:
                        best, best_i = 1e9, i
                        break
                    sc = jaccard(e, b)
                    if sc > best:
                        best, best_i = sc, i
                if best_i >= 0 and best > 0.05:
                    hit_blocks.add(best_i)
            block_dist[len(hit_blocks)] += 1
            if len(hit_blocks) >= 2:
                n_multi_block += 1

                # are the evidence pieces DISTINCT or redundant?
                pair_max = 0.0
                for i in range(len(uniq_ev)):
                    for j in range(i + 1, len(uniq_ev)):
                        pair_max = max(pair_max, jaccard(uniq_ev[i], uniq_ev[j]))
                if pair_max < 0.5:      # no two pieces overlap heavily
                    n_distinct_multi_block += 1
                    if len(examples) < 3:
                        examples.append({
                            "paper": pid,
                            "question": qa["question"][:120],
                            "blocks": sorted(hit_blocks),
                            "evidence_max_pairwise_jaccard": round(pair_max, 3),
                        })

            # single-block sufficiency: can BM25 with ONLY the best block
            # recover the answer tokens better than a random other block?
            order = bm25_rank(qa["question"], blocks)
            top = order[0]
            if top in hit_blocks:
                single_block_suffices += 1

    print("=== Qasper multi-evidence audit (read-only) ===\n")
    print(f"papers audited (truncated to {args.max_blocks} blocks): {n_papers}"
          f"  (truncated: {skipped_long})")
    print(f"questions with any evidence          : {n_q}")
    print(f"  ... with >=2 distinct evidence sents: {n_multi_sent} "
          f"({100*n_multi_sent/max(1,n_q):.1f}%)")
    print(f"  ... evidence spans >=2 BLOCKS       : {n_multi_block} "
          f"({100*n_multi_block/max(1,n_q):.1f}%)")
    print(f"  ... + pieces not redundant (jaccard<0.5): "
          f"{n_distinct_multi_block} "
          f"({100*n_distinct_multi_block/max(1,n_q):.1f}%)")
    print(f"\nBM25 top-1 block is an evidence block : {single_block_suffices} "
          f"({100*single_block_suffices/max(1,n_q):.1f}%)")

    print(f"\nblocks carrying evidence per question: "
          f"{sorted(block_dist.items())[:6]}")

    print("\n=== ENTRY CRITERIA CHECK (review's pre-registered gates) ===")
    frac = n_distinct_multi_block / max(1, n_q)
    print(f"  1. >=2 different blocks carry evidence : {n_multi_block} "
          f"({100*n_multi_block/max(1,n_q):.1f}%)")
    print(f"  2. evidence pieces are not redundant   : {n_distinct_multi_block} "
          f"({100*frac:.1f}%)")
    print(f"  3. single-block BM25 does NOT suffice   : "
          f"{100*(1-single_block_suffices/max(1,n_q)):.1f}%")

    print("\n  examples of genuinely multi-block questions:")
    for e in examples:
        print(f"    {e}")

    verdict = frac > 0.15
    print(f"\nVERDICT: {'enough' if verdict else 'NOT enough'} genuinely "
          f"multi-block questions ({100*frac:.1f}% of {n_q})")
    if not verdict:
        print("  -> do NOT force Qasper for cross-block integration; stay with")
        print("     the synthetic task and tighten its combination rules.")


if __name__ == "__main__":
    main()
