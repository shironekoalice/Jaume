"""Jaume-Memory task C: cross-input memory probes (plan §5.C / §7).

Six probes built from controllable synthetic documents; each document is split
into chunks and fed through the model in order, carrying the slot state (JM-R)
or encoding each chunk independently (control).

  1. far_retrieval    a fact in chunk 1, question in the last chunk
  2. aggregation      facts spread over chunks, question needs several of them
  3. conflict         a fact is stated then corrected (100 -> 120)
  4. distractor       irrelevant chunks inserted between fact and question
  5. isolation        document A then document B; B must not be polluted by A
  6. order            the same chunks shuffled

A probe item is answered by ranking candidate answers with cosine similarity
against the final state vector (JM-R) or the mean of independent chunk vectors
(control). Accuracy = fraction where the gold answer ranks first.

Usage:
    python scripts/eval_memory_crossinput.py --model out/B6/final --probes all
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jaume.inference.encode import JaumeEmbedding

TOPICS = [
    ("the server", "runs on port 8080"), ("the backup", "happens every Sunday"),
    ("the API key", "is stored in vault"), ("the database", "uses PostgreSQL 14"),
    ("the cache", "expires in 30 minutes"), ("the report", "is sent on Friday"),
    ("the gateway", "listens on 10.0.0.1"), ("the queue", "holds 5000 jobs"),
    ("the model", "was trained for 3 epochs"), ("the cluster", "has 12 nodes"),
    ("the budget", "is 48000 dollars"), ("the deadline", "is 15 November"),
    ("the index", "is rebuilt nightly"), ("the log", "is kept for 90 days"),
    ("the alert", "goes to ops@example.com"), ("the timeout", "is 250 milliseconds"),
]
FILLER = [
    "The weather today is mild and slightly windy in the coastal region.",
    "Several birds were observed near the pond during the morning survey.",
    "A delivery truck parked outside the warehouse for about twenty minutes.",
    "The committee reviewed the printed agenda before the meeting started.",
    "Paint on the northern wall has begun to peel after the recent rain.",
]


def enc(model, texts, batch_size=32, max_length=128):
    outs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            e = model.tokenizer(chunk, padding=True, truncation=True,
                                max_length=max_length, return_tensors="pt").to(model.device)
            emb, _ = model.model(e["input_ids"], e["attention_mask"])
            outs.append(emb.float().cpu())
    return torch.cat(outs)


# ------------------------------------------------------------------ probes
def build_far_retrieval(n=120):
    """A fact appears in an early chunk; the query is deliberately VAGUE so it
    cannot be answered from the query text alone.

    Design note: the first version asked "About the server, what should I
    know?" while the candidates each restated their subject, so a bag-of-words
    match between query and candidate picked the answer without any document
    reading (leakage 1.000). Here the query is "What should I know about the
    configuration?" -- it carries no subject -- so the only way to select the
    right candidate is through the document/chunk representation.
    """
    items = []
    for i in range(n):
        subj, fact = TOPICS[i % len(TOPICS)]
        others = [f"{s} {f}." for s, f in TOPICS if s != subj]
        chunks = [f"{subj.capitalize()} {fact}."] + FILLER[:3] + ["What is the configuration?"]
        cands = [f"{subj.capitalize()} {fact}."] + others[:7]
        items.append({"chunks": chunks,
                      "query": "What should I know about the configuration?",
                      "candidates": cands, "gold": 0})
    return items


def build_aggregation(n=120):
    items = []
    for i in range(n):
        a, b = TOPICS[(2 * i) % len(TOPICS)], TOPICS[(2 * i + 1) % len(TOPICS)]
        chunks = [f"{a[0].capitalize()} {a[1]}.", FILLER[0], f"{b[0].capitalize()} {b[1]}.", FILLER[1]]
        gold_txt = f"{a[0].capitalize()} {a[1]}; {b[0]} {b[1]}."
        cands = [gold_txt] + [f"{s.capitalize()} {f}." for s, f in TOPICS if s not in (a[0], b[0])][:7]
        items.append({"chunks": chunks, "query": "Summarise the configuration.",
                      "candidates": cands, "gold": 0})
    return items


def build_conflict(n=120):
    """Fact stated then corrected: the newer value must win.

    Design note (this probe failed three times before it was usable):
      v1 contrasted "costs 100 dollars" vs "costs 120 dollars". Both are
      near-identical strings, so the ranking was decided by a tokenizer
      artefact, not by the model's state (scores were 0.00 or 1.00 depending on
      the arm -- pure noise).
      v2 labelled the candidates "Current ..." vs "Outdated ...", which made the
      task solvable by keyword matching: B0 (mean pooling, no memory at all)
      scored 1.000. That measures the candidate text, not the memory.
      v3 wrote both candidates in the same register differing only in the final
      digit. Measured on B0: every memory strategy scored 0.000 -- the two
      candidate embeddings were near-collinear so argmax was a tie broken by
      position. An unsatisfiable-by-similarity probe measures nothing.

    v4 (this version): the update and the stale fact describe *different*
    values in *different* words ("adopts the Meridian protocol" vs "adopts the
    Bellweather protocol"), so the two candidates are genuinely different
    strings. The distractor is the stale statement reworded as if still
    current, so only a reader that tracks which update came last can pick
    correctly. The query deliberately does NOT name the subject, so the answer
    is not recoverable by query-candidate overlap (checked by lexical_leakage).
    """
    items = []
    for i in range(n):
        subj, _ = TOPICS[i % len(TOPICS)]
        old, new = "Bellweather", "Meridian"
        chunks = [f"{subj.capitalize()} adopts the {old} protocol.", FILLER[0],
                  f"Following a revision, {subj} adopts the {new} protocol."]
        cands = [f"The current protocol is {new}.",
                 f"The current protocol is {old}."] + \
                [f"The current protocol is {p}."
                 for p in ("Halcyon", "Norwood", "Ravenna",
                           "Tessaly", "Umbra", "Vireo")][:6]
        items.append({"chunks": chunks,
                      "query": "Which one of these is still in force at the end?",
                      "candidates": cands, "gold": 0})
    return items


def build_distractor(n=120):
    """A fact, then irrelevant chunks. The query is vaguer than the candidate.

    The query must NOT name the subject, otherwise the answer is recoverable by
    query-candidate lexical overlap without reading anything (measured leakage
    0.900 in the first version).
    """
    items = []
    for i in range(n):
        subj, fact = TOPICS[i % len(TOPICS)]
        chunks = [f"{subj.capitalize()} {fact}."] + [FILLER[j % len(FILLER)] for j in range(i % 4 + 2)]
        cands = [f"{subj.capitalize()} {fact}."] + [f"{s.capitalize()} {f}." for s, f in TOPICS if s != subj][:7]
        items.append({"chunks": chunks,
                      "query": "What should I know about the configuration?",
                      "candidates": cands, "gold": 0})
    return items


def build_isolation(n=120):
    """Doc A then doc B: the answer must come from B, not A.

    The query is vague ("Which setting is currently in effect?") and the
    distractor candidate is A's fact, so picking correctly requires the state to
    reflect B rather than being polluted by A -- not just keyword matching.
    """
    items = []
    for i in range(n):
        a = TOPICS[i % len(TOPICS)]
        b = TOPICS[(i + 5) % len(TOPICS)]
        chunks = [f"{a[0].capitalize()} {a[1]}.", FILLER[0],
                  f"{b[0].capitalize()} {b[1]}.", FILLER[1]]
        cands = [f"{b[0].capitalize()} {b[1]}.", f"{a[0].capitalize()} {a[1]}."] + \
                [f"{s.capitalize()} {f}." for s, f in TOPICS if s not in (a[0], b[0])][:6]
        items.append({"chunks": chunks,
                      "query": "Which setting is currently in effect?",
                      "candidates": cands, "gold": 0})
    return items


def build_order(n=120):
    """Same chunks, shuffled order. The query must not name the subject."""
    items = []
    for i in range(n):
        subj, fact = TOPICS[i % len(TOPICS)]
        chunks = [FILLER[0], f"{subj.capitalize()} {fact}.", FILLER[1], FILLER[2]]
        shuffled = [chunks[(i + k) % len(chunks)] for k in range(len(chunks))]
        cands = [f"{subj.capitalize()} {fact}."] + [f"{s.capitalize()} {f}." for s, f in TOPICS if s != subj][:7]
        items.append({"chunks": shuffled,
                      "query": "What should I know about the configuration?",
                      "candidates": cands, "gold": 0})
    return items


PROBES = {
    "far_retrieval": build_far_retrieval,
    "aggregation": build_aggregation,
    "conflict": build_conflict,
    "distractor": build_distractor,
    "isolation": build_isolation,
    "order": build_order,
}


def lexical_leakage(items, margin=0.08):
    """Fraction of items where the gold candidate is the BEST token-overlap
    match for the query *and* beats the runner-up by a clear margin.

    A probe whose gold answer wins on surface overlap alone is not measuring
    memory. `margin` matters: when two candidates are near-identical (e.g. the
    same sentence with 100 vs 120) the overlap difference is tiny and the
    "winner" is a tokenizer artefact, so a bare argmax would be noise. Requiring
    a margin only flags genuine shortcuts.

    This check exists because an earlier `conflict` probe was fully solvable by
    keyword matching (its candidates contained "Current" / "Outdated") and
    reported 1.000 for a model with no memory at all.
    """
    def toks(s):
        return set(w.lower().strip(".,?:") for w in s.split())

    hits = 0
    for it in items:
        q = toks(it["query"])
        scores = []
        for c in it["candidates"]:
            t = toks(c)
            scores.append(len(q & t) / max(1, len(q | t)))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        best = order[0]
        runner = scores[order[1]] if len(order) > 1 else 0.0
        if best == it["gold"] and (scores[best] - runner) >= margin:
            hits += 1
    return hits / len(items)


def chunk_overlap_upper_bound(items, margin=0.0):
    """How well a bag-of-words matcher can do using the CHUNKS, not the query.

    This is the baseline `lexical_leakage` misses. `lexical_leakage` compares
    the query against the candidates, but the leak that actually broke the
    conflict probe was candidate-vs-CHUNK overlap: the gold candidate restated
    a sentence that appeared verbatim in the memory chunks, so a model that
    stores raw text (or simply matches surface forms) wins without doing any
    cross-chunk reasoning. Measured on B0 that path scored query_only 1.000.

    The returned number is the accuracy of a keyword matcher that is given the
    chunks and picks the candidate sharing the most surface tokens with the
    LAST chunk (the most recent fact). Any arm whose memory score does not beat
    this number has demonstrated nothing beyond surface matching.
    """
    def toks(s):
        return set(w.lower().strip(".,?:") for w in s.split())

    hits = 0
    for it in items:
        last = toks(it["chunks"][-1])
        scores = [len(last & toks(c)) / max(1, len(last | toks(c)))
                  for c in it["candidates"]]
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        best = order[0]
        runner = scores[order[1]] if len(order) > 1 else 0.0
        if best == it["gold"] and (scores[best] - runner) >= margin:
            hits += 1
    return hits / len(items)

RECURRENT_ARMS = ("recurrent_memory", "recurrent_memory_gate")

# The four cross-input strategies that must all be compared (plan section 4).
# Without the first three, a good score for the recurrent arm cannot be
# attributed to memory transfer rather than to chunk averaging or to the
# candidate text itself.
STRATEGIES = (
    "independent",   # 1. encode each chunk independently, then average
    "prev_only",     # 2. keep only the PREVIOUS chunk's representation
    "reset",         # 3. recurrent weights, but reset state at every chunk
    "carried",       # 4. full slot state carried across chunks (+ gate)
    "concat",        # 5. concatenate neighbouring chunks and encode in one go
)


def _encode(model, texts):
    return enc(model, texts)


def precompute_chunks(model, items, max_length=128, batch_size=64):
    """Encode every chunk of every item in one batched pass (stateless).

    `independent` / `reset` / `prev_only` / `concat` all reduce to arithmetic on
    chunk-level embeddings with NO state. Doing that arithmetic is cheap; the
    cost was one tokenizer call plus one batch-of-one forward per chunk, which
    left the GPU idle (1500 items x 2 chunks x 4 strategies = 12000 tiny
    forwards). This pre-pass does it in batched forwards instead, and the
    per-strategy functions below read from the table.

    `carried` cannot use the table: it must run the real recurrent walk so the
    state actually propagates. It stays serial, which is the honest cost of
    measuring cross-input memory.
    """
    flat, index = [], []
    for i, it in enumerate(items):
        for j, ch in enumerate(it["chunks"]):
            flat.append(ch)
            index.append((i, j))
    if not flat:
        return {}
    table = {}
    with torch.no_grad():
        for s in range(0, len(flat), batch_size):
            part = flat[s : s + batch_size]
            e = model.tokenizer(part, padding=True, truncation=True,
                                max_length=max_length,
                                return_tensors="pt").to(model.device)
            emb, _ = model.model(e["input_ids"], e["attention_mask"])
            emb = emb.float().cpu()
            for k, (i, j) in enumerate(index[s : s + batch_size]):
                table[(i, j)] = emb[k : k + 1]
    return table


def _doc_vector(model, it, strategy, table=None, idx=None):
    """Returns an L2-normalised document vector for the given strategy."""
    chunks = it["chunks"]
    recurrent = model.model.config.pooling in RECURRENT_ARMS

    def chunk_emb(j):
        """Chunk j's stateless embedding, from the table when available."""
        if table is not None and (idx, j) in table:
            return table[(idx, j)]
        e = model.tokenizer([chunks[j]], padding=True, truncation=True,
                            max_length=128, return_tensors="pt").to(model.device)
        with torch.no_grad():
            p, _, _ = model.model.encode(e["input_ids"], e["attention_mask"], None)
        return p.float().cpu()

    if strategy == "independent":
        # P0 audit fix: this MUST use the same per-chunk, state-free walk as
        # `carried` does on its first step, so the only difference between the
        # two strategies is whether the state is passed. Using the precomputed
        # table is safe here because `precompute_chunks` encodes each chunk in
        # isolation, exactly like the serial walk did.
        pooled = [chunk_emb(j) for j in range(len(chunks))]
        v = torch.cat(pooled, dim=0).mean(dim=0, keepdim=True)
        return torch.nn.functional.normalize(v, dim=-1)

    if strategy == "prev_only":
        # only the last non-question chunk represents the document
        body = chunks[:-1] if len(chunks) > 1 else chunks
        j = len(body) - 1
        return torch.nn.functional.normalize(chunk_emb(j), dim=-1)

    if strategy == "reset":
        # same walk as `independent`: no state is passed at any step.
        # Kept as a separate name because the plan requires it, but note that
        # for a recurrent pooler this is the "state disabled" condition.
        return _doc_vector(model, it, "independent", table, idx)

    if strategy == "concat":
        # the whole document in one forward pass (no chunking at all)
        joined = " ".join(chunks)
        embs = _encode(model, [joined])
        return torch.nn.functional.normalize(embs, dim=-1)

    # strategy == "carried"
    if not recurrent:
        # a pooler with no state cannot carry anything; delegate explicitly so
        # this equality is documented rather than an accident of duplicated code
        return _doc_vector(model, it, "independent", table, idx)
    state = None
    for ch in chunks:
        e = model.tokenizer([ch], padding=True, truncation=True,
                            max_length=128, return_tensors="pt").to(model.device)
        with torch.no_grad():
            _, _, state = model.model.encode(e["input_ids"], e["attention_mask"], state)
    v = state.mean(dim=1).float().cpu()
    return torch.nn.functional.normalize(v, dim=-1)


def run_probe(model, items, strategy, mode="combined", cache=None, table=None):
    """Top-1 accuracy over the candidate list for one cross-input strategy.

    Scoring modes:
      query_only  : rank by similarity to the query embedding only (the honest
                    floor any embedding model already achieves)
      memory_only : rank by similarity to the document vector only
      combined    : the two L2-normalised cosine scores summed

    `cache` avoids re-encoding queries and candidates per strategy: those
    embeddings do not depend on the strategy, but calling this function once
    per strategy re-encoded them every time (4x wasted work, which made a
    1500-item sweep take tens of minutes).
    """
    if cache is None:
        cache = {}
    correct = 0
    for i, it in enumerate(items):
        key = it.get("_id", i)
        doc_vec = _doc_vector(model, it, strategy, table, i)

        if ("q", key) not in cache:
            cache[("q", key)] = torch.nn.functional.normalize(
                enc(model, [it["query"]]), dim=-1)
            cache[("c", key)] = torch.nn.functional.normalize(
                enc(model, it["candidates"]), dim=-1)
        qv = cache[("q", key)]
        cv = cache[("c", key)]
        q_score = (qv @ cv.T).squeeze(0)
        m_score = (doc_vec @ cv.T).squeeze(0)

        if mode == "query_only":
            combined = q_score
        elif mode == "memory_only":
            combined = m_score
        else:
            combined = q_score + m_score
        if int(combined.argmax()) == it["gold"]:
            correct += 1
    return correct / len(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--probes", default="all")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--strategies", default="independent,prev_only,reset,carried,concat",
                    help="comma list of cross-input strategies to compare")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model = JaumeEmbedding.from_pretrained(args.model)
    arm = model.model.config.pooling
    recurrent = arm in RECURRENT_ARMS
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    print(f"[memory-C] arm={arm} recurrent={recurrent} device={model.device}")
    print(f"[memory-C] strategies: {strategies}")

    names = list(PROBES) if args.probes == "all" else [p.strip() for p in args.probes.split(",")]
    results = {"model": args.model, "arm": arm, "recurrent": recurrent,
               "strategies": strategies, "probes": {}}
    for name in names:
        items = PROBES[name](args.n)
        for i, it in enumerate(items):
            it["_id"] = (name, i)
        cache = {}
        row = {}
        # memory_only under each strategy -- this is the column that actually
        # isolates what the cross-input state contributes
        for st in strategies:
            row[st] = round(run_probe(model, items, st, "memory_only", cache), 4)
        row["query_only"] = round(run_probe(model, items, "independent",
                                            "query_only", cache), 4)
        import random
        rng = random.Random(0)
        row["random"] = round(sum(rng.randrange(len(it["candidates"])) == it["gold"]
                                  for it in items) / len(items), 4)
        row["n"] = len(items)
        row["lexical_leakage"] = round(lexical_leakage(items), 4)
        row["chunk_overlap_ceiling"] = round(chunk_overlap_upper_bound(items), 4)
        results["probes"][name] = row
        cols = "  ".join(f"{st} {row[st]:.3f}" for st in strategies)
        print(f"[memory-C] {name:15s} {cols}  | query_only {row['query_only']:.3f} "
              f"(rand {row['random']:.3f}, leak {row['lexical_leakage']:.3f}, "
              f"chunk-ceiling {row['chunk_overlap_ceiling']:.3f})")

    out_path = args.out or os.path.join(args.model, "memory_crossinput.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[memory-C] wrote {out_path}")


if __name__ == "__main__":
    main()
