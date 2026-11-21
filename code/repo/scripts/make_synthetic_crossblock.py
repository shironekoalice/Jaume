"""P1: a minimal synthetic task that CANNOT be solved without cross-chunk memory.

Design goal (per review): do not ask "is this task hard?" but "do we know WHY a
model answers correctly?". The task is built so that each single chunk is
provably insufficient:

    chunk A (entity -> attribute)   : "Project ORION uses the blue key.
                                       Project VEGA uses the red key."
    chunk B (entity -> target)      : "The one to inspect is project VEGA."
    query                           : "Which colour key should be used?"

    - seeing only A: unknown WHICH project is meant  -> unanswerable
    - seeing only B: unknown WHICH colour            -> unanswerable
    - only by retaining A and combining with B is the answer determined

Anti-shortcut measures (each one addresses a concrete failure seen earlier):

  * entity names sampled from a generated pool, so no fixed vocabulary to memorise
  * attribute values (colours) sampled independently of the entity
  * chunk ORDER shuffled per item, so position is uninformative
  * 4-8 distractor entity->attribute pairs inside the SAME chunk A, so mean
    pooling over A cannot single out the answer
  * the query never names the entity or the attribute
  * candidates are colour words only, so no candidate contains the entity name
  * train/test split uses DISJOINT entity pools and DISJOINT sentence templates

The module reports the same anti-shortcut baselines used in the JM-R probes
(`lexical_leakage`, `chunk_overlap_ceiling`) so a broken task is caught before
any 350M training budget is spent.

Usage:
    python scripts/make_synthetic_crossblock.py --n 600 --out D:/Jaume/data/synth_crossblock
"""
import argparse
import json
import os
import random

COLOURS = ["red", "blue", "green", "amber", "violet", "teal",
           "crimson", "indigo", "ochre", "azure", "scarlet", "olive"]

# Three disjoint template families: train / dev / test. Keeping them disjoint
# means a model cannot win by memorising a sentence pattern.
TEMPLATES = {
    "train": [
        "Project {e} uses the {c} key.",
        "{e} is assigned the {c} key.",
        "The {c} key belongs to project {e}.",
    ],
    "dev": [
        "{e} has been issued the {c} key.",
        "Records show project {e} holding the {c} key.",
    ],
    "test": [
        "The key issued to {e} is {c}.",
        "For project {e} the key colour is {c}.",
    ],
}

TARGET_TEMPLATES = {
    "train": ["Memo {m} names the asset to inspect.",
              "The asset in memo {m} must be examined."],
    "dev": ["The subject of review is the asset in memo {m}."],
    "test": ["The item under review is the asset in memo {m}."],
}

QUERIES = [
    "Which colour key should be used?",
    "What colour is called for here?",
    "State the colour that applies.",
]

SYLL_A = ["or", "ve", "ka", "lu", "mi", "ta", "ne", "zo",
          "ra", "pi", "su", "de", "fy", "gu", "ho", "ja"]
SYLL_B = ["ion", "ax", "eth", "orn", "ula", "yme", "ora", "isk",
          "and", "evo", "ith", "ossa", "urn", "ely", "ado", "ing"]
# a middle bank: 16 * 24 * 16 = 6144 reachable words, comfortably above the
# few thousand needed across all splits
SYLL_C = ["ar", "en", "il", "os", "un", "ek", "im", "ot",
          "al", "er", "is", "om", "uv", "eg", "ip", "ob",
          "ad", "ef", "ig", "ol", "up", "es", "in", "ov"]

# entity words are claimed across splits so train/dev/test never share one
_TAKEN = set()


def make_entities(rng, n, taken, max_tries=None):
    """Generate pronounceable pseudo-words that do not collide.

    A previous version looped until it had n unused words but never inserted
    the newly created words into the exclusion set it was testing against when
    the pool request exceeded the reachable space, so it spun forever. Here the
    loop is bounded and the requested count is clamped to what the syllable
    space can actually supply.
    """
    import itertools
    space = len(SYLL_A) * len(SYLL_B) * (len(SYLL_C) + 1)
    if n > space:
        raise ValueError(f"requested {n} entities but only {space} are reachable")
    if max_tries is None:
        max_tries = n * 500
    out = []
    tries = 0
    while len(out) < n and tries < max_tries:
        tries += 1
        # always three syllables: the two-syllable space alone is only 256 and
        # collides immediately once the pooled vocabulary grows past that
        w = rng.choice(SYLL_A) + rng.choice(SYLL_C) + rng.choice(SYLL_B)
        w = w.upper()
        if w not in taken:
            taken.add(w)
            out.append(w)
    if len(out) < n:
        raise RuntimeError(f"only produced {len(out)}/{n} unique entities")
    return out


def build(n, split, seed, easy=False):
    """Build n items for a split.

    `easy=True` reduces the task to its minimum form so that the question
    "is this learnable at all?" can be answered before anything about memory is
    concluded: only 2 entity-colour pairs in chunk A, 4 candidates, short memo
    ids and short chunks (well under the 128-token cap).

    Entity pools are made disjoint across splits by generating each split's
    words with a split-specific RNG seed and rejecting any word already claimed
    by an earlier split. Templates are also disjoint, so neither surface form
    nor vocabulary carries over from train to test.
    """
    rng = random.Random(seed + {"train": 0, "dev": 7919, "test": 104729}[split])
    # Each split needs at most n distinct target entities, plus enough
    # distractors to fill a chunk. The total across the three splits must stay
    # inside the reachable syllable space (16*16*16 = 4096), so the pool is
    # sized per split rather than from the pooled total.
    pool_n = min(max(n // 2 + 64, 64), 1200)
    pool = make_entities(rng, pool_n, _TAKEN)

    items = []
    enttmpl = TEMPLATES[split]
    tgttmpl = TARGET_TEMPLATES[split]
    for i in range(n):
        # the target entity and its attribute
        target = pool[(i * 2) % len(pool)]
        answer = rng.choice(COLOURS)
        # distractors: other entity->attribute pairs, none sharing the answer
        k = rng.randint(1, 1) if easy else rng.randint(4, 8)
        others = [e for e in pool if e != target]
        rng.shuffle(others)
        pairs = []
        used_colours = {answer}
        for e in others[:k]:
            c = rng.choice([c for c in COLOURS if c not in used_colours])
            used_colours.add(c)
            pairs.append((e, c))
        pairs.append((target, answer))
        rng.shuffle(pairs)  # the answer pair is at a random position

        # Chunk B must NOT name the entity: if it did, matching the entity name
        # between query and the answer table was enough to lower a contrastive
        # loss, and the model never learned to look up the colour. Instead
        # chunk B names a MEMO id, and chunk A binds memo id -> entity -> colour,
        # so the chain memo -> entity -> colour can only be followed by reading
        # chunk A.
        memo_ids = list(range(100, 100 + len(pairs)))
        rng.shuffle(memo_ids)
        bound = [(e, c, m) for (e, c), m in zip(pairs, memo_ids)]
        target_memo = next(m for e, c, m in bound if e == target)

        chunk_a = " ".join(
            rng.choice(enttmpl).format(e=e, c=c) + f" Filed under memo {m}."
            for e, c, m in bound)
        chunk_b = rng.choice(tgttmpl).format(m=target_memo)
        chunks = [chunk_a, chunk_b]
        rng.shuffle(chunks)  # order is uninformative

        # candidates: the answer plus colours that appear in chunk A
        present = [c for _, c in pairs]
        decoys = [c for c in present if c != answer]
        rng.shuffle(decoys)
        if easy:
            # 4 candidates total. Fill with colours drawn from the full palette
            # only if chunk A does not supply enough decoys, so the answer is
            # never uniquely implied by "the only option present in the text".
            extra = [c for c in COLOURS if c not in present]
            rng.shuffle(extra)
            cands = [answer] + decoys
            while len(cands) < 4 and extra:
                cands.append(extra.pop())
            rng.shuffle(cands)
        else:
            cands = [answer] + decoys[:7]
        rng.shuffle(cands)
        gold = cands.index(answer)
        cands = [c.capitalize() for c in cands]

        items.append({
            "chunks": chunks,
            "query": rng.choice(QUERIES),
            "candidates": cands,
            "gold": gold,
            "target_entity": target,
            "answer": answer,
        })
    return items


def toks(s):
    return set(w.lower().strip(".,?") for w in s.split())


def lexical_leakage(items):
    """Query-vs-candidate overlap accuracy (must be near chance)."""
    hits = 0
    for it in items:
        q = toks(it["query"])
        scores = [len(q & toks(c)) / max(1, len(q | toks(c)))
                  for c in it["candidates"]]
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        if scores[order[0]] > 0 and order[0] == it["gold"]:
            hits += 1
    return hits / len(items)


def chunk_overlap_ceiling(items):
    """Accuracy of a bag-of-words matcher given the LAST chunk.

    This is the baseline that killed the previous conflict probe (it scored
    1.00). If this number is high, the task is solvable by surface matching and
    is therefore worthless.
    """
    hits = 0
    for it in items:
        last = toks(it["chunks"][-1])
        scores = [len(last & toks(c)) / max(1, len(last | toks(c)))
                  for c in it["candidates"]]
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        if scores[order[0]] > 0 and order[0] == it["gold"]:
            hits += 1
    return hits / len(items)


def single_chunk_ceiling(items):
    """Best accuracy obtainable from ONE chunk alone (either chunk, any order).

    A task is only meaningful if neither chunk alone suffices. We approximate
    "informed guess from one chunk" with a keyword matcher on each chunk plus a
    constant-guess fallback, and report the best of them.
    """
    best = 0.0
    for idx in range(len(items[0]["chunks"])):
        hits = 0
        for it in items:
            side = toks(it["chunks"][idx])
            scores = [len(side & toks(c)) / max(1, len(side | toks(c)))
                      for c in it["candidates"]]
            order = sorted(range(len(scores)), key=lambda i: -scores[i])
            if scores[order[0]] > 0 and order[0] == it["gold"]:
                hits += 1
        best = max(best, hits / len(items))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--out", default="D:/Jaume/data/synth_crossblock")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--easy", action="store_true",
                    help="minimum-difficulty variant: 2 entity-colour pairs in "
                         "chunk A and 4 candidates, to establish whether the "
                         "task is learnable at all")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    report = {}
    for split, n in (("train", args.n), ("dev", args.n // 4),
                     ("test", args.n // 4)):
        items = build(n, split, args.seed, easy=args.easy)
        path = os.path.join(args.out, f"{split}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        n_cand = [len(it["candidates"]) for it in items]
        rand = sum(1.0 / c for c in n_cand) / len(n_cand)
        report[split] = {
            "n": n,
            "avg_candidates": round(sum(n_cand) / len(n_cand), 2),
            "random": round(rand, 4),
            "lexical_leakage": round(lexical_leakage(items), 4),
            "chunk_overlap_ceiling": round(chunk_overlap_ceiling(items), 4),
            "single_chunk_ceiling": round(single_chunk_ceiling(items), 4),
        }
        print(f"[synth] {split:5s} n={n:5d} cand={report[split]['avg_candidates']:.1f} "
              f"random={report[split]['random']:.3f} "
              f"leak={report[split]['lexical_leakage']:.3f} "
              f"chunk_ceil={report[split]['chunk_overlap_ceiling']:.3f} "
              f"single_chunk={report[split]['single_chunk_ceiling']:.3f}")

    with open(os.path.join(args.out, "validity.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print()
    print("TASK VALIDITY CHECK (all must hold before spending training budget):")
    ok = True
    for split, r in report.items():
        cond = (r["lexical_leakage"] <= 0.10
                and r["chunk_overlap_ceiling"] <= 0.25
                and r["single_chunk_ceiling"] <= 0.25)
        print(f"  {split:5s} leak<=0.10 {r['lexical_leakage']:.3f} | "
              f"chunk_ceil<=0.25 {r['chunk_overlap_ceiling']:.3f} | "
              f"single_chunk<=0.25 {r['single_chunk_ceiling']:.3f} "
              f"-> {'PASS' if cond else 'FAIL'}")
        ok = ok and cond
    print(f"\n  overall: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("  -> the task is solvable without cross-chunk memory; rebuild it "
              "before training anything.")


if __name__ == "__main__":
    main()
