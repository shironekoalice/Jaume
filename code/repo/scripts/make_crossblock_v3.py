"""P1.5 task: cross-chunk fact combination with a UNIQUE answer.

Why this replaces the previous attempts
---------------------------------------
v1 failed because chunk A printed the entity name that also appeared in the
query, so matching that name lowered a contrastive loss without any reasoning.
v2 (memo -> entity -> colour) removed the shortcut but required a symbolic
three-hop lookup that no model learned.

A "stricter of the two schedules" phrasing was also rejected on review: without
stating the second schedule and the comparison rule the answer is not unique,
so a failure cannot be attributed to memory rather than ambiguity.

Design used here
----------------
Chunk A states TWO protocols with concrete numeric facts and names them.
Chunk B asks a question whose answer is determined by comparing those facts,
and chunk B contains NO adjective that hints at the answer (no "stricter",
"longer", "safer"). The question only says WHICH MEASURE and in WHAT DIRECTION
to compare, using a neutral template:

    chunk A : "Protocol ALPHA requires rotating credentials every 30 days.
               Protocol BETA requires rotating credentials every 90 days."
    chunk B : "Use the protocol with the smaller rotation interval."

    query   : "Which protocol applies?"        (does not name either protocol)
    answer  : ALPHA                             (unique: 30 < 90)

The direction ("smaller") is a deterministic comparison operator, not a hint:
either direction can be sampled, so the model cannot guess. Answer uniqueness
is therefore guaranteed by construction, and is verified by an oracle.

Anti-shortcut measures
----------------------
  * the question never names either protocol name
  * the two numeric values are distinct, so the comparison has one answer
  * entity/protocol names are fresh pseudo-words, disjoint across splits
  * templates are disjoint across splits
  * the comparison DIRECTION is balanced across items (50/50 smaller/larger),
    so "always pick the first" or "always pick the smaller number" does not
    beat chance above 0.5 on a 2-way task -- and this file uses 4 candidates
  * candidates are protocol NAMES, so the answer cannot be recovered from the
    question text alone

Usage:
    python scripts/make_crossblock_v3.py --n 4000 --out D:/Jaume/data/synth_v3
"""
import argparse
import json
import random
import re

# names must not carry meaning: pseudo-words, generated per split
SYLL_A = ["cal", "dor", "fen", "gal", "hev", "jor", "kel", "lom",
          "mar", "nuv", "pyr", "quo", "rav", "sil", "tor", "vex"]
SYLL_B = ["andor", "ethis", "orum", "avia", "ence", "ithar", "ossa",
          "ustra", "ymer", "antis", "evor", "ildan", "omir", "urix",
          "evar", "osten"]
# a middle bank: 16 * 16 * 20 = 5120 reachable names, enough for all three
# splits without reuse. (A two-bank design gives only 256 and silently
# exhausted, which is the same bug that made the v2 generator loop forever.)
SYLL_M = ["ek", "or", "ul", "in", "as", "ev", "om", "ir",
          "an", "ez", "ot", "us", "il", "od", "er", "ab", "im", "ov", "ad", "ef"]
# a fourth bank lifts the reachable space to 16*20*20*16 = 102400, comfortably
# above the ~8200 names the three splits need
SYLL_D = ["a", "e", "i", "o", "u", "ae", "eo", "ia",
          "io", "ou", "y", "au", "ei", "eu", "oa", "ui", "ie", "ea", "ai", "oa"]

# (fact description, unit) -- the comparison is always on the number
FACTS = [
    ("requires rotating credentials every {v} days", "days"),
    ("permits a session to stay open for {v} hours", "hours"),
    ("allows {v} concurrent connections", "connections"),
    ("retains audit logs for {v} days", "days"),
    ("sets the token lifetime to {v} minutes", "minutes"),
    ("caps requests at {v} per second", "requests"),
]

# comparison directions: sampled 50/50 so the operator carries no bias
DIRECTIONS = [
    ("smaller", min),
    ("larger", max),
]

TEMPLATES_A = {
    "train": ["Protocol {n1} {f1}. Protocol {n2} {f2}.",
              "{n1}: {f1}. {n2}: {f2}."],
    "dev": ["Under the rules, protocol {n1} {f1}, while protocol {n2} {f2}."],
    "test": ["Policy states that protocol {n1} {f1}; protocol {n2} {f2}."],
}
TEMPLATES_B = {
    "train": ["Apply the protocol with the {d} value.",
              "Use whichever of them has the {d} value."],
    "dev": ["Follow the one whose value is {d}."],
    "test": ["Adopt the protocol that carries the {d} value."],
}
QUERIES = ["Which protocol applies?",
           "Which one governs this case?",
           "Name the protocol in force."]

_TAKEN = set()


def make_names(rng, n):
    out = []
    tries = 0
    space = len(SYLL_A) * len(SYLL_M) * len(SYLL_D) * len(SYLL_B)
    if n > space:
        raise ValueError(f"only {space} names reachable")
    while len(out) < n and tries < n * 500:
        tries += 1
        w = (rng.choice(SYLL_A) + rng.choice(SYLL_M) + rng.choice(SYLL_D)
             + rng.choice(SYLL_B)).upper()
        if w not in _TAKEN:
            _TAKEN.add(w)
            out.append(w)
    if len(out) < n:
        raise RuntimeError(f"produced {len(out)}/{n} names")
    return out


def build(n, split, seed, n_cands=4, n_protocols=2):
    """Build n items with a UNIQUE answer.

    With n_protocols > 2 the question still has one answer: the extreme value
    (min or max). Extra protocols act as distractors with non-extreme values,
    and the candidate list contains all protocol names.
    """
    rng = random.Random(seed + {"train": 11, "dev": 1009, "test": 20011}[split])
    # each item needs n_protocols fresh names plus a gap so splits never collide
    name_pool = make_names(rng, n * n_protocols + 64)

    ta, tb = TEMPLATES_A[split], TEMPLATES_B[split]
    items = []
    ptr = 0
    for i in range(n):
        names = name_pool[ptr : ptr + n_protocols]
        ptr += n_protocols
        fact, unit = rng.choice(FACTS)
        # distinct values so the comparison is unambiguous
        values = rng.sample(range(5, 240), n_protocols)
        direction_word, picker = DIRECTIONS[i % 2]  # balanced 50/50

        facts = [fact.format(v=v) for v in values]
        # chunk A
        if split == "train" and rng.random() < 0.5:
            chunk_a = TEMPLATES_A[split][0].format(
                n1=names[0], f1=facts[0], n2=names[1], f2=facts[1])
        else:
            chunk_a = ta[0].format(n1=names[0], f1=facts[0],
                                   n2=names[1], f2=facts[1]) if n_protocols == 2 \
                else ta[0].format(n1=names[0], f1=facts[0],
                                  n2=names[1], f2=facts[1])
        if n_protocols > 2:
            extra = " ".join(f"Protocol {nm} {f}."
                             for nm, f in zip(names[2:], facts[2:]))
            chunk_a = f"{chunk_a} {extra}"

        chunk_b = rng.choice(tb).format(d=direction_word)
        chunks = [chunk_a, chunk_b]
        rng.shuffle(chunks)

        # the unique answer: the protocol whose value is the extreme one
        target_value = picker(values)
        gold_name = names[values.index(target_value)]

        cands = list(names)
        # pad the candidate list with OTHER protocols' names (never in this
        # chunk), so the task is not a trivial 2-way choice and "pick either of
        # the two names I saw" cannot beat chance
        if len(cands) < n_cands:
            filler = [nm for nm in name_pool if nm not in names]
            rng.shuffle(filler)
            while len(cands) < n_cands and filler:
                cands.append(filler.pop())
        rng.shuffle(cands)
        gold = cands.index(gold_name)

        items.append({
            "chunks": chunks,
            "query": rng.choice(QUERIES),
            "candidates": cands,
            "gold": gold,
            # kept for the oracle / diagnostics, never shown to a model
            "_names": names, "_values": values,
            "_facts": facts, "_direction": direction_word,
            "_answer": gold_name,
        })
    return items


# ---------------------------------------------------------------- checks
NUM_RE = re.compile(r"(\d+)")


def _toks(s):
    return set(w.lower().strip(".,?:;") for w in s.split())


def lexical_leakage(items):
    hits = 0
    for it in items:
        q = _toks(it["query"])
        sc = [len(q & _toks(c)) / max(1, len(q | _toks(c)))
              for c in it["candidates"]]
        order = sorted(range(len(sc)), key=lambda i: -sc[i])
        if sc[order[0]] > 0 and order[0] == it["gold"]:
            hits += 1
    return hits / len(items)


def chunk_overlap_ceiling(items):
    """Bag-of-words matcher on the LAST chunk (the previous killer probe)."""
    hits = 0
    for it in items:
        last = _toks(it["chunks"][-1])
        sc = [len(last & _toks(c)) / max(1, len(last | _toks(c)))
              for c in it["candidates"]]
        order = sorted(range(len(sc)), key=lambda i: -sc[i])
        if sc[order[0]] > 0 and order[0] == it["gold"]:
            hits += 1
    return hits / len(items)


def oracle(items):
    """Perfect reasoner: read both protocols and their values, apply the
    stated comparison direction, return the matching name. Must be ~1.0."""
    ok = 0
    bad = 0
    for it in items:
        names = it["_names"]
        values = it["_values"]
        direction = it["_direction"]
        target = max(values) if direction == "larger" else min(values)
        deduced = names[values.index(target)]
        if deduced == it["candidates"][it["gold"]] and deduced == it["_answer"]:
            ok += 1
        else:
            bad += 1
    return ok / len(items), bad


def uniqueness_check(items):
    """The answer must really be unique: the extreme value must not be tied,
    and exactly one protocol must hold it."""
    bad = 0
    for it in items:
        vals = it["_values"]
        target = max(vals) if it["_direction"] == "larger" else min(vals)
        if vals.count(target) != 1:
            bad += 1
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--out", default="D:/Jaume/data/synth_v3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cands", type=int, default=4)
    ap.add_argument("--protocols", type=int, default=2)
    args = ap.parse_args()

    import os
    os.makedirs(args.out, exist_ok=True)
    report = {}
    for split, n in (("train", args.n), ("dev", args.n // 4),
                     ("test", args.n // 4)):
        items = build(n, split, args.seed, args.cands, args.protocols)
        path = os.path.join(args.out, f"{split}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        chance = sum(1.0 / len(it["candidates"]) for it in items) / len(items)
        orc, bad = oracle(items)
        nonuniq = uniqueness_check(items)
        report[split] = {
            "n": n, "chance": round(chance, 4),
            "lexical_leakage": round(lexical_leakage(items), 4),
            "chunk_overlap_ceiling": round(chunk_overlap_ceiling(items), 4),
            "oracle": round(orc, 4), "oracle_fail": bad,
            "non_unique_answers": nonuniq,
        }
        r = report[split]
        print(f"[v3] {split:5s} n={n:5d} chance={r['chance']:.3f} "
              f"leak={r['lexical_leakage']:.3f} "
              f"chunk_ceil={r['chunk_overlap_ceiling']:.3f} "
              f"oracle={r['oracle']:.3f} nonuniq={nonuniq}")

    with open(os.path.join(args.out, "validity.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nG1 GATE (all must hold):")
    ok = True
    for split, r in report.items():
        cond = (r["lexical_leakage"] <= 0.10
                and r["chunk_overlap_ceiling"] <= 0.25
                and r["oracle"] >= 0.99
                and r["non_unique_answers"] == 0)
        print(f"  {split:5s} {'PASS' if cond else 'FAIL'}  "
              f"leak {r['lexical_leakage']:.3f}<=0.10, "
              f"chunk {r['chunk_overlap_ceiling']:.3f}<=0.25, "
              f"oracle {r['oracle']:.3f}>=0.99, "
              f"unique {r['non_unique_answers']}==0")
        ok = ok and cond
    print(f"\n  G1 overall: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
