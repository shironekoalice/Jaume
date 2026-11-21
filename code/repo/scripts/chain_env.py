"""A task whose difficulty is controlled by CHAIN LENGTH, to test the
depth-saturation criterion directly.

The criterion from docs/loop-spec.md 8.10: a loop can only help when the task
is depth-limited, i.e. when 16L clearly beats 8L.  Every previous task
saturated at 8 layers, so every loop experiment was doomed.

This environment builds a pure chain-composition task: each hop is one
lookup and the answer requires composing `hops` of them:

    rules: a0>a1 a1>a2 a2>a3 ...   query: a0 answer: a3

With hops=3 a shallow model can do it; with hops=12 it cannot, because the
composition depth exceeds what a fixed number of layers can execute.  That
gives a knob to sweep against depth.
"""

import random

SYMBOLS = [f"n{i}" for i in range(40)]


def make_chain(n_nodes, hops, symbols, seed):
    rng = random.Random(seed)
    pool = list(symbols)
    rng.shuffle(pool)
    if hops + 1 > len(pool):
        return None
    chain = pool[:hops + 1]
    rules = [(chain[i], chain[i + 1]) for i in range(hops)]
    extra = []
    for _ in range(max(2, hops // 2)):
        a, b = rng.sample(pool, 2)
        if (a, b) not in rules and a != chain[0]:
            extra.append((a, b))
    rng.shuffle(extra)
    return {"chain": chain, "rules": rules, "extra": extra,
            "query": chain[0], "answer": chain[-1], "hops": hops}


def render(p):
    all_rules = p["rules"] + p["extra"]
    random.Random(hash(tuple(p["chain"])) & 0xffff).shuffle(all_rules)
    rules_txt = " ".join(f"{a}>{b}" for a, b in all_rules)
    return {
        "prefix": f"rules: {rules_txt} query: {p['query']} answer:",
        "answer": f" {p['answer']}",
        "hops": p["hops"],
    }


def build_docs(n_tasks, hops, seed):
    docs = []
    for t in range(n_tasks):
        p = make_chain(40, hops, SYMBOLS, seed * 1000003 + t)
        if p is None:
            continue
        docs.append(render(p))
    return docs
