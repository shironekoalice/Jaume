"""A task attention cannot shortcut: iterated state updates applied IN ORDER.

Chain lookup (a0>a1 a1>a2 ... query a0) came out at loss 0.0000 for both 8
and 16 layers, even at 12 hops -- attention can follow the whole chain in one
layer, so chain length is not a depth knob.

This environment uses a small register machine instead:

    ops: A5 M3 A2 S1 ...   start 7 => ?

A=add, M=mul, S=subtract, all mod 13.  The answer is the result of applying
the ops left to right.  A single attention pass can SEE all ops but cannot
compose them arithmetically in order, so n_ops is a genuine depth knob.
"""

import random

MOD = 13


def make_program(n_ops, seed):
    rng = random.Random(seed)
    ops = []
    for _ in range(n_ops):
        k = rng.choice(["A", "M", "S"])
        v = rng.randint(1, 5) if k != "M" else rng.randint(2, 3)
        ops.append((k, v))
    start = rng.randint(0, MOD - 1)
    x = start
    for k, v in ops:
        if k == "A":
            x = (x + v) % MOD
        elif k == "M":
            x = (x * v) % MOD
        else:
            x = (x - v) % MOD
    return {"ops": ops, "start": start, "answer": x, "n_ops": n_ops}


def render(p):
    ops_txt = " ".join(f"{k}{v}" for k, v in p["ops"])
    return {
        "prefix": f"ops: {ops_txt} start {p['start']} =>",
        "answer": f" {p['answer']}",
        "n_ops": p["n_ops"],
    }


def build_docs(n_tasks, n_ops, seed):
    out = []
    for t in range(n_tasks):
        out.append(render(make_program(n_ops, seed * 1000003 + t)))
    return out
