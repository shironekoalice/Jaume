"""A task that is depth-limited because the INPUT exceeds one forward pass.

Why the previous tasks all saturated at 8 layers
------------------------------------------------
Chain lookup, two-stage derive, and register programs all had inputs of a few
dozen tokens.  An 8-layer x 512-dim transformer has enormous parallel
expressive power over 40 tokens -- it solved all three at loss 0.0000.  Depth
was never the binding constraint; the task was simply small.

What actually requires depth
----------------------------
A computation whose intermediate state must be carried over MORE TOKENS THAN
THE CONTEXT WINDOW.  This environment streams a long array through a small
recurrent buffer:

    stream: 3 7 1 9 4 ... (N numbers, potentially thousands)
    task:   report (sum mod 97) and (max) after processing all N

The model sees the whole stream at once only if N fits the context; to test
whether depth is the constraint, we make N large enough that the compute must
be sequential -- one pass cannot reduce a 4000-token array without either
deep iteration or a scratchpad.

For the depth sweep this produces the regime we need:
  * small N -- 8 layers suffice (parallel reduction);
  * large N -- 16 layers should beat 8, because the reduction needs more
    sequential stages than 8 layers provide.
"""

import random

MOD = 97


def make_stream(n_items, seed):
    rng = random.Random(seed)
    vals = [rng.randint(0, 96) for _ in range(n_items)]
    total = sum(vals) % MOD
    mx = max(vals)
    return {"vals": vals, "sum": total, "max": mx, "n": n_items}


def render(p):
    body = " ".join(str(v) for v in p["vals"])
    return {
        "prefix": f"stream: {body} reduce:",
        # two numbers, mod-97 sum and the max; both tokenised as one span
        "answer": f" {p['sum']} {p['max']}",
        "n": p["n"],
    }


def build_docs(n_tasks, n_items, seed):
    return [render(make_stream(n_items, seed * 1000003 + t))
            for t in range(n_tasks)]
