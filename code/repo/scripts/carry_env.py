"""Multi-digit addition: the canonical SERIAL dependency task.

Why this task and not the previous ones
---------------------------------------
Every task we tried so far was solvable in one attention pass:

  * chain lookup        -> answer token already in the input;
  * parity / mod-count  -> gradient-invisible, never learned at any depth;
  * constraint path     -> attention computes the closure in one pass;
  * colour counting     -> parallel aggregation.

Column addition is different.  The carry into column i depends on the carry
out of column i-1.  That is a genuine left-to-right (or right-to-left) chain
of state, and the answer digits do not appear anywhere in the operand text.

It is also the task the loop literature uses to claim depth gains, so it is
the fair test for the depth-saturation criterion: if 8L already matches 16L
here, the criterion is not an artefact of our synthetic tasks.
"""

import random


def add_with_carry(a, b):
    """Column addition, returning (digits, carries) for the worked form."""
    n = max(len(a), len(b))
    a = a.zfill(n)
    b = b.zfill(n)
    out = []
    carry = 0
    carries = []
    for i in range(n - 1, -1, -1):
        s = int(a[i]) + int(b[i]) + carry
        out.append(str(s % 10))
        carry = s // 10
        carries.append(carry)
    if carry:
        out.append(str(carry))
    return "".join(reversed(out)), carries


def make_doc(n_digits, rng):
    a = "".join(str(rng.randrange(10)) for _ in range(n_digits)).lstrip("0") or "0"
    b = "".join(str(rng.randrange(10)) for _ in range(n_digits)).lstrip("0") or "0"
    ans, carries = add_with_carry(a, b)
    # number of carry events >=1 is what makes the task serial; skip the
    # (common, trivial) all-no-carry case so every example exercises it
    if sum(carries) == 0:
        return None
    prefix = f"{a}+{b}="
    return " " + prefix, " " + ans, len(carries)


def build_docs(n_tasks, n_digits, seed):
    docs = []
    for t in range(n_tasks * 3):
        if len(docs) >= n_tasks:
            break
        rng = random.Random(seed * 1000003 + t)
        d = make_doc(n_digits, rng)
        if d is None:
            continue
        docs.append(d)
    return docs
