"""A task that CANNOT be solved in one round: two-stage constraint propagation.

Why the previous tasks failed
-----------------------------
The planning task looked multi-step but was single-step solvable: greedy looks
only at the current assignment and constraints, so round 2 has nothing to
consume from round 1.  The probe confirmed it -- the loop converged FASTER on
that task (first-round update 0.45 vs 0.66 on text) and the gate trained to
0.96, i.e. the optimiser suppressed the later rounds.

The structural property a loop needs
------------------------------------
Round t's input must DEPEND on round t-1's conclusion.  This environment
builds it explicitly -- a two-stage puzzle where:

  stage 1  DERIVE.  Some items are pinned to a slot by `fixed`.  Those pins,
           plus the `same` constraints, force further items into known slots.
           The solver runs a propagation fixpoint and records which items
           became determined in EACH round.

  stage 2  APPLY.  The remaining undetermined items must be placed using the
           facts that stage 1 produced.  This stage literally cannot start
           until stage 1 has run, because the set of remaining items is
           defined by stage 1's result.

So a single forward pass that never revisits its own intermediate conclusion
has strictly less information than one that does.  That is the property the
previous task lacked.

Observable structure for the probe
----------------------------------
`derive_depths(env)` returns, for each item, the propagation round in which
it became determined (0 = given, 1 = implied by givens, 2 = implied by round
1, ...).  A loop that is genuinely doing staged work should show hidden-state
changes that align with these rounds; a loop that is just re-applying one map
will not.  This gives the probe a ground truth to correlate against -- which
generic text can never provide.

Task format
-----------
    derive: item3 is at slot 2 (given). item5 is at slot 2 (implied).
    place: item0 at slot 0, item1 at slot 1

Kept deliberately small so the corpus is cheap and the loss stays comparable
to the previous planning runs.
"""

import itertools
import random


def _sat(assign, cons):
    for c in cons:
        k = c[0]
        if k == "fixed":
            if assign[c[1]] != c[2]:
                return False
        elif k == "same":
            if assign[c[1]] != assign[c[2]]:
                return False
        elif k == "not_same":
            if assign[c[1]] == assign[c[2]]:
                return False
        elif k == "before":
            if not (assign[c[1]] < assign[c[2]]):
                return False
    return True


class DeriveEnv:
    """Two-stage puzzle: derive forced placements, then place the rest."""

    def __init__(self, n_items, n_slots, constraints, seed=0):
        self.n_items = n_items
        self.n_slots = n_slots
        self.constraints = list(constraints)
        self.seed = seed
        self.assignment = [-1] * n_items
        self.determined_at = [-1] * n_items
        self.round = 0

    # ---- stage 1: propagation --------------------------------------
    def derive(self):
        """Run the fixpoint. Returns the list of (round, item, slot) facts."""
        a = [-1] * self.n_items
        depth = [None] * self.n_items
        rnd = 0
        for c in self.constraints:
            if c[0] == "fixed":
                item, slot = c[1], c[2]
                if a[item] == -1:
                    a[item] = slot
                    depth[item] = 0
                elif a[item] != slot:
                    return None, None    # contradictory
        changed = True
        while changed:
            changed = False
            rnd += 1
            if rnd > self.n_items + 1:
                break
            # `same` propagates a known slot to an unknown partner
            for c in self.constraints:
                if c[0] != "same":
                    continue
                x, y = c[1], c[2]
                if a[x] >= 0 and a[y] < 0:
                    a[y] = a[x]
                    depth[y] = rnd
                    changed = True
                elif a[y] >= 0 and a[x] < 0:
                    a[x] = a[y]
                    depth[x] = rnd
                    changed = True
                elif a[x] >= 0 and a[y] >= 0 and a[x] != a[y]:
                    return None, None
            # `not_same` + "all other slots taken" forces the last slot
            if self.n_slots <= 6:
                for item in range(self.n_items):
                    if a[item] >= 0:
                        continue
                    taken = {a[j] for j in range(self.n_items) if a[j] >= 0}
                    forbidden = set(taken)
                    for c in self.constraints:
                        if c[0] == "not_same":
                            if c[1] == item and a[c[2]] >= 0:
                                forbidden.add(a[c[2]])
                            if c[2] == item and a[c[1]] >= 0:
                                forbidden.add(a[c[1]])
                    allowed = [s for s in range(self.n_slots)
                               if s not in forbidden]
                    if len(allowed) == 1:
                        a[item] = allowed[0]
                        depth[item] = rnd
                        changed = True
        self.assignment = a
        self.determined_at = depth
        self.round = rnd
        return [(depth[i], i, a[i]) for i in range(self.n_items)
                if a[i] >= 0], depth

    def remaining(self):
        return [i for i in range(self.n_items) if self.assignment[i] < 0]

    def max_depth(self):
        ds = [d for d in self.determined_at if d is not None]
        return max(ds) if ds else 0


def make_chain_puzzle(n_items, n_slots, chain_len, seed):
    """A puzzle whose constraint propagation is EXACTLY chain_len rounds deep.

    Every earlier task was solved by one attention pass:
      * chain lookup  -> the answer token was already in the input;
      * parity/count  -> gradient-invisible, never learned at any depth;
      * derive (random) -> propagation depth was uncontrolled and shallow.

    Here the same-constraint graph is a PATH: item0 == item1 == ... == itemK,
    and only item0 is fixed.  Propagating the fixed slot to itemK takes
    exactly K rounds -- each round extends the known set by one node.  That is
    a genuine serial dependency: round t+1 needs round t's conclusion.
    Crucially it stays learnable, because each step is a trivial copy rule.
    """
    rng = random.Random(seed)
    n_items = max(n_items, chain_len + 1)
    pivot = rng.randrange(n_slots)
    cons = [("fixed", 0, pivot)]
    # REVERSE order on purpose: derive() walks the list once per round, so a
    # forward-ordered path collapses in round 1.  Reversed, round t only ever
    # extends the frontier by one node -> propagation depth is exactly K.
    for i in range(chain_len - 1, -1, -1):
        cons.append(("same", i, i + 1))
    for _ in range(2):
        if n_items > chain_len + 1:
            a = rng.randrange(chain_len + 1, n_items)
            b = rng.randrange(chain_len + 1)
            cons.append(("not_same", a, b))
    env = DeriveEnv(n_items, n_slots, cons, seed)
    facts, depth = env.derive()
    if facts is None or depth is None:
        return None, None
    if max(d for d in depth if d is not None) != chain_len:
        return None, None
    return cons, env


def make_puzzle(n_items, n_slots, n_fixed, n_same, n_notsame, seed):
    """Generate a puzzle whose stage 1 has at least two propagation levels."""
    rng = random.Random(seed)
    for _ in range(200):
        cons = []
        used_slots = {}
        for _ in range(n_fixed):
            it = rng.randrange(n_items)
            sl = rng.randrange(n_slots)
            if it in used_slots:
                continue
            used_slots[it] = sl
            cons.append(("fixed", it, sl))
        for _ in range(n_same):
            x, y = rng.sample(range(n_items), 2)
            cons.append(("same", x, y))
        for _ in range(n_notsame):
            x, y = rng.sample(range(n_items), 2)
            cons.append(("not_same", x, y))
        env = DeriveEnv(n_items, n_slots, cons, seed)
        facts, depth = env.derive()
        if facts is None:
            continue
        # require at least one item determined at round >= 2: that is what
        # makes the task genuinely two-stage
        if depth and max(d for d in depth if d is not None) >= 2:
            return cons, env
    return None, None


def render(env):
    """Linearise the derive+place answer as training text."""
    d = env.determined_at
    derived = [(i, env.assignment[i], d[i]) for i in range(env.n_items)
               if env.assignment[i] >= 0]
    derived.sort(key=lambda t: (t[2], t[0]))
    rem = env.remaining()
    given = [f"item{i}=slot{s}" for i, s, dep in derived if dep == 0]
    implied = [f"item{i}=slot{s}" for i, s, dep in derived if dep and dep > 0]
    return {
        "given": " ".join(given),
        "implied": " ".join(implied),
        "remaining": rem,
        "depth": d,
    }
