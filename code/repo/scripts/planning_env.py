"""A constraint-based planning environment with an exact optimum.

Why this exists alongside ALFWorld
----------------------------------
ALFWorld is the academic benchmark for text-based embodied planning, but it has
two properties that make it a poor fit for a 350M encoder:

  * the action space is free-form text, so the model must GENERATE commands --
    that is an LLM capability, not a scoring capability;
  * published results need large LLMs plus heavy prompt engineering, so a small
    encoder scores near zero for reasons unrelated to planning.

This environment instead tests the capability the Jaume heads actually have:
scoring candidate actions. The task is a small constraint-satisfaction
planning problem with three properties that make it scientifically useful:

  1. an EXACT optimum is computable by brute force, so we can report the gap
     to optimal rather than just success/failure;
  2. difficulty is a single knob (the number of constraints), so the
     "where does the model start helping" question becomes measurable;
  3. the reward is dense and verifiable: a state either satisfies all
     constraints or it does not.

Task
----
  items    = {0..n-1}
  slots    = {0..m-1}
  state    = assignment[item] -> slot or -1 (unplaced)
  action   = place(item, slot) | remove(item)
  goal     = a set of constraints of the form
               before(a, b)   slot(a) < slot(b)
               not_same(a,b)  slot(a) != slot(b)
               same(a,b)      slot(a) == slot(b)
               fixed(a, s)    slot(a) == s
  success  = every item placed and every constraint satisfied

The optimum is found by enumerating all assignments when m^n is small
(the default grid keeps it under ~10^6).

Usage
-----
    from planning_env import PlanningEnv, generate_task
    env = PlanningEnv(*generate_task(n_items=6, n_slots=4, n_constraints=6, seed=0))
    env.reset()
    env.step(("place", 0, 2))
    print(env.satisfied(), env.goal_reached())
"""

import itertools
import random


class PlanningEnv:
    """A finite constraint-satisfaction planning problem."""

    def __init__(self, n_items, n_slots, constraints, seed=0):
        self.n_items = n_items
        self.n_slots = n_slots
        self.constraints = list(constraints)
        self.seed = seed
        self.reset()

    # ---- state -------------------------------------------------------
    def reset(self):
        self.assignment = [-1] * self.n_items
        self.steps = 0
        return self.state()

    def state(self):
        return {
            "assignment": list(self.assignment),
            "n_items": self.n_items,
            "n_slots": self.n_slots,
            "constraints": self.constraints,
            "steps": self.steps,
        }

    def legal_actions(self):
        """Every (item, slot) the model could place next, plus removals."""
        acts = []
        for it in range(self.n_items):
            for sl in range(self.n_slots):
                if self.assignment[it] != sl:
                    acts.append(("place", it, sl))
        for it in range(self.n_items):
            if self.assignment[it] != -1:
                acts.append(("remove", it, -1))
        return acts

    def step(self, action):
        kind = action[0]
        if kind == "place":
            _, it, sl = action
            if not (0 <= it < self.n_items and 0 <= sl < self.n_slots):
                return self.state(), -1.0, False
            self.assignment[it] = sl
        elif kind == "remove":
            _, it, _ = action
            if not (0 <= it < self.n_items):
                return self.state(), -1.0, False
            self.assignment[it] = -1
        else:
            return self.state(), -1.0, False
        self.steps += 1
        return self.state(), 0.0, self.goal_reached()

    # ---- goal --------------------------------------------------------
    def satisfied(self):
        a = self.assignment
        for c in self.constraints:
            kind = c[0]
            if kind == "before":
                _, x, y = c
                if a[x] == -1 or a[y] == -1 or not (a[x] < a[y]):
                    return False
            elif kind == "not_same":
                _, x, y = c
                if a[x] == -1 or a[y] == -1 or a[x] == a[y]:
                    return False
            elif kind == "same":
                _, x, y = c
                if a[x] == -1 or a[y] == -1 or a[x] != a[y]:
                    return False
            elif kind == "fixed":
                _, x, s = c
                if a[x] != s:
                    return False
        return True

    def goal_reached(self):
        return all(v != -1 for v in self.assignment) and self.satisfied()

    def progress(self):
        """Fraction of constraints currently satisfied (dense reward)."""
        if not self.constraints:
            return 1.0
        ok = 0
        a = self.assignment
        for c in self.constraints:
            kind = c[0]
            if kind == "before":
                _, x, y = c
                ok += int(a[x] != -1 and a[y] != -1 and a[x] < a[y])
            elif kind == "not_same":
                _, x, y = c
                ok += int(a[x] != -1 and a[y] != -1 and a[x] != a[y])
            elif kind == "same":
                _, x, y = c
                ok += int(a[x] != -1 and a[y] != -1 and a[x] == a[y])
            elif kind == "fixed":
                _, x, s = c
                ok += int(a[x] == s)
        return ok / len(self.constraints)

    # ---- exact optimum ----------------------------------------------
    def optimum(self):
        """Brute-force the best achievable number of satisfied constraints.

        Returns (best_satisfied_count, best_assignment) or None when the
        state space is too large to enumerate.
        """
        if self.n_slots ** self.n_items > 2_000_000:
            return None
        best = (-1, None)
        for combo in itertools.product(range(self.n_slots), repeat=self.n_items):
            a = list(combo)
            ok = 0
            for c in self.constraints:
                kind = c[0]
                if kind == "before":
                    _, x, y = c
                    ok += int(a[x] < a[y])
                elif kind == "not_same":
                    _, x, y = c
                    ok += int(a[x] != a[y])
                elif kind == "same":
                    _, x, y = c
                    ok += int(a[x] == a[y])
                elif kind == "fixed":
                    _, x, s = c
                    ok += int(a[x] == s)
            if ok > best[0]:
                best = (ok, a)
        return best


def generate_task(n_items=6, n_slots=4, n_constraints=6, seed=0, ensure_feasible=True):
    """Create a task, optionally retrying until a full solution exists."""
    rng = random.Random(seed)
    for attempt in range(200):
        constraints = []
        for _ in range(n_constraints):
            kind = rng.choice(["before", "not_same", "same", "fixed"])
            x, y = rng.sample(range(n_items), 2) if n_items >= 2 else (0, 0)
            if kind == "fixed":
                constraints.append(("fixed", x, rng.randrange(n_slots)))
            else:
                constraints.append((kind, x, y))
        env = PlanningEnv(n_items, n_slots, constraints, seed=seed)
        opt = env.optimum()
        if opt is None:
            return n_items, n_slots, constraints, seed
        if not ensure_feasible or opt[0] == n_constraints:
            return n_items, n_slots, constraints, seed
    return n_items, n_slots, constraints, seed
