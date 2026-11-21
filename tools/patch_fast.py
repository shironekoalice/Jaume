import io
p = "code/repo/scripts/train_lm_planning.py"
src = io.open(p, encoding="utf-8").read()

old = """def build_trajectory_corpus(n_tasks, n_items, n_slots, n_constraints,
                            budget, seed, greedy_only=True):
    \"\"\"One document per greedy step: prefix + action, target = action text.\"\"\"
    rng = random.Random(seed)
    docs = []
    for t in range(n_tasks):
        a, b, cons, sd = generate_task(n_items, n_slots, n_constraints,
                                       seed=seed * 100003 + t)
        env = PlanningEnv(a, b, cons, sd)"""
new = """def _raw_constraints(rng, n_items, n_slots, n_constraints):
    \"\"\"Generate constraints WITHOUT the expensive feasibility search.

    generate_task() retries up to 200 times and calls env.optimum() (a brute
    force over m^n assignments) on every attempt, which is needed when the
    task is going to be SCORED against an exact optimum.  Language-model
    training does not need that: the corpus is the greedy trajectory, and the
    greedy heuristic produces a legal trajectory for any constraint set.  So
    the feasibility search is skipped here -- 40000 tasks took 20+ minutes
    with it and well under a minute without.
    \"\"\"
    cons = []
    for _ in range(n_constraints):
        kind = rng.choice(["before", "not_same", "same", "fixed"])
        x, y = rng.sample(range(n_items), 2) if n_items >= 2 else (0, 0)
        if kind == "fixed":
            cons.append(("fixed", x, rng.randrange(n_slots)))
        else:
            cons.append((kind, x, y))
    return cons


def build_trajectory_corpus(n_tasks, n_items, n_slots, n_constraints,
                            budget, seed, greedy_only=True):
    \"\"\"One document per greedy step: prefix + action, target = action text.\"\"\"
    rng = random.Random(seed)
    docs = []
    for t in range(n_tasks):
        sd = seed * 100003 + t
        cons = _raw_constraints(random.Random(sd), n_items, n_slots,
                                n_constraints)
        a, b = n_items, n_slots
        env = PlanningEnv(a, b, cons, sd)"""
assert old in src
src = src.replace(old, new)
io.open(p, "w", encoding="utf-8", newline="\n").write(src)
print("patched: skip feasibility search for LM corpus")
