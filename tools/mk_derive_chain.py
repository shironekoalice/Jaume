import io
p = r"code\repo\scripts\derive_env.py"
s = io.open(p, encoding="utf-8").read()
marker = "def make_puzzle("
assert marker in s
new_fn = '''def make_chain_puzzle(n_items, n_slots, chain_len, seed):
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
    for i in range(chain_len):
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


'''
s = s.replace(marker, new_fn + marker, 1)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("added make_chain_puzzle")
