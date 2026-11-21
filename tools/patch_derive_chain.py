import io
p = r"code\repo\scripts\train_lm_derive.py"
s = io.open(p, encoding="utf-8").read()

s = s.replace("from derive_env import make_puzzle, render",
              "from derive_env import make_puzzle, make_chain_puzzle, render", 1)

old = """def build_docs(n_tasks, n_items, n_slots, n_fixed, n_same, n_notsame, seed):
    \"\"\"One document per puzzle. target mask covers BOTH target spans.\"\"\"
    docs = []
    for t in range(n_tasks):
        cons, env = make_puzzle(n_items, n_slots, n_fixed, n_same, n_notsame,
                                seed * 1000003 + t)
        if env is None:
            continue"""
new = """def build_docs(n_tasks, n_items, n_slots, n_fixed, n_same, n_notsame, seed,
               chain_len=0):
    \"\"\"One document per puzzle. target mask covers BOTH target spans.\"\"\"
    docs = []
    for t in range(n_tasks):
        if chain_len > 0:
            cons, env = make_chain_puzzle(n_items, n_slots, chain_len,
                                          seed * 1000003 + t)
        else:
            cons, env = make_puzzle(n_items, n_slots, n_fixed, n_same,
                                    n_notsame, seed * 1000003 + t)
        if env is None:
            continue"""
assert old in s
s = s.replace(old, new, 1)

old = """    ap.add_argument("--n-notsame", type=int, default=3)"""
new = """    ap.add_argument("--n-notsame", type=int, default=3)
    ap.add_argument("--chain-len", type=int, default=0,
                    help="if >0, path of same-constraints with exact depth")"""
assert old in s
s = s.replace(old, new, 1)

old = """    docs = build_docs(args.n_tasks, args.n_items, args.n_slots,
                      args.n_fixed, args.n_same, args.n_notsame, args.seed)"""
new = """    docs = build_docs(args.n_tasks, args.n_items, args.n_slots,
                      args.n_fixed, args.n_same, args.n_notsame, args.seed,
                      args.chain_len)"""
assert old in s
s = s.replace(old, new, 1)

old = """    ap.add_argument("--device", default=None)"""
new = """    ap.add_argument("--depth", type=int, default=0,
                    help="override config num_layers (0 = keep config)")
    ap.add_argument("--device", default=None)"""
assert old in s
s = s.replace(old, new, 1)

old = """    cfg = JaumeConfig.load(args.config)"""
new = """    cfg = JaumeConfig.load(args.config)
    if args.depth > 0:
        cfg.num_layers = args.depth"""
assert old in s
s = s.replace(old, new, 1)

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("patched train_lm_derive")
