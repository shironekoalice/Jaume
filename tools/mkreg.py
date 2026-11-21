import io
src = io.open("code/repo/scripts/train_chain.py", encoding="utf-8").read()
src = src.replace("from chain_env import build_docs",
                  "from register_env import build_docs")
src = src.replace('ap.add_argument("--hops", type=int, default=12)',
                  'ap.add_argument("--n-ops", type=int, default=16)')
src = src.replace("args.hops", "args.n_ops")
src = src.replace("hops={args.n_ops}", "n_ops={args.n_ops}")
io.open("code/repo/scripts/train_register.py", "w",
        encoding="utf-8", newline="\n").write(src)
print("written")
