import io
src = io.open("code/repo/scripts/train_chain.py", encoding="utf-8").read()
src = src.replace("from chain_env import build_docs", "from stream_env import build_docs")
src = src.replace('ap.add_argument("--hops", type=int, default=12)',
                  'ap.add_argument("--n-items", type=int, default=512)')
src = src.replace("args.hops", "args.n_items")
src = src.replace("hops={args.n_ops}", "n={args.n_items}")
src = src.replace("n_ops={args.n_ops}", "n={args.n_items}")
# answer is now two tokens; predict the first as before but score full span
src = src.replace("""        ans_all[k] = a[0]""", """        ans_all[k] = a[0]""")
io.open("code/repo/scripts/train_stream.py", "w",
        encoding="utf-8", newline="\n").write(src)
print("written")
