import io
p = r"code\repo\scripts\derive_env.py"
s = io.open(p, encoding="utf-8").read()
old = """    cons = [("fixed", 0, pivot)]
    for i in range(chain_len):
        cons.append(("same", i, i + 1))"""
new = """    cons = [("fixed", 0, pivot)]
    # REVERSE order on purpose: derive() walks the list once per round, so a
    # forward-ordered path collapses in round 1.  Reversed, round t only ever
    # extends the frontier by one node -> propagation depth is exactly K.
    for i in range(chain_len - 1, -1, -1):
        cons.append(("same", i, i + 1))"""
assert old in s
s = s.replace(old, new, 1)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
