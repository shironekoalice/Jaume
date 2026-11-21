import io
p = r"code\repo\scripts\train_vlm_probe.py"
s = io.open(p, encoding="utf-8").read()
old = '''                mod = 2 if args.task == "parity" else 3
                n_on = (i * 7 + (i // 16)) % (GRID * GRID)
                img, lab = make_image_parity(n_on, mod, rng)'''
new = '''                mod = 2 if args.task == "parity" else 3
                # balance labels exactly: class round-robin, then spread n_on
                n_on = (i // mod) % (GRID * GRID)
                n_on = n_on - (n_on % mod) + (i % mod)
                if n_on >= GRID * GRID:
                    n_on -= mod
                img, lab = make_image_parity(n_on, mod, rng)'''
assert old in s
s = s.replace(old, new, 1)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
