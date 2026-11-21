"""Verify the register task is not solved by a shortcut."""
import sys, random
sys.path.insert(0, "code/repo/scripts")
from register_env import make_program, render, MOD

# 1) is the answer distribution uniform? (if it collapses to one value, the
#    0.0000 loss is suspicious)
from collections import Counter
c = Counter()
for t in range(20000):
    p = make_program(16, 1000003 * 5 + t)
    c[p["answer"]] += 1
print("answer distribution over 20k programs:", dict(sorted(c.items())))
print("entropy-ish: max_count/min_count =", max(c.values()), "/", min(c.values()))

# 2) is the answer determined by ONLY the last op? (a shortcut)
last_op_only = Counter()
for t in range(20000):
    p = make_program(16, 1000003 * 5 + t)
    k, v = p["ops"][-1]
    last_op_only[(p["start"], k, v, p["answer"])] += 1
uniq = len({(s, k, v, a) for (s, k, v, a), _ in last_op_only.items()})
print("distinct (start,last_op,answer) tuples:", uniq)

# 3) baseline: predicting the most common answer
best = c.most_common(1)[0]
print("majority-answer baseline accuracy:", best[1] / 20000, "answer=", best[0])

# 4) does the model actually need the ops? check answer spread per start value
per_start = {}
for t in range(5000):
    p = make_program(16, 777 * t + 11)
    per_start.setdefault(p["start"], Counter())[p["answer"]] += 1
print("answers per start value (first 3 starts):")
for s in sorted(per_start)[:3]:
    print("  start", s, "->", dict(per_start[s]))
