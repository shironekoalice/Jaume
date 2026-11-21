import sys, random, torch
sys.path.insert(0, "code/repo/scripts")
from train_vlm_probe import make_image, SHAPES

# how ambiguous is the task?  Check: given max_shapes=N, what is the
# distribution of the answer?  If one value dominates, accuracy is capped.
from collections import Counter
for mx in (6, 10):
    c = Counter()
    for i in range(4000):
        k = random.Random(1000 + i).randint(1, mx)
        q = SHAPES[i % 3]
        img, cnt = make_image(k, q, 1000 + i)
        c[cnt] += 1
    tot = sum(c.values())
    best = max(c.values()) / tot
    print("max_shapes=%d  answers=%s  majority=%.3f" % (mx, dict(sorted(c.items())), best))
