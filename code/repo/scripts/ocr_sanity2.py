"""Is the text_locate task information-theoretically solvable at all?

Render the SAME word in each of the 4 boxes and check pixel identity; then
check whether different words produce different pixels. If different words
produce near-identical bitmaps, the task is broken.
"""
import sys
sys.path.insert(0, "/home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master/code/repo/scripts")
from ocr_suite import build, WORDS
import numpy as np

items = [b for b in build(60, seed=31) if b[1] == "text_locate"]
print("n =", len(items))

# extract the word strip of the ANSWER box and of a non-answer box,
# then compare across samples with the same word.
from collections import defaultdict
by_word = defaultdict(list)
for img, fam, which, ans, _, q in items:
    word = q.split()[-1].rstrip("?")
    a = np.asarray(img.convert("L"))
    idx = int(ans.split()[-1]) - 1
    bx = [20, 170, 20, 170][idx]; by = [40, 40, 230, 230][idx]
    strip = a[by + 55:by + 85, bx + 25:bx + 105]
    by_word[word].append(strip)

print("\nwords present:", list(by_word.keys()))
for w, strips in by_word.items():
    arr = np.stack(strips).astype(np.float32)
    # variance across samples that all show the same word:
    # should be ~0 if rendering is deterministic per word
    print("  %-6s n=%d  mean=%.1f  across-sample std=%.2f  shape=%s"
          % (w, len(strips), arr.mean(), arr.std(), arr.shape[1:]))

# cross-word difference: are different words visually distinct?
words = list(by_word.keys())
if len(words) >= 2:
    m0 = np.stack(by_word[words[0]]).mean(0)
    m1 = np.stack(by_word[words[1]]).mean(0)
    diff = np.abs(m0 - m1).mean()
    print("\n  mean |A-B| between '%s' and '%s' = %.2f (0 = identical)"
          % (words[0], words[1], diff))
    # normalise by within-word variance
    within = np.stack(by_word[words[0]]).std()
    print("  within-word std = %.2f  -> signal/noise = %.2f"
          % (within, diff / max(within, 1e-6)))
