import sys
sys.path.insert(0, r"code\repo\scripts")
from carry_env import add_with_carry, build_docs
print(add_with_carry("1234", "9877"))
for p, a, n in build_docs(5, 8, 1):
    print(repr(p), "->", repr(a), "carries", n)
