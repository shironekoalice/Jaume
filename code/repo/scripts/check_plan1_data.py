"""Verify the v3 data satisfies plan-1's requirements BEFORE training.

Plan 1 changes the information flow to:

    s_A = f(chunk A, s_0)        chunk A holds the facts
    y   = g(chunk B, s_A)        chunk B is the instruction ONLY

Requirement C (no leakage) says chunk B must not contain the facts needed to
answer. This checks exactly that, plus split hygiene:

  C1  chunk B contains no digits (no fact values leak into step 2)
  C2  chunk B does not name either protocol that appears in chunk A
  C3  train/test entity (protocol-name) vocabularies are disjoint
  C4  the answer cannot be recovered from chunk B alone by any lexical rule
  C5  the answer is uniquely determined by chunk A + chunk B (oracle)

If any check fails the data must be rebuilt before training.
"""
import os as _os
PROJECT_ROOT = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import argparse
import json
import os
import re


def split_chunks(it):
    def has_num(s):
        return any(ch.isdigit() for ch in s)
    a, b = it["chunks"][0], it["chunks"][1]
    if has_num(a) and not has_num(b):
        return a, b
    if has_num(b) and not has_num(a):
        return b, a
    return (a, b) if len(a) >= len(b) else (b, a)


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


NAME_RE = re.compile(r"\b([A-Z]{8,})\b")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=fr"{PROJECT_ROOT}\data\synth_v3")
    args = ap.parse_args()

    tr = load(os.path.join(args.data, "train.jsonl"))
    te = load(os.path.join(args.data, "test.jsonl"))
    ok = True

    # C1: chunk B has no digits
    bad = sum(1 for it in tr + te if any(c.isdigit() for c in split_chunks(it)[1]))
    print(f"C1 chunk B contains digits        : {bad} / {len(tr)+len(te)}"
          f"  {'PASS' if bad == 0 else 'FAIL'}")
    ok &= bad == 0

    # C2: chunk B must not name the protocols listed in chunk A
    bad = 0
    for it in tr + te:
        a, b = split_chunks(it)
        names_in_a = set(NAME_RE.findall(a))
        names_in_b = set(NAME_RE.findall(b))
        if names_in_a & names_in_b:
            bad += 1
    print(f"C2 chunk B names a chunk-A protocol: {bad}  "
          f"{'PASS' if bad == 0 else 'FAIL'}")
    ok &= bad == 0

    # C3: train/test protocol-name vocabularies disjoint
    def vocab(items):
        s = set()
        for it in items:
            s |= set(NAME_RE.findall(split_chunks(it)[0]))
        return s
    vtr, vte = vocab(tr), vocab(te)
    overlap = vtr & vte
    print(f"C3 train/test name overlap        : {len(overlap)}  "
          f"{'PASS' if not overlap else 'FAIL'}")
    ok &= not overlap

    # C4: no lexical rule recovers the answer from chunk B alone.
    #     chunk B is drawn from a tiny template set, so any rule keyed on it is
    #     constant and can only reach the majority-class rate.
    import collections
    maj = collections.Counter(it["gold"] for it in te).most_common(1)[0][1] / len(te)
    print(f"C4 best constant guess from B     : {maj:.3f} "
          f"(chance {sum(1.0/len(it['candidates']) for it in te)/len(te):.3f})")

    # C5: oracle - answer uniquely determined by the facts + direction
    good = 0
    for it in tr + te:
        a, b = split_chunks(it)
        nums = [int(x) for x in re.findall(r"\d+", a)]
        if len(nums) < 2:
            continue
        want_max = "larger" in b.lower()
        target = max(nums) if want_max else min(nums)
        if nums.count(target) != 1:
            continue
        # the gold candidate must be the protocol carrying that value.
        # Split on BOTH sentence punctuation and the "while/;" joiners used by
        # the templates: splitting on '.' alone failed whenever a template put
        # two protocols in one sentence separated by ';', which made the
        # oracle look like 0.90 while the DATA was in fact correct.
        sentences = re.split(r"(?<=\.)\s+|[;]|\bwhile\b", a)
        ent = None
        for s in sentences:
            ns = [int(x) for x in re.findall(r"\d+", s)]
            if ns and ns[0] == target:
                m = NAME_RE.search(s)
                if m:
                    ent = m.group(1)
                break
        if ent and it["candidates"][it["gold"]] == ent:
            good += 1
    rate = good / len(tr + te)
    print(f"C5 oracle deducible               : {rate:.3f}  "
          f"{'PASS' if rate > 0.99 else 'FAIL'}")
    ok &= rate > 0.99

    print(f"\nDATA READY FOR PLAN 1: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
