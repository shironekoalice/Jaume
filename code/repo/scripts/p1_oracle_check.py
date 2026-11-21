"""P1 oracle check: is the answer actually deducible from the two chunks?

v2 — the task now carries the chain  memo id -> entity -> colour:

    chunk A: "<ENTITY> uses the <colour> key. Filed under memo <id>."
    chunk B: "The item under review is the asset in memo <id>."
    query  : "Which colour key should be used?"

A perfect reasoner must (1) read the memo id from chunk B, (2) find that memo id
in chunk A, (3) read the entity and its colour, (4) confirm the gold candidate.
If this fails the generator is broken and no model can learn the task.
"""
import os as _os
PROJECT_ROOT = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import argparse
import json
import re

COLOURS = {"red", "blue", "green", "amber", "violet", "teal",
           "crimson", "indigo", "ochre", "azure", "scarlet", "olive"}
MEMO_RE = re.compile(r"memo (\d+)", re.I)
ENT_RE = re.compile(r"\b([A-Z]{4,})\b")


def colour_in(text):
    low = text.lower()
    for c in COLOURS:
        if re.search(rf"\b{c}\b", low):
            return c
    return None


def parse_memo_pairs(chunk):
    """Return {memo_id: colour}.

    Chunk A is a run of "<entity> ... <colour>. Filed under memo <id>." units.
    Splitting on sentence boundaries separates the colour sentence from its
    memo sentence, so the two must be re-paired: a memo id applies to the
    nearest preceding sentence that contains a colour.
    """
    table = {}
    pending = None
    for sent in re.split(r"(?<=\.)\s+", chunk):
        c = colour_in(sent)
        if c:
            pending = c
        m = MEMO_RE.search(sent)
        if m and pending is not None:
            table[m.group(1)] = pending
            pending = None
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=fr"{PROJECT_ROOT}\data\synth_crossblock\test.jsonl")
    args = ap.parse_args()

    items = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]

    ok = no_memo = no_colour = mismatch = 0
    for it in items:
        # chunk B is the one naming a memo but carrying no colour table
        tables = [parse_memo_pairs(c) for c in it["chunks"]]
        b_idx = 0 if len(tables[0]) <= len(tables[1]) else 1
        a_idx = 1 - b_idx
        m_b = MEMO_RE.search(it["chunks"][b_idx])
        if not m_b:
            no_memo += 1
            continue
        memo = m_b.group(1)
        table = tables[a_idx]
        if memo not in table:
            no_colour += 1
            continue
        deduced = table[memo]
        gold = it["candidates"][it["gold"]].lower()
        if deduced == gold:
            ok += 1
        else:
            mismatch += 1
            if mismatch <= 5:
                print(f"  MISMATCH memo={memo} deduced={deduced} gold={gold}")

    n = len(items)
    print(f"items             : {n}")
    print(f"deducible (oracle): {ok}  ({ok/n:.3f})")
    print(f"no memo in B      : {no_memo}")
    print(f"memo not in A     : {no_colour}")
    print(f"answer mismatches : {mismatch}  ({mismatch/n:.3f})")
    print()
    if ok / n > 0.99:
        print("PASS: answer fully deducible via memo -> entity -> colour.")
    else:
        print("FAIL: generator does not guarantee a deducible answer.")


if __name__ == "__main__":
    main()
