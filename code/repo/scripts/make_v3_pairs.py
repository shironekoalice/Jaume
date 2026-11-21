"""Convert v3 cross-block items to (query, positive) training pairs.

Same logic as `make_synth_train.py` but for the v3 schema:

    query    = "<chunk B instruction> <question>"   (never names a protocol)
    positive = "<chunk A facts> <answer name>"

Why the answer name is appended to the positive: the contrastive objective must
be lowered only by pairing the *right* chunk A with the question. If the
positive were the bare answer name, the model could learn a mapping from the
question text alone. Putting chunk A on the positive side and the answer name
with it forces the model to use chunk A.

Training here is a means of fitting the OUTPUT PATH, not the final measurement:
the measurement always re-runs the four conditions (concat/reset/carried/
shuffled) with the same evaluation code.

Usage:
    python scripts/make_v3_pairs.py --data D:/Jaume/data/synth_v3
"""
import os as _os
PROJECT_ROOT = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import argparse
import json
import os


def split_chunks(it):
    def has_num(s):
        return any(ch.isdigit() for ch in s)
    a, b = it["chunks"][0], it["chunks"][1]
    if has_num(a) and not has_num(b):
        return a, b
    if has_num(b) and not has_num(a):
        return b, a
    return (a, b) if len(a) >= len(b) else (b, a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=fr"{PROJECT_ROOT}\data\synth_v3")
    args = ap.parse_args()

    for split in ("train", "dev"):
        src = os.path.join(args.data, f"{split}.jsonl")
        dst = os.path.join(args.data, f"{split}_pairs.jsonl")
        rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
        out = []
        for it in rows:
            a, b = split_chunks(it)
            ans = it["candidates"][it["gold"]]
            out.append({
                # The query carries the instruction AND the fact table, so the
                # passage side can be the bare answer. Training the model to map
                # "instruction + table -> answer name" is exactly the skill the
                # four conditions then test: with the table in memory (carried)
                # versus without it (reset).
                "query": f"{b} {a} {it['query']}",
                "positive": ans,
            })
        with open(dst, "w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[v3-pairs] {split}: {len(out)} pairs -> {dst}")
        if out:
            print(f"    query   : {out[0]['query'][:100]}")
            print(f"    positive: {out[0]['positive'][:100]}")


if __name__ == "__main__":
    main()
