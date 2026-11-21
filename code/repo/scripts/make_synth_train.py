"""P2 data prep: turn the synthetic cross-block task into TRAINING pairs that
actually require the memory state.

The naive conversion (`query` = the question, `positive` = chunk A) does NOT
require cross-chunk memory: the question alone is identical across items, so the
contrastive objective can only learn to prefer "some chunk A", not the right
one. The chunk B information (which entity is under review) never enters.

The fix: make chunk B part of the *query side* of the pair, while chunk A stays
on the *document side*. Then the only way to score the pair highly is to read
chunk B, find the target entity, and locate that entity's colour inside chunk A.

Pairing used here (one item -> two training pairs):

    pair 1 (positive):  query = "<chunk B> <question>"     positive = chunk A
    pair 2 (negative):  query = same                       negative = a chunk A'
                        built from a DIFFERENT item (wrong entity table)

In-batch negatives already supply wrong chunk A's, so pair 2 is implicit: with
MNRL, every other item's chunk A is a negative for this query. The requirement
is therefore only that the query encodes WHICH entity is wanted, which the
shared question text alone cannot do.

This is deliberately a *sentinel test of the training objective*: if the model
can learn this, the objective does reward cross-chunk reading. If it cannot,
the objective (or the pooling) is the blocker -- which is exactly the question
P2 must answer.

Usage:
    python scripts/make_synth_train.py \
        --src data/synth_crossblock/train.jsonl \
        --out data/synth_crossblock/train_pairs.jsonl
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--query-mode", default="chunkB_first",
                    choices=["chunkB_first", "question_only"],
                    help="chunkB_first = the cross-chunk condition (default); "
                         "question_only = the memoryless control")
    ap.add_argument("--positive-mode", default="entity_colour",
                    choices=["entity_colour", "answer_only", "chunk_a"],
                    help="entity_colour = '<entity> <colour>' (unique, default); "
                         "answer_only = just the colour (collides across items); "
                         "chunk_a = the whole table (LEAKY)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.src, encoding="utf-8") if l.strip()]
    out = []
    for it in rows:
        chunks = it["chunks"]
        # identify chunk A (the entity->colour table) as the longer chunk
        a_idx = 0 if len(chunks[0]) >= len(chunks[1]) else 1
        b_idx = 1 - a_idx
        chunk_a, chunk_b = chunks[a_idx], chunks[b_idx]
        if args.query_mode == "chunkB_first":
            query = f"{chunk_b} {it['query']}"
        else:
            query = it["query"]
        if args.positive_mode == "answer_only":
            # The colour word alone. Chunk A contains the entity name that also
            # appears in the query, so making the positive the whole table let
            # the model lower the loss by matching that name without ever
            # reading which colour belongs to it (train loss fell to 1.27 while
            # test accuracy stayed at chance). Using only the colour forces the
            # table to be read.
            positive = it["answer"]
        elif args.positive_mode == "entity_colour":
            # "<entity> <colour>": unique per item, so in-batch negatives are
            # genuine (a shared colour word alone made ~500 items per colour
            # act as false negatives for each other), while still requiring the
            # memo -> entity -> colour chain because the query only names a
            # memo id.
            positive = f"{it['target_entity'].lower()} {it['answer']}"
        else:
            positive = chunk_a
        out.append({"query": query, "positive": positive})

    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[synth-train] {args.src} -> {args.out}  {len(out)} pairs "
          f"(query-mode={args.query_mode})")
    print(f"[synth-train] example query   : {out[0]['query'][:110]}")
    print(f"[synth-train] example positive: {out[0]['positive'][:110]}")


if __name__ == "__main__":
    main()
