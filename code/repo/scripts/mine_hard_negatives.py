"""Stage 3: mine hard negatives with a trained encoder, then write jsonl.

Usage:
    python scripts/mine_hard_negatives.py \
        --model output/350m-nli/final \
        --pairs /root/autodl-tmp/data/all_nli.jsonl \
        --output output/nli-hardnegatives.jsonl \
        --max-samples 40000 --num-negatives 1
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jaume.data.dataset import PairDataset
from jaume.data.negative_mining import HardNegativeMiner
from jaume.inference.encode import JaumeEmbedding


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="trained checkpoint dir")
    p.add_argument("--pairs", required=True, help="jsonl with query/positive rows")
    p.add_argument("--output", required=True)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--num-negatives", type=int, default=1)
    p.add_argument("--top-k", type=int, default=64)
    p.add_argument("--min-score", type=float, default=0.30)
    p.add_argument("--max-score", type=float, default=0.90)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-length", type=int, default=128)
    args = p.parse_args()

    dataset = PairDataset(args.pairs, max_samples=args.max_samples)
    queries = [r["query"] for r in dataset.pairs]
    positives = [r["positive"] for r in dataset.pairs]
    print(f"[mine] {len(queries)} pairs")

    # corpus = unique positives (hard negatives are other pairs' positives)
    corpus = sorted(set(positives))
    print(f"[mine] corpus: {len(corpus)} unique passages")

    wrapper = JaumeEmbedding.from_pretrained(args.model)
    miner = HardNegativeMiner(
        wrapper.model,
        wrapper.tokenizer,
        device=wrapper.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    corpus_texts, corpus_emb = miner.build_index(corpus)
    print(f"[mine] corpus encoded: {tuple(corpus_emb.shape)}")

    rows = miner.mine(
        queries, positives, corpus_texts, corpus_embeddings=corpus_emb,
        num_negatives=args.num_negatives, top_k=args.top_k,
        min_score=args.min_score, max_score=args.max_score,
    )
    miner.save(rows, args.output)
    coverage = len(rows) / max(1, len(queries))
    print(f"[mine] {len(rows)}/{len(queries)} pairs got hard negatives "
          f"({coverage:.0%}) -> {args.output}")
    if rows:
        scores = [s for r in rows for s in r["hard_negative_scores"]]
        print(f"[mine] hard negative score range: {min(scores):.3f}–{max(scores):.3f} "
              f"mean {sum(scores)/len(scores):.3f}")


if __name__ == "__main__":
    main()
