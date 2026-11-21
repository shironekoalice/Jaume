#!/bin/bash
# Stage 1 -> 2 -> 3 full pipeline with metrics logging, unattended.
set -e
cd "$(dirname "$0")/.."
DATA=/root/autodl-tmp/data
OUT=output/pipeline

echo "=== [1/5] Stage 1: MLM pretraining ==="
python -u scripts/train_mlm.py \
    --config configs/jaume-350m.json \
    --data $DATA/mlm_corpus.jsonl \
    --output $OUT/350m-mlm \
    --max-samples 300000 --epochs 1 \
    --batch-size 64 --grad-ckpt --lr 1e-4 --warmup-steps 400 \
    --max-length 128 --save-every 1000

echo "=== [2/5] Stage 2: contrastive from MLM init (all entailment pairs) ==="
python -u scripts/train_embedding.py \
    --config configs/jaume-350m.json \
    --train-data $DATA/all_nli.jsonl \
    --output $OUT/350m-stage2 \
    --init-from $OUT/350m-mlm/final \
    --epochs 1 --batch-size 256 --gc-chunk 64 --grad-ckpt \
    --lr 1e-4 --warmup-steps 30 --max-length 128 --use-matryoshka \
    --num-workers 4 --log-every 20

echo "=== [2b] eval stage2 ==="
python -u scripts/evaluate.py --model $OUT/350m-stage2/final --data $DATA/stsbenchmark_sts.jsonl | tee $OUT/eval-stage2.txt
python -u scripts/evaluate.py --model $OUT/350m-stage2/final --data $DATA/sts22-crosslingual-sts.jsonl >> $OUT/eval-stage2.txt

echo "=== [3/5] Stage 3a: mine hard negatives with new encoder ==="
python -u scripts/mine_hard_negatives.py \
    --model $OUT/350m-stage2/final \
    --pairs $DATA/all_nli.jsonl \
    --output $OUT/nli-hardnegatives.jsonl \
    --num-negatives 1 --max-length 128

echo "=== [4/5] Stage 3b: retrain with hard negatives ==="
python -u scripts/train_embedding.py \
    --config configs/jaume-350m.json \
    --train-data $OUT/nli-hardnegatives.jsonl \
    --output $OUT/350m-stage3 \
    --init-from $OUT/350m-stage2/final \
    --epochs 1 --batch-size 256 --gc-chunk 64 --grad-ckpt \
    --lr 3e-5 --warmup-steps 30 --max-length 128 --use-matryoshka \
    --num-workers 4 --log-every 20

echo "=== [5/5] eval stage3 ==="
python -u scripts/evaluate.py --model $OUT/350m-stage3/final --data $DATA/stsbenchmark_sts.jsonl | tee $OUT/eval-stage3.txt
python -u scripts/evaluate.py --model $OUT/350m-stage3/final --data $DATA/stsbenchmark_sts.jsonl --dimensions 128 >> $OUT/eval-stage3.txt
python -u scripts/evaluate.py --model $OUT/350m-stage3/final --data $DATA/sts22-crosslingual-sts.jsonl >> $OUT/eval-stage3.txt

echo "=== pipeline done ==="
