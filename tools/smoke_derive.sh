#!/bin/bash
SNAP=/home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master
cd "$SNAP/code/repo"
timeout 600 python -u scripts/train_lm_derive.py \
  --config configs/jaume-350m-loop.json \
  --init-from /mnt/workspace/output/350m-causal-mlm/final \
  --output /tmp/_smoke_derive --n-tasks 300 --num-loops 2 \
  --epochs 1 --batch-size 8 --grad-accum 1 --max-length 160 \
  --lr 1e-5 --warmup-steps 5 --val-docs 32 \
  --eval-every 20 --log-every 10 --device npu 2>&1 | tail -14