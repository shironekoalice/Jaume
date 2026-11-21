#!/bin/bash
SNAP=/home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master
cd "$SNAP/code/repo"
echo "=== smoke: stream, 64 items, tiny run ==="
timeout 600 python -u scripts/train_stream.py \
  --config configs/jaume-350m-loop.json \
  --init-from /mnt/workspace/output/350m-causal-mlm/final \
  --output /tmp/_smoke_stream --n-tasks 400 --n-items 64 --depth 8 \
  --epochs 1 --batch-size 8 --grad-accum 1 --max-length 384 \
  --lr 3e-4 --warmup-steps 10 --val-docs 32 \
  --eval-every 10 --log-every 10 --device npu 2>&1 | tail -10