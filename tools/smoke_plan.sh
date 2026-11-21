#!/bin/bash
SNAP=/home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master
cd "$SNAP/code/repo"
echo "=== smoke: planning-task LM, tiny, on NPU ==="
B=/mnt/workspace/output/350m-causal-mlm/final
timeout 900 python -u scripts/train_lm_planning.py \
  --config configs/jaume-350m-loop.json --init-from $B \
  --output /tmp/_smoke_plan --n-tasks 30 --num-loops 2 \
  --epochs 1 --batch-size 4 --grad-accum 1 --max-length 128 \
  --val-docs 16 --eval-every 5 --log-every 3 --lr 1e-5 --device npu 2>&1 | tail -14