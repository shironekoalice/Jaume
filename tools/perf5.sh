#!/bin/bash
# Memory-safe but NPU-saturating: batch 24 x accum 2 (48 effective) instead of
# 8 x 8.  Bigger per-call batch keeps the accelerator busy; the loop's 4x
# activation cost is absorbed by a smaller effective batch, not by starving
# each call.
SNAP=/home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master
cd "$SNAP/code/repo"
nohup python -u scripts/train_lm_planning.py \
  --config configs/jaume-350m-loop.json \
  --init-from /mnt/workspace/output/350m-causal-mlm/final \
  --output /tmp/_perf2 --n-tasks 2000 --num-loops 4 \
  --epochs 1 --batch-size 24 --grad-accum 2 --max-length 192 \
  --lr 3e-4 --warmup-steps 20 --val-docs 128 \
  --eval-every 30 --log-every 10 --device npu \
  > /tmp/perf2.log 2>&1 &
echo "PID=$!"
sleep 100
echo "=== log ==="
tail -5 /tmp/perf2.log
echo "=== AICore ==="
for i in 1 2 3 4; do npu-smi info 2>&1 | sed -n '8p'; sleep 3; done