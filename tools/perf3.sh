#!/bin/bash
SNAP=/home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master
cd "$SNAP/code/repo"
nohup python -u scripts/train_lm_planning.py \
  --config configs/jaume-350m-loop.json \
  --init-from /mnt/workspace/output/350m-causal-mlm/final \
  --output /tmp/_perf --n-tasks 2000 --num-loops 4 \
  --epochs 1 --batch-size 8 --grad-accum 8 --max-length 192 \
  --lr 3e-4 --warmup-steps 20 --val-docs 128 \
  --eval-every 30 --log-every 10 --device npu \
  > /tmp/perf.log 2>&1 &
echo "perf PID=$!"
sleep 120
echo "=== log ==="
tail -8 /tmp/perf.log
echo "=== AICore ==="
for i in 1 2 3 4 5; do npu-smi info 2>&1 | sed -n '7p' | awk -F'|' "{print \$4}"; sleep 4; done