#!/bin/bash
pkill -f plan_finetune 2>/dev/null
pkill -f chain3.sh 2>/dev/null
sleep 3
SNAP=/home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master
CKPT=/mnt/workspace/output/350m-causal-mlm/final
cd "$SNAP/code/repo"
echo "=== fine-tune, dense reporting ==="
python -u scripts/plan_finetune.py --model "$CKPT" \
  --n-constraints 8 --train-tasks 150 --test-tasks 100 \
  --budget 30 --epochs 6 --label imitation --unfreeze -1 \
  --lr-head 1e-3 --lr-backbone 1e-5 --micro-groups 8 --accum 16 \
  --max-length 96 --device npu \
  --out /mnt/workspace/output/plan-ft-full.json \
  2>&1 | tee /mnt/workspace/plan-ft-full.log
echo "=== FT DONE $(date +%H:%M:%S) ==="