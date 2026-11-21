#!/bin/bash
SNAP=/home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master
CKPT=/mnt/workspace/output/350m-causal-mlm/final
cd "$SNAP/code/repo"

echo "=== [1] FINE-TUNE, whole encoder unfrozen (decisive) ==="
python -u scripts/plan_finetune.py --model "$CKPT" \
  --n-constraints 8 --train-tasks 300 --test-tasks 100 \
  --budget 30 --epochs 10 --label imitation --unfreeze -1 \
  --lr-head 1e-3 --lr-backbone 1e-5 --micro-groups 32 --accum 4 \
  --device npu --out /mnt/workspace/output/plan-ft-full.json \
  2>&1 | tee /mnt/workspace/plan-ft-full.log
echo "=== [1] DONE $(date +%H:%M:%S) ==="

echo "=== [2] CONTROL: frozen encoder, joint encoding ==="
python -u scripts/plan_ce_study.py --model "$CKPT" --encoding joint \
  --n-constraints 8 --train-tasks 300 --test-tasks 100 \
  --budget 30 --widths 1 3 --epochs 30 --device npu \
  --out /mnt/workspace/output/plan-ce-joint.json \
  2>&1 | tee /mnt/workspace/plan-ce-joint.log
echo "=== [2] DONE $(date +%H:%M:%S) ==="