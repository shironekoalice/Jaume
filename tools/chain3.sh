#!/bin/bash
SNAP=/home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master
CKPT=/mnt/workspace/output/350m-causal-mlm/final
cd "$SNAP/code/repo"

echo "=== [1a] FINE-TUNE full encoder, small micro-batch ==="
python -u scripts/plan_finetune.py --model "$CKPT" \
  --n-constraints 8 --train-tasks 300 --test-tasks 100 \
  --budget 30 --epochs 10 --label imitation --unfreeze -1 \
  --lr-head 1e-3 --lr-backbone 1e-5 --micro-groups 8 --accum 16 \
  --max-length 96 --device npu \
  --out /mnt/workspace/output/plan-ft-full.json \
  2>&1 | tee /mnt/workspace/plan-ft-full.log
echo "=== [1a] exit=$? at $(date +%H:%M:%S) ==="

if ! grep -q "train top-1" /mnt/workspace/plan-ft-full.log; then
  echo "=== [1b] retry with top-4 layers only ==="
  python -u scripts/plan_finetune.py --model "$CKPT" \
    --n-constraints 8 --train-tasks 300 --test-tasks 100 \
    --budget 30 --epochs 10 --label imitation --unfreeze 4 \
    --lr-head 1e-3 --lr-backbone 2e-5 --micro-groups 16 --accum 8 \
    --max-length 96 --device npu \
    --out /mnt/workspace/output/plan-ft-top4.json \
    2>&1 | tee /mnt/workspace/plan-ft-top4.log
fi
echo "=== FT DONE $(date +%H:%M:%S) ==="

echo "=== [2] CONTROL: frozen encoder, joint encoding ==="
python -u scripts/plan_ce_study.py --model "$CKPT" --encoding joint \
  --n-constraints 8 --train-tasks 300 --test-tasks 100 \
  --budget 30 --widths 1 3 --epochs 30 --device npu \
  --out /mnt/workspace/output/plan-ce-joint.json \
  2>&1 | tee /mnt/workspace/plan-ce-joint.log
echo "=== ALL DONE $(date +%H:%M:%S) ==="