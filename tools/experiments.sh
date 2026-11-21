#!/bin/bash
# Sequential experiment runner for the Ascend box.
# Usage: bash /tmp/experiments.sh <mode>
#   mode = ce-split | ce-joint | diag | head | gate | all
SNAP=/home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master
CKPT=/mnt/workspace/output/350m-causal-mlm/final
cd "$SNAP/code/repo"
mkdir -p /mnt/workspace/output

run_ce () {
  local enc=$1
  echo "=== CE / $enc ==="
  python -u scripts/plan_ce_study.py --model "$CKPT" \
    --encoding "$enc" --n-constraints 8 --train-tasks 300 --test-tasks 100 \
    --budget 30 --width 3 --epochs 30 --device npu \
    --out "/mnt/workspace/output/plan-ce-$enc.json" \
    2>&1 | tee "/mnt/workspace/plan-ce-$enc.log"
}

case "$1" in
  ce-split) run_ce split ;;
  ce-joint) run_ce joint ;;
  diag)
    python -u scripts/plan_diag.py --model "$CKPT" --n-constraints 8 \
      --train-tasks 300 --test-tasks 50 --budget 30 --width 3 --epochs 30 \
      --device npu --out /mnt/workspace/output/plan-diag.json \
      2>&1 | tee /mnt/workspace/plan-diag.log ;;
  head)
    python -u scripts/plan_head_study.py --model "$CKPT" --n-constraints 8 \
      --train-tasks 300 --test-tasks 100 --budget 30 --width 3 --epochs 30 \
      --device npu --out /mnt/workspace/output/plan-head-study.json \
      2>&1 | tee /mnt/workspace/plan-head-study.log ;;
  all)
    run_ce split
    run_ce joint
    ;;
  *) echo "unknown mode: $1"; exit 1 ;;
esac
echo "=== DONE $1 ==="