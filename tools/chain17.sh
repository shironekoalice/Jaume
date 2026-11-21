#!/bin/bash
SNAP=/home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master
B=/mnt/workspace/output/350m-causal-mlm/final
OUT=/mnt/workspace/output
cd "$SNAP/code/repo"
common="--model $B --n-train 6000 --n-val 600 --max-shapes 6 --epochs 8 --batch-size 32 --lr 1e-3 --device npu"
run () {
  local name=$1; shift
  echo "=== [$name] $(date +%H:%M:%S) ==="
  python -u scripts/train_vlm_probe.py $common "$@" 2>&1 | tee $OUT/$name.log
  echo "=== [$name] done $(date +%H:%M:%S) ==="
}
run vlm2-frozen --unfreeze-layers 0
run vlm2-top4   --unfreeze-layers 4
echo "=== DONE $(date +%H:%M:%S) ==="