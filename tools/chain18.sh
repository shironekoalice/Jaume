#!/bin/bash
SNAP=/home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master
B=/mnt/workspace/output/350m-causal-mlm/final
OUT=/mnt/workspace/output
cd "$SNAP/code/repo"
run () {
  local name=$1; shift
  echo "=== [$name] $(date +%H:%M:%S) ==="
  python -u scripts/train_vlm_probe.py --model $B --n-train 6000 --n-val 600 \
    --max-shapes 6 --epochs 6 --batch-size 32 --lr 1e-3 --device npu "$@" \
    2>&1 | tee $OUT/$name.log
  echo "=== [$name] done $(date +%H:%M:%S) ==="
}
# B1: ResNet18 stem (random init), same counting task
run vlmB-resnet18 --visual-backbone resnet18 --unfreeze-layers 4
# B2: colour-conditioned counting (3 colours x 3 shapes = 9 queries)
run vlmB-color --task color_count --unfreeze-layers 4
echo "=== DONE $(date +%H:%M:%S) ==="