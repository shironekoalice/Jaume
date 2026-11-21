#!/bin/bash
# The 8.10 filter on a task attention cannot shortcut.
# If 16L beats 8L at 16 ops (and not at 8), we finally have a depth-limited
# regime where a loop could pay off.
SNAP=/home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master
B=/mnt/workspace/output/350m-causal-mlm/final
OUT=/mnt/workspace/output
cd "$SNAP/code/repo"

common="--config configs/jaume-350m-loop.json --init-from $B
        --n-tasks 30000 --epochs 1 --batch-size 24 --grad-accum 2
        --max-length 192 --lr 3e-4 --warmup-steps 100 --val-docs 256
        --eval-every 300 --log-every 300 --device npu"

run () {
  local name=$1; shift
  echo "=== [$name] $(date +%H:%M:%S) ==="
  python -u scripts/train_register.py $common "$@" 2>&1 | tee $OUT/$name.log
  grep -a "FINAL VAL" $OUT/$name.log || echo "NO FINAL"
  echo "=== [$name] done $(date +%H:%M:%S) ==="
}

run reg-o8-L8   --n-ops 8  --depth 8  --output $OUT/reg-o8-L8
run reg-o8-L16  --n-ops 8  --depth 16 --output $OUT/reg-o8-L16
run reg-o16-L8  --n-ops 16 --depth 8  --output $OUT/reg-o16-L8
run reg-o16-L16 --n-ops 16 --depth 16 --output $OUT/reg-o16-L16

echo "=== FILTER DONE $(date +%H:%M:%S) ==="