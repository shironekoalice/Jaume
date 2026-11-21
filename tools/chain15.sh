#!/bin/bash
# Depth filter on sequential reduction: does longer input create depth demand?
SNAP=/home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master
B=/mnt/workspace/output/350m-causal-mlm/final
OUT=/mnt/workspace/output
cd "$SNAP/code/repo"

common="--config configs/jaume-350m-loop.json --init-from $B
        --epochs 1 --batch-size 8 --grad-accum 2 --lr 3e-4
        --warmup-steps 60 --val-docs 128 --eval-every 200 --log-every 200
        --device npu"

run () {
  local name=$1; shift
  echo "=== [$name] $(date +%H:%M:%S) ==="
  python -u scripts/train_stream.py $common "$@" 2>&1 | tee $OUT/$name.log
  grep -a "FINAL VAL" $OUT/$name.log || echo "NO FINAL"
  echo "=== [$name] done $(date +%H:%M:%S) ==="
}

run st-n256-L8   --n-tasks 6000 --n-items 256  --depth 8  --max-length 1536 --output $OUT/st-n256-L8
run st-n256-L16  --n-tasks 6000 --n-items 256  --depth 16 --max-length 1536 --output $OUT/st-n256-L16
run st-n64-L8    --n-tasks 6000 --n-items 64   --depth 8  --max-length 512  --output $OUT/st-n64-L8
run st-n64-L16   --n-tasks 6000 --n-items 64   --depth 16 --max-length 512  --output $OUT/st-n64-L16

echo "=== STREAM FILTER DONE $(date +%H:%M:%S) ==="