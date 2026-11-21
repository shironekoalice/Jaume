#!/bin/bash
# The 8.10 filter: does chain length create a depth-limited regime?
# If 16L beats 8L at 12 hops (and not at 3 hops), we finally have a task
# where a loop could pay off.
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
  python -u scripts/train_chain.py $common "$@" 2>&1 | tee $OUT/$name.log
  grep -a "FINAL VAL" $OUT/$name.log || echo "NO FINAL"
  echo "=== [$name] done $(date +%H:%M:%S) ==="
}

# short chain: should saturate at 8L (control)
run chain-h3-L8  --hops 3  --depth 8  --output $OUT/chain-h3-L8
run chain-h3-L16 --hops 3  --depth 16 --output $OUT/chain-h3-L16
# long chain: if depth matters, 16L should beat 8L here
run chain-h12-L8  --hops 12 --depth 8  --output $OUT/chain-h12-L8
run chain-h12-L16 --hops 12 --depth 16 --output $OUT/chain-h12-L16

echo "=== FILTER DONE $(date +%H:%M:%S) ==="