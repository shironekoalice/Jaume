#!/bin/bash
pkill -f train_lm.py 2>/dev/null
pkill -f chain5.sh 2>/dev/null
sleep 3
SNAP=/home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master
CKPT=/mnt/workspace/output/350m-causal-mlm/final
DATA=$SNAP/data/mlm_corpus.jsonl
OUT=/mnt/workspace/output
cd "$SNAP/code/repo"

# Matched block budget: 40000 blocks of 256 tokens = 10.24M tokens.
#   A  base      40000 blocks  loops=1            1x wall-clock
#   B  loop      40000 blocks  loops=4            4x wall-clock (same tokens)
#   C  loop-wc   10000 blocks  loops=4            1x wall-clock  <-- the fair one
#   D  loop-wc+align 10000 blocks loops=4 + align 1x wall-clock
common="--config configs/jaume-350m-loop.json --init-from $CKPT --data $DATA
        --epochs 1 --batch-size 16 --grad-accum 4 --max-length 256
        --lr 3e-4 --warmup-steps 200 --val-blocks 64 --eval-every 200
        --eval-batches 8 --log-every 100 --device npu"

run () {
  local name=$1; shift
  echo "=== [$name] $(date +%H:%M:%S) $* ==="
  python -u scripts/train_lm.py $common "$@" 2>&1 | tee $OUT/$name.log
  grep -a "FINAL VAL" $OUT/$name.log || true
  echo "=== [$name] done $(date +%H:%M:%S) ==="
}

run jaume-loop-A-base      --num-loops 1 --max-blocks 40000 --output $OUT/jaume-loop-A-base
run jaume-loop-B-loop      --num-loops 4 --max-blocks 40000 --output $OUT/jaume-loop-B-loop
run jaume-loop-C-loop-wc   --num-loops 4 --max-blocks 10000 --output $OUT/jaume-loop-C-loop-wc
run jaume-loop-D-loop-wc-align --num-loops 4 --max-blocks 10000 --loop-align-weight 0.1 --output $OUT/jaume-loop-D-loop-wc-align

echo "=== [E] probe the interior of the looped model ==="
python -u scripts/probe_loop.py --model $OUT/jaume-loop-B-loop \
  --num-loops 4 --sweep-loops 1 2 4 8 --data $DATA \
  --num-samples 32 --batch-size 4 --max-length 256 --device npu \
  --out $OUT/loop-probe-350m.json 2>&1 | tee $OUT/loop-probe-350m.log
echo "=== ALL DONE $(date +%H:%M:%S) ==="