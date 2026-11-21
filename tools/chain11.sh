#!/bin/bash
# Two-stage derive/place task: stage 2 depends on stage 1's conclusion.
# Judgement unchanged: same-wall-clock D2 must beat D1, and the probe's
# round-distance must be SLOWER than the single-step planning task's 0.45.
SNAP=/home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master
B=/mnt/workspace/output/350m-causal-mlm/final
OUT=/mnt/workspace/output
cd "$SNAP/code/repo"

common="--config configs/jaume-350m-loop.json --init-from $B
        --n-items 6 --n-slots 4 --n-fixed 3 --n-same 2 --n-notsame 3
        --epochs 1 --batch-size 24 --grad-accum 2 --max-length 160
        --lr 3e-4 --warmup-steps 100 --val-docs 256
        --eval-every 300 --log-every 150 --device npu"

run () {
  local name=$1; shift
  echo "=== [$name] $(date +%H:%M:%S) ==="
  python -u scripts/train_lm_derive.py $common "$@" 2>&1 | tee $OUT/$name.log
  grep -a "FINAL VAL" $OUT/$name.log || echo "NO FINAL"
  echo "=== [$name] done $(date +%H:%M:%S) ==="
}

# D1: depth-16 reference on the two-stage task
run derive-D1-base   --num-loops 1 --n-tasks 40000 --output $OUT/derive-D1-base
# D2: depth-64, same wall-clock as D1 (1/4 the data)
run derive-D2-loop-wc --num-loops 4 --n-tasks 10000 --output $OUT/derive-D2-loop-wc
# D3: depth-64, same tokens as D2's competitor (4x compute) - does loop help AT ALL
run derive-D3-loop   --num-loops 4 --n-tasks 40000 --output $OUT/derive-D3-loop
# D4: gate on the two-stage task
run derive-D4-gate   --num-loops 4 --loop-gate --n-tasks 40000 --output $OUT/derive-D4-gate

echo "=== [D5] probe the two-stage model ==="
python -u scripts/probe_loop.py --model $OUT/derive-D3-loop \
  --num-loops 4 --sweep-loops 1 2 4 8 --data $SNAP/data/mlm_corpus.jsonl \
  --num-samples 24 --batch-size 4 --max-length 160 --device npu \
  --out $OUT/loop-probe-derive.json 2>&1 | tee $OUT/loop-probe-derive.log
echo "=== ALL DONE $(date +%H:%M:%S) ==="