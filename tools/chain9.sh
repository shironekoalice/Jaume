#!/bin/bash
# P2/P3/P4 with memory-safe settings for the 4x loop.
# The loop multiplies activation memory by T (4x here), so the batch must
# shrink accordingly.  Using batch 8 + grad_accum 8 keeps the effective
# batch at 64 (same as before) while cutting peak activation by 4x.
SNAP=/home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master
B=/mnt/workspace/output/350m-causal-mlm/final
OUT=/mnt/workspace/output
cd "$SNAP/code/repo"

common="--config configs/jaume-350m-loop.json --init-from $B
        --n-tasks 40000 --n-constraints 8 --budget 12
        --epochs 1 --batch-size 8 --grad-accum 8 --max-length 192
        --lr 3e-4 --warmup-steps 200 --val-docs 512
        --eval-every 400 --log-every 200 --device npu"

run () {
  local name=$1; shift
  echo "=== [$name] $(date +%H:%M:%S) ==="
  python -u scripts/train_lm_planning.py $common "$@" 2>&1 | tee $OUT/$name.log
  grep -a "FINAL VAL" $OUT/$name.log || true
  echo "=== [$name] done $(date +%H:%M:%S) ==="
}

run jaume-plan-P2-loop   --num-loops 4 --output $OUT/jaume-plan-P2-loop
run jaume-plan-P3-loop-wc --num-loops 4 --n-tasks 10000 --output $OUT/jaume-plan-P3-loop-wc
run jaume-plan-P4-gate    --num-loops 4 --loop-gate --output $OUT/jaume-plan-P4-gate

echo "=== [P5] probe the looped model ==="
python -u scripts/probe_loop.py --model $OUT/jaume-plan-P2-loop \
  --num-loops 4 --sweep-loops 1 2 4 8 --data $SNAP/data/mlm_corpus.jsonl \
  --num-samples 32 --batch-size 4 --max-length 192 --device npu \
  --out $OUT/loop-probe-planning.json 2>&1 | tee $OUT/loop-probe-planning.log
echo "=== ALL DONE $(date +%H:%M:%S) ==="