#!/bin/bash
# A-plan: loop on a task where rounds have different semantics.
# Judgement: the round-distance decay must be SLOWER than the 0.6 measured
# on generic text, and the gate must learn a NON-uniform assignment.
SNAP=/home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master
B=/mnt/workspace/output/350m-causal-mlm/final
OUT=/mnt/workspace/output
cd "$SNAP/code/repo"

common="--config configs/jaume-350m-loop.json --init-from $B
        --n-tasks 40000 --n-constraints 8 --budget 12
        --epochs 1 --batch-size 32 --grad-accum 2 --max-length 256
        --lr 3e-4 --warmup-steps 200 --val-docs 512
        --eval-every 300 --log-every 150 --device npu"

run () {
  local name=$1; shift
  echo "=== [$name] $(date +%H:%M:%S) ==="
  python -u scripts/train_lm_planning.py $common "$@" 2>&1 | tee $OUT/$name.log
  grep -a "FINAL VAL" $OUT/$name.log || true
  echo "=== [$name] done $(date +%H:%M:%S) ==="
}

# P1: depth-16 reference (no loop) on the planning task
run jaume-plan-P1-base   --num-loops 1 --output $OUT/jaume-plan-P1-base
# P2: depth-64 loop, same tokens (4x compute) -- does the loop help AT ALL here
run jaume-plan-P2-loop   --num-loops 4 --output $OUT/jaume-plan-P2-loop
# P3: same wall-clock as P1 -- the fair comparison
run jaume-plan-P3-loop-wc --num-loops 4 --n-tasks 10000 --output $OUT/jaume-plan-P3-loop-wc
# P4: gate, to see whether the optimiser now WANTS non-uniform rounds
run jaume-plan-P4-gate    --num-loops 4 --loop-gate --n-tasks 40000 --output $OUT/jaume-plan-P4-gate

# Probe the looped model: is the decay slower than 0.6?
python -u scripts/probe_loop.py --model $OUT/jaume-plan-P2-loop \
  --num-loops 4 --sweep-loops 1 2 4 8 --data $SNAP/data/mlm_corpus.jsonl \
  --num-samples 32 --batch-size 4 --max-length 256 --device npu \
  --out $OUT/loop-probe-planning.json 2>&1 | tee $OUT/loop-probe-planning.log
echo "=== ALL DONE $(date +%H:%M:%S) ==="