#!/bin/bash
# Full chain, sized to finish in ~3h at the measured 0.5 step/s, then shut down.
# P1 is already done (val 0.1540).  P2 = main loop run; P3 = same wall-clock;
# P4 = gate; P5 = probe.  All use batch 24 x accum 2 for NPU saturation.
SNAP=/home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master
B=/mnt/workspace/output/350m-causal-mlm/final
OUT=/mnt/workspace/output
cd "$SNAP/code/repo"

common="--config configs/jaume-350m-loop.json --init-from $B
        --n-constraints 8 --budget 12
        --epochs 1 --batch-size 24 --grad-accum 2 --max-length 192
        --lr 3e-4 --warmup-steps 100 --val-docs 256
        --eval-every 300 --log-every 150 --device npu"

run () {
  local name=$1; shift
  echo "=== [$name] $(date +%H:%M:%S) ==="
  python -u scripts/train_lm_planning.py $common "$@" 2>&1 | tee $OUT/$name.log
  grep -a "FINAL VAL" $OUT/$name.log || echo "NO FINAL"
  echo "=== [$name] done $(date +%H:%M:%S) ==="
}

# P2: loop, 12000 tasks (144k docs -> 3000 steps -> ~100 min)
run jaume-plan-P2-loop --num-loops 4 --n-tasks 12000 --output $OUT/jaume-plan-P2-loop
# P3: same wall-clock as P1 (P1 did 40000 tasks at depth 16; depth 64 is 4x
#     the compute, so 10000 tasks is the matched budget)
run jaume-plan-P3-loop-wc --num-loops 4 --n-tasks 10000 --output $OUT/jaume-plan-P3-loop-wc
# P4: gate, smaller still so it finishes
run jaume-plan-P4-gate --num-loops 4 --loop-gate --n-tasks 8000 --output $OUT/jaume-plan-P4-gate

echo "=== [P5] probe ==="
python -u scripts/probe_loop.py --model $OUT/jaume-plan-P2-loop \
  --num-loops 4 --sweep-loops 1 2 4 8 --data $SNAP/data/mlm_corpus.jsonl \
  --num-samples 24 --batch-size 4 --max-length 192 --device npu \
  --out $OUT/loop-probe-planning.json 2>&1 | tee $OUT/loop-probe-planning.log

echo "=== ALL DONE $(date +%H:%M:%S) ==="
sync
sleep 5
echo "=== SHUTTING DOWN ==="
sudo -n shutdown -h now