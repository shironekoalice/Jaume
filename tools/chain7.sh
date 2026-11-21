#!/bin/bash
# Break-the-contraction round: gate + same-wall-clock, against A.
#   E  loop-gate-wc : loops=4, gate ON, 10k blocks (same wall-clock as A)
#   F  loop-gate    : loops=4, gate ON, 40k blocks (same tokens as B)
# Judgement: E must beat A (4.9406) to justify the gate.
SNAP=/home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master
CKPT=/mnt/workspace/output/350m-causal-mlm/final
DATA=$SNAP/data/mlm_corpus.jsonl
OUT=/mnt/workspace/output
cd "$SNAP/code/repo"

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

run jaume-loop-E-gate-wc   --num-loops 4 --loop-gate --max-blocks 10000 --output $OUT/jaume-loop-E-gate-wc
run jaume-loop-F-gate      --num-loops 4 --loop-gate --max-blocks 40000 --output $OUT/jaume-loop-F-gate

echo "=== [G] probe the gated model ==="
python -u scripts/probe_loop.py --model $OUT/jaume-loop-F-gate \
  --num-loops 4 --sweep-loops 1 2 4 8 --data $DATA \
  --num-samples 32 --batch-size 4 --max-length 256 --device npu \
  --out $OUT/loop-probe-gated.json 2>&1 | tee $OUT/loop-probe-gated.log
echo "=== ALL DONE $(date +%H:%M:%S) ==="