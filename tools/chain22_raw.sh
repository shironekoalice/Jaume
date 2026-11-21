#!/bin/bash
SNAP=/home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master
B=/mnt/workspace/output/350m-causal-mlm/final
CFG=$SNAP/code/repo/configs/jaume-350m-causal.json
OUT=/mnt/workspace/output
cd "$SNAP/code/repo" || exit 1
run () {
  local name=$1; shift
  echo "=== [$name] $(date +%H:%M:%S) ==="
  python -u scripts/train_carry.py --config $CFG --init-from $B \
    --n-digits 8 --n-train 20000 --n-val 300 --epochs 3 \
    --batch-size 32 --grad-accum 1 --lr 3e-4 --max-length 128 \
    --eval-every 300 --log-every 100 \
    --output $OUT/carry-$name --device npu "$@" \
    2>&1 | tee $OUT/carry-$name.log | grep -aE '\[carry\] (VAL_ACC|FINAL_ACC|params|train=|dense|warm)'
  echo "=== [$name] rc=${PIPESTATUS[0]} $(date +%H:%M:%S) ==="
}
echo '=== SMOKE depth8 ==='
python -u scripts/train_carry.py --config $CFG --init-from $B \
  --n-digits 4 --n-train 128 --n-val 32 --depth 8 --epochs 1 \
  --batch-size 16 --max-length 64 --eval-every 5 --log-every 5 \
  --output $OUT/carry-smoke --device npu > $OUT/carry-smoke.log 2>&1
rc=$?; tail -8 $OUT/carry-smoke.log
if [ $rc -ne 0 ]; then echo "=== SMOKE FAILED rc=$rc ==="; exit 1; fi
echo '=== SMOKE OK ==='
run L8 --depth 8
run L16 --depth 16
echo "=== ALL DONE $(date +%H:%M:%S) ==="
