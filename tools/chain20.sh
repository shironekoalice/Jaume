#!/bin/bash
SNAP=/home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master
B=/mnt/workspace/output/350m-causal-mlm/final
OUT=/mnt/workspace/output
cd "$SNAP/code/repo" || exit 1
run () {
  local name=$1; shift
  echo "=== [$name] $(date +%H:%M:%S) ==="
  python -u scripts/train_vlm_probe.py --model $B --n-train 12000 --n-val 1200 \
    --max-shapes 6 --epochs 8 --batch-size 32 --lr 1e-3 --device npu \
    --task parity --unfreeze-layers 4 "$@" \
    2>&1 | tee $OUT/$name.log | grep -aE '\[vlm\] (ep|depth|train=|generating)'
  echo "=== [$name] rc=${PIPESTATUS[0]} $(date +%H:%M:%S) ==="
}
echo '=== SMOKE parity depth=8 ==='
python -u scripts/train_vlm_probe.py --model $B --n-train 256 --n-val 64 \
  --max-shapes 6 --epochs 1 --batch-size 32 --lr 1e-3 --device npu \
  --task parity --unfreeze-layers 4 --depth 8 \
  > $OUT/vlmP-smoke.log 2>&1
rc=$?
tail -8 $OUT/vlmP-smoke.log
if [ $rc -ne 0 ]; then echo "=== SMOKE FAILED rc=$rc, ABORT ==="; exit 1; fi
echo '=== SMOKE OK ==='
for D in 8 16 32; do run vlmP-L$D --depth $D; done
echo "=== ALL DONE $(date +%H:%M:%S) ==="
