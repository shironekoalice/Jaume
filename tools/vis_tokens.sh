#!/bin/bash
source /opt/home/developer/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1
source /opt/home/developer/Ascend/nnal/atb/set_env.sh >/dev/null 2>&1
SNAP=/home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master
cd "$SNAP/code/repo" || exit 1
run () {
  local name=$1; shift
  echo "=== [$name] $(date +%H:%M:%S) ==="
  python -u scripts/train_vision_suite.py --model /mnt/workspace/output/350m-causal-mlm/final \
    --n-train 9000 --n-val 1200 --epochs 10 --batch-size 32 --depth 16 \
    --unfreeze-layers 4 --device npu --out /mnt/workspace/output/vision-$name "$@" \
    > /mnt/workspace/output/vis-$name.log 2>&1
  echo "=== [$name] rc=$? ==="
  grep -a "FINAL PROFILE" -A 8 /mnt/workspace/output/vis-$name.log
}
run tok192 --n-tokens 192
run tok192-r192 --n-tokens 192 --res 192
echo ALL_DONE
