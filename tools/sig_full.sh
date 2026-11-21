#!/bin/bash
source /opt/home/developer/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1
source /opt/home/developer/Ascend/nnal/atb/set_env.sh >/dev/null 2>&1
SNAP=/home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master
S=/home/developer/.cache/modelscope/models/google--siglip2-base-patch16-224/snapshots/master
cd "$SNAP/code/repo" || exit 1
run () {
  local tag=$1; shift
  echo "=== [$tag] $(date +%H:%M:%S) ==="
  python -u scripts/train_vision_siglip.py --model /mnt/workspace/output/350m-causal-mlm/final \
    --siglip-path "$S" --n-train 9000 --n-val 1200 --epochs 10 --batch-size 16 \
    --depth 16 --unfreeze-layers 4 --device npu --tag "$tag" \
    --out /mnt/workspace/output/vision-$tag "$@" \
    > /mnt/workspace/output/sig-$tag.log 2>&1
  echo "=== [$tag] rc=$? $(date +%H:%M:%S) ==="
}
run sigfrozen
run sigtop4 --siglip-unfreeze-blocks 4 --siglip-lr 2e-5
echo ALL_DONE
