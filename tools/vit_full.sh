#!/bin/bash
source /opt/home/developer/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1
source /opt/home/developer/Ascend/nnal/atb/set_env.sh >/dev/null 2>&1
SNAP=/home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master
W=/home/developer/.cache/modelscope/models/timm--vit_small_patch16_224.augreg_in1k/snapshots/master/model.safetensors
cd "$SNAP/code/repo" || exit 1
run () {
  local tag=$1; shift
  echo "=== [$tag] $(date +%H:%M:%S) ==="
  python -u scripts/train_vision_vit.py --model /mnt/workspace/output/350m-causal-mlm/final \
    --n-train 9000 --n-val 1200 --epochs 10 --batch-size 24 --depth 16 \
    --unfreeze-layers 4 --arch vit_small_patch16_224 --weights "$W" \
    --device npu --tag "$tag" --out /mnt/workspace/output/vision-$tag "$@" \
    > /mnt/workspace/output/vit-$tag.log 2>&1
  echo "=== [$tag] rc=$? ==="
}
run vit-frozen
run vit-top4 --vit-unfreeze-blocks 4 --vit-lr 5e-5
echo ALL_DONE
