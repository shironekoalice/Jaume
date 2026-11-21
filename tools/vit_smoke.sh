#!/bin/bash
source /opt/home/developer/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1
source /opt/home/developer/Ascend/nnal/atb/set_env.sh >/dev/null 2>&1
SNAP=/home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master
W=/home/developer/.cache/modelscope/models/timm--vit_small_patch16_224.augreg_in1k/snapshots/master/model.safetensors
cd "$SNAP/code/repo" || exit 1
python -u scripts/train_vision_vit.py --model /mnt/workspace/output/350m-causal-mlm/final \
  --n-train 256 --n-val 64 --epochs 1 --batch-size 8 --depth 16 \
  --unfreeze-layers 4 --arch vit_small_patch16_224 --weights "$W" \
  --device npu --tag smoke > /mnt/workspace/output/vit-smoke.log 2>&1
echo RC=$?
grep -aE "loaded|trainable|FINAL|ep 1/1|Error|Traceback" /mnt/workspace/output/vit-smoke.log | head -12
