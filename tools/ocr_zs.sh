#!/bin/bash
source /opt/home/developer/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1
source /opt/home/developer/Ascend/nnal/atb/set_env.sh >/dev/null 2>&1
SNAP=/home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master
S=/home/developer/.cache/modelscope/models/google--siglip2-base-patch16-224/snapshots/master
cd "$SNAP/code/repo" || exit 1
python -u scripts/train_ocr_siglip.py --model /mnt/workspace/output/350m-causal-mlm/final --siglip-path "$S" --n-train 192 --n-val 240 --epochs 1 --batch-size 8 --depth 16 --unfreeze-layers 4 --img-size 320 --zero-shot --device npu --tag zs > /mnt/workspace/output/ocr-zs.log 2>&1
echo RC=$?
