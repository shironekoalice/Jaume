#!/bin/bash
source /opt/home/developer/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1
source /opt/home/developer/Ascend/nnal/atb/set_env.sh >/dev/null 2>&1
SNAP=/home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master
cd "$SNAP/code/repo" || exit 1
python -u scripts/train_vision_suite.py --model /mnt/workspace/output/350m-causal-mlm/final --n-train 9000 --n-val 1200 --epochs 12 --batch-size 32 --depth 16 --unfreeze-layers 4 --device npu --out /mnt/workspace/output/vision-suite-v1 > /mnt/workspace/output/vis-full.log 2>&1
echo RC=$?
