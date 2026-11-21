#!/bin/bash
source /opt/home/developer/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1
source /opt/home/developer/Ascend/nnal/atb/set_env.sh >/dev/null 2>&1
SNAP=/home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master
cd "$SNAP/code/repo" || exit 1
python -u scripts/train_vision_suite.py --model /mnt/workspace/output/350m-causal-mlm/final --n-train 384 --n-val 96 --epochs 2 --batch-size 32 --depth 16 --unfreeze-layers 4 --device npu > /mnt/workspace/output/vis-smoke.log 2>&1
echo RC=$?
