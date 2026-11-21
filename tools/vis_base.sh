#!/bin/bash
source /opt/home/developer/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1
source /opt/home/developer/Ascend/nnal/atb/set_env.sh >/dev/null 2>&1
SNAP=/home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master
cd "$SNAP/code/repo" || exit 1
python -u scripts/vision_baseline.py --n-train 9000 --n-val 1200 --epochs 12 --batch-size 64 --lr 2e-3 --device npu > /mnt/workspace/output/vis-base.log 2>&1
echo RC=$?
