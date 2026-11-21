#!/bin/bash
SNAP=/home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master
cd "$SNAP/code/repo"
python -u scripts/train_vlm_probe.py --model /mnt/workspace/output/350m-causal-mlm/final --n-train 12000 --n-val 1200 --max-shapes 6 --epochs 6 --batch-size 32 --lr 1e-3 --device npu --task color_count --unfreeze-layers 4 2>&1 | tee /mnt/workspace/output/vlmB-color.log
