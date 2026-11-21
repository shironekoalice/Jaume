#!/bin/bash
SNAP=/home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master
cd "$SNAP/code/repo"
python -u scripts/train_vlm_probe.py --model /mnt/workspace/output/350m-causal-mlm/final --n-train 500 --n-val 100 --max-shapes 6 --epochs 1 --batch-size 16 --lr 1e-3 --device npu --task color_count --unfreeze-layers 4 2>&1 | tail -18
