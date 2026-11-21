#!/bin/bash
cd /home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master/code/repo || exit 1
export CFG=/home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master/code/repo/configs/jaume-350m-causal.json
python -u scripts/train_carry.py --config /home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master/code/repo/configs/jaume-350m-causal.json --init-from /mnt/workspace/output/350m-causal-mlm/final --n-digits 8 --n-train 20000 --n-val 300 --epochs 8 --batch-size 32 --grad-accum 1 --lr 3e-4 --max-length 128 --eval-every 200 --log-every 100 --output /mnt/workspace/output/carry8-L8 --depth 8 --device npu > /mnt/workspace/output/carry8-L8.log 2>&1
echo L8_DONE
python -u scripts/train_carry.py --config /home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master/code/repo/configs/jaume-350m-causal.json --init-from /mnt/workspace/output/350m-causal-mlm/final --n-digits 8 --n-train 20000 --n-val 300 --epochs 8 --batch-size 32 --grad-accum 1 --lr 3e-4 --max-length 128 --eval-every 200 --log-every 100 --output /mnt/workspace/output/carry8-L16 --depth 16 --device npu > /mnt/workspace/output/carry8-L16.log 2>&1
echo L16_DONE
