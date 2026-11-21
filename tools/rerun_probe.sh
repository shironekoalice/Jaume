#!/bin/bash
SNAP=/home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master
cd "$SNAP/code/repo"
python -u scripts/probe_loop.py --model /mnt/workspace/output/jaume-loop-F-gate \
  --num-loops 4 --sweep-loops 1 2 4 8 --data $SNAP/data/mlm_corpus.jsonl \
  --num-samples 32 --batch-size 4 --max-length 256 --device npu \
  --out /mnt/workspace/output/loop-probe-gated.json 2>&1 | tee /mnt/workspace/output/loop-probe-gated.log
echo DONE