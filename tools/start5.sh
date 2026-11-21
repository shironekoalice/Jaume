#!/bin/bash
SNAP=/home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master
cd "$SNAP/code/repo"
ls -l scripts/train_lm.py scripts/probe_loop.py jaume/modeling/loop_lm.py configs/jaume-350m-loop.json
ls -l /home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master/data/mlm_corpus.jsonl
nohup bash /tmp/chain5.sh > /mnt/workspace/chain5.log 2>&1 &
echo "chain5 PID=$!"