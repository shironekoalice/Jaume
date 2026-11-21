#!/bin/bash
date +%H:%M:%S
pgrep -af train_lm.py | head -2
echo "--- B log ---"
tail -8 /mnt/workspace/output/jaume-loop-B-loop.log 2>/dev/null || echo "not started"
echo "--- chain6 ---"
tail -4 /mnt/workspace/chain6.log 2>/dev/null