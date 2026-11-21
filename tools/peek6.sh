#!/bin/bash
date +%H:%M:%S
pgrep -c -f train_lm.py
echo "--- A log tail ---"
tail -8 /mnt/workspace/output/jaume-loop-A-base.log