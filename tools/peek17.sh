#!/bin/bash
date +%H:%M:%S
grep -aE "^===|VAL loss" /mnt/workspace/chain16.log 2>/dev/null | tail -24
echo "procs=$(pgrep -c -f train_vlm_probe)"