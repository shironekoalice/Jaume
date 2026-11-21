#!/bin/bash
date +%H:%M:%S
echo "=== resnet18 ==="
grep -a "ep " /mnt/workspace/output/vlmB-resnet18.log | tail -2
grep -a "ep " /mnt/workspace/output/vlmB-color.log | tail -4
echo "procs=$(pgrep -c -f train_vlm_probe)"
