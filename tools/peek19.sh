#!/bin/bash
date +%H:%M:%S
for f in vlmB-resnet18 vlmB-color; do
  echo "--- $f ---"
  grep -a "ep " /mnt/workspace/output/$f.log 2>/dev/null | tail -3
done
grep -aE "^===" /mnt/workspace/chain18.log 2>/dev/null | tail -4
echo "procs=$(pgrep -c -f train_vlm_probe)"
