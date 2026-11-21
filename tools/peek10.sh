#!/bin/bash
date +%H:%M:%S
echo "=== chain9 tail ==="
tail -6 /mnt/workspace/chain9.log 2>/dev/null
echo "=== P2 progress ==="
tail -8 /mnt/workspace/output/jaume-plan-P2-loop.log 2>/dev/null
echo "=== procs ==="
pgrep -af train_lm_planning | head -2
echo "=== npu ==="
npu-smi info 2>&1 | sed -n '6,8p'