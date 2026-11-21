#!/bin/bash
pkill -f train_lm_planning 2>/dev/null; pkill -f chain9.sh 2>/dev/null; sleep 3
rm -f /mnt/workspace/output/jaume-plan-P2-loop.log /mnt/workspace/output/jaume-plan-P3-loop-wc.log /mnt/workspace/output/jaume-plan-P4-gate.log
nohup bash /tmp/chain9.sh > /mnt/workspace/chain9.log 2>&1 &
echo "chain9 restarted PID=$!"
sleep 120
tail -6 /mnt/workspace/output/jaume-plan-P2-loop.log 2>/dev/null
echo "=== npu ==="
npu-smi info 2>&1 | sed -n '7p'