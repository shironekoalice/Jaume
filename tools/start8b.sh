#!/bin/bash
pkill -f train_lm_planning 2>/dev/null; pkill -f chain8.sh 2>/dev/null; sleep 3
rm -f /mnt/workspace/output/jaume-plan-P*.log
nohup bash /tmp/chain8.sh > /mnt/workspace/chain8.log 2>&1 &
echo "chain8 restarted PID=$!"
sleep 90
echo "=== P1 ==="
tail -6 /mnt/workspace/output/jaume-plan-P1-base.log 2>/dev/null