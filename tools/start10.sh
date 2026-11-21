#!/bin/bash
pkill -f train_lm_planning 2>/dev/null; sleep 2
nohup setsid bash /tmp/chain10.sh > /mnt/workspace/chain10.log 2>&1 &
echo "chain10 PID=$!"
sleep 60
tail -4 /mnt/workspace/output/jaume-plan-P2-loop.log 2>/dev/null