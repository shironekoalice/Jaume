#!/bin/bash
pkill -f train_lm.py 2>/dev/null; pkill -f chain7.sh 2>/dev/null; sleep 3
nohup bash /tmp/chain7.sh > /mnt/workspace/chain7.log 2>&1 &
echo "chain7 restarted PID=$!"
sleep 25
tail -4 /mnt/workspace/output/jaume-loop-E-gate-wc.log 2>/dev/null