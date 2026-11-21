#!/bin/bash
nohup bash /tmp/chain7.sh > /mnt/workspace/chain7.log 2>&1 &
echo "chain7 PID=$!"
sleep 20
tail -5 /mnt/workspace/output/jaume-loop-E-gate-wc.log 2>/dev/null || echo starting