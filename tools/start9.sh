#!/bin/bash
nohup bash /tmp/chain9.sh > /mnt/workspace/chain9.log 2>&1 &
echo "chain9 PID=$!"
sleep 60
tail -5 /mnt/workspace/output/jaume-plan-P2-loop.log 2>/dev/null