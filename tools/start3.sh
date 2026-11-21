#!/bin/bash
nohup bash /tmp/chain3.sh > /mnt/workspace/chain3.log 2>&1 &
echo "chain3 PID=$!"
sleep 25
tail -6 /mnt/workspace/plan-ft-full.log 2>/dev/null