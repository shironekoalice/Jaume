#!/bin/bash
nohup setsid bash /tmp/chain12.sh > /mnt/workspace/chain12.log 2>&1 &
echo "chain12 PID=$!"
sleep 60
tail -5 /mnt/workspace/output/derive-L8.log 2>/dev/null || tail -5 /mnt/workspace/chain12.log