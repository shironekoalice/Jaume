#!/bin/bash
nohup setsid bash /tmp/chain13.sh > /mnt/workspace/chain13.log 2>&1 &
echo "PID=$!"
sleep 90
tail -5 /mnt/workspace/output/chain-h3-L8.log 2>/dev/null || tail -5 /mnt/workspace/chain13.log