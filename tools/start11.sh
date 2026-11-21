#!/bin/bash
nohup setsid bash /tmp/chain11.sh > /mnt/workspace/chain11.log 2>&1 &
echo "chain11 PID=$!"
sleep 90
tail -6 /mnt/workspace/output/derive-D1-base.log 2>/dev/null