#!/bin/bash
nohup setsid bash /tmp/chain14.sh > /mnt/workspace/chain14.log 2>&1 &
echo "PID=$!"
sleep 90
tail -5 /mnt/workspace/output/reg-o8-L8.log 2>/dev/null || tail -5 /mnt/workspace/chain14.log