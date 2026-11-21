#!/bin/bash
nohup setsid bash /tmp/chain18.sh > /mnt/workspace/chain18.log 2>&1 &
echo "PID=$!"
sleep 150
tail -6 /mnt/workspace/output/vlmB-resnet18.log 2>/dev/null || tail -8 /mnt/workspace/chain18.log