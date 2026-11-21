#!/bin/bash
nohup setsid bash /tmp/chain16.sh > /mnt/workspace/chain16.log 2>&1 &
echo "PID=$!"
sleep 120
tail -6 /mnt/workspace/output/vlm-V1-frozen.log 2>/dev/null || tail -6 /mnt/workspace/chain16.log