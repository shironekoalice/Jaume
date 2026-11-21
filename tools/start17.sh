#!/bin/bash
nohup setsid bash /tmp/chain17.sh > /mnt/workspace/chain17.log 2>&1 &
echo "PID=$!"
sleep 120
tail -5 /mnt/workspace/output/vlm2-frozen.log 2>/dev/null || tail -5 /mnt/workspace/chain17.log