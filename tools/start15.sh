#!/bin/bash
nohup setsid bash /tmp/chain15.sh > /mnt/workspace/chain15.log 2>&1 &
echo "PID=$!"
sleep 120
tail -4 /mnt/workspace/output/st-n256-L8.log 2>/dev/null || tail -6 /mnt/workspace/chain15.log