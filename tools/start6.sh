#!/bin/bash
nohup bash /tmp/chain6.sh > /mnt/workspace/chain6.log 2>&1 &
echo "chain6 PID=$!"
sleep 25
tail -8 /mnt/workspace/output/jaume-loop-A-base.log 2>/dev/null