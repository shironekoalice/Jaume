#!/bin/bash
nohup bash /tmp/chain8.sh > /mnt/workspace/chain8.log 2>&1 &
echo "chain8 PID=$!"
sleep 30
tail -6 /mnt/workspace/output/jaume-plan-P1-base.log 2>/dev/null