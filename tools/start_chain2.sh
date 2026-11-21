#!/bin/bash
pkill -f plan_ce_study 2>/dev/null
pkill -f chain.sh 2>/dev/null
sleep 2
nohup bash /tmp/chain2.sh > /mnt/workspace/chain2.log 2>&1 &
echo "chain2 PID=$!"
sleep 20
echo "--- chain2 log ---"
tail -8 /mnt/workspace/chain2.log
echo "--- ft log ---"
tail -5 /mnt/workspace/plan-ft-full.log 2>/dev/null || echo "no ft log yet"