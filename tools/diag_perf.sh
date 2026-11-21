#!/bin/bash
echo "=== AICore sampling over 60s ==="
for i in 1 2 3 4 5 6; do
  npu-smi info 2>&1 | sed -n '7p' | awk '{print $4}'
  sleep 10
done
echo "=== progress rate ==="
date +%H:%M:%S
grep -a "step" /mnt/workspace/output/jaume-plan-P2-loop.log | tail -2
sleep 120
date +%H:%M:%S
grep -a "step" /mnt/workspace/output/jaume-plan-P2-loop.log | tail -2
echo "=== host CPU of the trainer ==="
ps -o pid,pcpu,rss,etime -p 104710