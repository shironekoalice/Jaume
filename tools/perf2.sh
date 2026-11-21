#!/bin/bash
echo "=== log tail ==="
tail -6 /mnt/workspace/output/jaume-plan-P2-loop.log
echo "=== AICore sampling ==="
for i in 1 2 3 4 5 6 7 8; do
  L=$(npu-smi info 2>&1 | sed -n '7p')
  echo "$(date +%H:%M:%S) $L"
  sleep 6
done
echo "=== progress ==="
grep -a "step " /mnt/workspace/output/jaume-plan-P2-loop.log | tail -3
echo "=== pid ==="
P=$(pgrep -f train_lm_planning | head -1)
ps -o pid,pcpu,rss,etime,stat -p $P