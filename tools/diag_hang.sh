#!/bin/bash
date +%H:%M:%S
echo "=== step check 1 ==="
grep -ac "step" /mnt/workspace/output/jaume-plan-P2-loop.log
echo "=== wait 90s ==="
sleep 90
date +%H:%M:%S
echo "=== step check 2 ==="
grep -ac "step" /mnt/workspace/output/jaume-plan-P2-loop.log
tail -3 /mnt/workspace/output/jaume-plan-P2-loop.log
echo "=== npu proc ==="
ps -o pid,pcpu,rss,stat,etime -p 104710
echo "=== py-spy? ==="
which py-spy 2>/dev/null || echo "no py-spy"