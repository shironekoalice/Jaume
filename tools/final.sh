#!/bin/bash
date +%H:%M:%S
uptime
ls -lh /mnt/workspace/output/jaume-plan-P*.log /mnt/workspace/output/loop-probe-planning.json 2>/dev/null
for f in P1-base P2-loop P3-loop-wc P4-gate; do
  printf "%-14s " "$f"
  grep -a "FINAL VAL" /mnt/workspace/output/jaume-plan-$f.log 2>/dev/null || echo "(none)"
done