#!/bin/bash
date +%H:%M:%S
echo "=== chain8 tail ==="
tail -10 /mnt/workspace/chain8.log 2>/dev/null
echo "=== FINAL per config ==="
for f in P1-base P2-loop P3-loop-wc P4-gate; do
  echo "--- $f ---"
  grep -a "FINAL VAL" /mnt/workspace/output/jaume-plan-$f.log 2>/dev/null || echo "(not finished)"
done
echo "=== procs ==="
pgrep -c -f train_lm_planning