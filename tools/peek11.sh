#!/bin/bash
date +%H:%M:%S
echo "=== chain9 tail ==="
tail -12 /mnt/workspace/chain9.log 2>/dev/null
echo "=== FINAL per config ==="
for f in P1-base P2-loop P3-loop-wc P4-gate; do
  printf "%-16s " "$f"
  grep -a "FINAL VAL" /mnt/workspace/output/jaume-plan-$f.log 2>/dev/null || echo "(not finished)"
done
echo "=== gate values ==="
grep -a "gate\[" /mnt/workspace/output/jaume-plan-P4-gate.log 2>/dev/null | tail -3
echo "=== procs ==="
pgrep -c -f train_lm_planning