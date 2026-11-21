#!/bin/bash
date +%H:%M:%S
echo "=== chain10 ==="
tail -20 /mnt/workspace/chain10.log 2>/dev/null
echo
echo "=== running procs ==="
pgrep -af "train_lm_planning|probe_loop|chain10" | head -6
echo
echo "=== FINAL vals ==="
for f in P1-base P2-loop P3-loop-wc P4-gate; do
  printf "%-14s " "$f"
  grep -a "FINAL VAL" /mnt/workspace/output/jaume-plan-$f.log 2>/dev/null || echo "(none)"
done