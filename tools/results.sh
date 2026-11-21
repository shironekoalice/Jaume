#!/bin/bash
echo "=== FINAL VAL per config ==="
for f in A-base B-loop C-loop-wc D-loop-wc-align; do
  echo "--- $f ---"
  grep -a "FINAL VAL" /mnt/workspace/output/jaume-loop-$f.log 2>/dev/null || echo MISSING
done
echo
echo "=== PROBE ==="
cat /mnt/workspace/output/loop-probe-350m.log 2>/dev/null | tail -20