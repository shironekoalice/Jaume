#!/bin/bash
date +%H:%M:%S
echo "=== chain11 ==="
grep -aE "^===|FINAL VAL" /mnt/workspace/chain11.log 2>/dev/null | tail -8
echo "=== current log ==="
for f in derive-D1-base derive-D2-loop-wc derive-D3-loop derive-D4-gate; do
  if [ -f /mnt/workspace/output/$f.log ]; then
    echo "--- $f ---"
    tail -3 /mnt/workspace/output/$f.log
  fi
done
echo "=== procs ==="
pgrep -c -f "train_lm_derive|probe_loop"