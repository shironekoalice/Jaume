#!/bin/bash
date +%H:%M:%S
grep -aE "^===|FINAL VAL|depth=" /mnt/workspace/chain12.log 2>/dev/null | tail -10
for f in derive-L8 derive-L32; do
  if [ -f /mnt/workspace/output/$f.log ]; then echo "--- $f ---"; tail -2 /mnt/workspace/output/$f.log; fi
done
pgrep -c -f train_lm_derive