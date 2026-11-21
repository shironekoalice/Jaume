#!/bin/bash
date +%H:%M:%S
grep -aE "^===|FINAL VAL" /mnt/workspace/chain13.log 2>/dev/null | tail -12
echo "procs=$(pgrep -c -f train_chain)"