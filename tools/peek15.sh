#!/bin/bash
date +%H:%M:%S
grep -aE "^===|FINAL VAL" /mnt/workspace/chain14.log 2>/dev/null | tail -14
echo "procs=$(pgrep -c -f train_register)"