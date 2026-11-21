#!/bin/bash
date +%H:%M:%S
grep -aE "^===|FINAL VAL" /mnt/workspace/chain15.log 2>/dev/null | tail -14
echo "procs=$(pgrep -c -f train_stream)"