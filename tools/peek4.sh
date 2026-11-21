#!/bin/bash
date +%H:%M:%S
echo "--- running ---"
pgrep -af chain5 | head -2
pgrep -af plan_finetune | head -1
echo "--- ft epochs ---"
grep -a epoch /mnt/workspace/plan-ft-full.log | tail -4
echo "--- chain5 log ---"
tail -3 /mnt/workspace/chain5.log 2>/dev/null || echo "(waiting for finetune)"