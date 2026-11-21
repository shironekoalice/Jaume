#!/bin/bash
date +%H:%M:%S
echo "--- ft log ---"
tail -12 /mnt/workspace/plan-ft-full.log 2>/dev/null || echo "no ft log"
echo "--- running ---"
pgrep -af plan_finetune | head -2
echo "--- npu ---"
npu-smi info 2>&1 | sed -n '5,9p'