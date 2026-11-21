#!/bin/bash
echo "=== perf log ==="
tail -12 /tmp/perf.log
echo "=== full npu-smi ==="
npu-smi info 2>&1 | sed -n '5,12p'
echo "=== procs ==="
pgrep -c -f train_lm_planning