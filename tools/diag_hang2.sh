#!/bin/bash
echo "=== stack of the hung process ==="
timeout 30 cat /proc/104710/stack 2>/dev/null | head -20 || echo "no kernel stack"
echo "=== wchan ==="
cat /proc/104710/wchan 2>/dev/null; echo
echo "=== threads in D state ==="
ps -L -o pid,tid,stat,wchan:30,pcpu -p 104710 2>/dev/null | head -10
echo "=== last log lines ==="
tail -2 /mnt/workspace/output/jaume-plan-P2-loop.log