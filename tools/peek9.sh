#!/bin/bash
echo "=== chain8 full ==="
cat /mnt/workspace/chain8.log 2>/dev/null | tail -40
echo
echo "=== P2 log ==="
tail -25 /mnt/workspace/output/jaume-plan-P2-loop.log 2>/dev/null