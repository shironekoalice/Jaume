#!/bin/bash
date +%H:%M:%S
tail -14 /mnt/workspace/plan-head-study.log
echo "proc=$(pgrep -c -f plan_head_study)"