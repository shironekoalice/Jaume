#!/bin/bash
pkill -f train_lm_planning 2>/dev/null
pkill -f chain9.sh 2>/dev/null
sleep 3
echo killed
pgrep -c -f train_lm_planning || echo 0