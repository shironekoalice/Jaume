#!/bin/bash
# wait for the running BCE study to finish, then start the CE study
while pgrep -f plan_head_study > /dev/null; do sleep 15; done
echo "head_study finished at $(date +%H:%M:%S)"
bash /tmp/experiments.sh ce-split