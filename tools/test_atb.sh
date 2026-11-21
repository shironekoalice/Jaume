#!/usr/bin/env bash
echo start
source /opt/home/developer/Ascend/nnal/atb/set_env.sh
echo after-source2
env | grep -i -E 'atb|ascend' | head -10