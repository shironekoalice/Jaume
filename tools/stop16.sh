#!/bin/bash
pkill -f train_vlm_probe 2>/dev/null; pkill -f chain16.sh 2>/dev/null; sleep 2
echo stopped