#!/bin/bash
pkill -f train_stream 2>/dev/null; pkill -f chain15.sh 2>/dev/null
sleep 2
echo "stopped (remaining 8L/16L pairs will not change the conclusion)"