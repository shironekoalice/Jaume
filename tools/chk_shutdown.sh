#!/bin/bash
whoami
sudo -n true 2>&1 && echo "sudo OK" || echo "NO passwordless sudo"
ps -p 1 -o pid,comm
for c in shutdown poweroff halt systemctl; do which $c 2>/dev/null || echo "missing $c"; done
ls -l /proc/sysrq-trigger 2>/dev/null || echo "no sysrq"