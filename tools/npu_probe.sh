#!/usr/bin/env bash
# Report the NPU visible on the current host.
#
# The CANN install location is discovered rather than hardcoded, so the script
# works on any account's home directory. Override with JAUME_CANN_HOME.
set -uo pipefail

CANN_HOME="${JAUME_CANN_HOME:-$(ls -d /opt/home/*/Ascend 2>/dev/null | head -1 || true)}"
if [ -n "$CANN_HOME" ]; then
  source "$CANN_HOME/ascend-toolkit/set_env.sh" >/dev/null 2>&1 || true
  source "$CANN_HOME/nnal/atb/set_env.sh" >/dev/null 2>&1 || true
fi

python -c 'import torch, torch_npu
print("torch", torch.__version__)
print("npu_available", torch.npu.is_available())
if torch.npu.is_available():
    print("device", torch.npu.get_device_name(0))
    free, total = torch.npu.mem_get_info()
    print(f"HBM free {free/2**30:.1f} GiB / total {total/2**30:.1f} GiB")'
