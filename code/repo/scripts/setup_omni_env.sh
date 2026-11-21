#!/usr/bin/env bash
# Reproducible Ascend environment for Jaume Omni.
#
# This image has torch 2.7.1 with torchaudio 2.11 (ABI mismatch). The isolated
# venv pins torchaudio 2.7.1. libgomp must be preloaded on aarch64.
set -euo pipefail

source /opt/home/developer/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1 || true

ENV=/mnt/workspace/venvs/jaume-omni
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ENV/bin/python"

if [ ! -x "$PY" ]; then
  python -m venv --system-site-packages "$ENV"
fi

"$PY" -m pip install -q --disable-pip-version-check \
  "torchaudio==2.7.1" \
  --index-url https://download.pytorch.org/whl/cpu

"$PY" -m pip install -q --disable-pip-version-check \
  soundfile funasr modelscope transformers tensorboard

"$PY" "$ROOT/scripts/install_patch_funasr.py" "$ENV"
echo "[omni-env] done"