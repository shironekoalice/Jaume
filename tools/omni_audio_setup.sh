#!/usr/bin/env bash
set -o pipefail
source /opt/home/developer/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1 || true
source /opt/home/developer/Ascend/nnal/atb/set_env.sh >/dev/null 2>&1 || true
echo '[omni-setup] stage env loaded'
ENV_DIR=/mnt/workspace/venvs/jaume-omni
mkdir -p /mnt/workspace/venvs /mnt/workspace/log/omni
if [ ! -f "$ENV_DIR/bin/python" ]; then
    python -m venv --system-site-packages "$ENV_DIR"
fi
echo '[omni-setup] venv ready'
"$ENV_DIR/bin/python" -m pip install --disable-pip-version-check --no-deps 'torchaudio==2.7.1' --index-url https://download.pytorch.org/whl/cpu
"$ENV_DIR/bin/python" -m pip install --disable-pip-version-check soundfile
echo '[omni-setup] running probe'
"$ENV_DIR/bin/python" /tmp/omni_env_probe.py