#!/usr/bin/env bash
set -o pipefail
source /opt/home/developer/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1 || true
source /opt/home/developer/Ascend/nnal/atb/set_env.sh >/dev/null 2>&1 || true
ENV=/mnt/workspace/venvs/jaume-omni
"$ENV/bin/python" -m pip install --disable-pip-version-check 'funasr>=1.3.26' >> /mnt/workspace/log/omni/funasr-install.log 2>&1
echo "funasr install rc=$?"
"$ENV/bin/python" -c "import funasr; print('funasr', funasr.__version__)" 2>&1 | tail -5