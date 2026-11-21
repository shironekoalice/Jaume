#!/usr/bin/env bash
set -o pipefail
source /opt/home/developer/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1 || true
source /opt/home/developer/Ascend/nnal/atb/set_env.sh >/dev/null 2>&1 || true
mkdir -p /mnt/workspace/models /mnt/workspace/log/omni
if [ ! -d /mnt/workspace/models/SenseVoiceSmall ]; then
  cd /mnt/workspace/models
  python -c "from modelscope.hub.snapshot_download import snapshot_download; p=snapshot_download('iic/SenseVoiceSmall', local_dir='/mnt/workspace/models/SenseVoiceSmall'); print('saved', p)" > /mnt/workspace/log/omni/sensevoice-dl.log 2>&1
  echo "sensevoice download rc=$?"
fi
ls -la /mnt/workspace/models/SenseVoiceSmall 2>/dev/null | head -15
du -sh /mnt/workspace/models/SenseVoiceSmall 2>/dev/null