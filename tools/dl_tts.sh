#!/usr/bin/env bash
set -o pipefail
source /opt/home/developer/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1 || true
source /opt/home/developer/Ascend/nnal/atb/set_env.sh >/dev/null 2>&1 || true
mkdir -p /mnt/workspace/data/tts
cd /mnt/workspace/data/tts
rm -f train_clean_100.tar.gz data_aishell3.tgz
for url in \
  "https://www.openslr.org/resources/141/train_clean_100.tar.gz" \
  "https://www.openslr.org/resources/93/data_aishell3.tgz" ; do
  f=$(basename "$url")
  echo "downloading $f" >> /mnt/workspace/log/omni/tts-dl2.log
  curl -L --retry 3 --connect-timeout 30 -o "$f" "$url" >> /mnt/workspace/log/omni/tts-dl2.log 2>&1
  echo "done $f rc=$?" >> /mnt/workspace/log/omni/tts-dl2.log
done
ls -lh /mnt/workspace/data/tts