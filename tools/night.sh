#!/bin/bash
source /opt/home/developer/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1
source /opt/home/developer/Ascend/nnal/atb/set_env.sh >/dev/null 2>&1
SNAP=/home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master
SIG=/home/developer/.cache/modelscope/models/google--siglip2-base-patch16-224/snapshots/master
TRO=/home/developer/.cache/modelscope/models/microsoft--trocr-base-printed/snapshots/master
cd "$SNAP/code/repo" || exit 1
echo waiting-for-det2
while pgrep -f train_detect > /dev/null; do sleep 120; done
echo starting-S3c
python -u scripts/train_dual_s3.py --model /mnt/workspace/output/350m-causal-mlm/final \
  --siglip-path "$SIG" --trocr-path "$TRO" \
  --funsd-root /mnt/workspace/data/funsd/FUNSD/funsd \
  --s2b-ckpt /mnt/workspace/output/dual-s2b/dual_s2b.pt \
  --n-ocr 1500 --n-vis 1200 --n-funsd 2500 --epochs 15 --batch-size 8 \
  --lr-proj 4e-4 --lr-backbone 4e-5 --depth 16 --unfreeze-layers 4 \
  --device npu --out /mnt/workspace/output/dual-s3c \
  > /mnt/workspace/output/dual-s3c.log 2>&1
echo S3C_RC=$?
grep -a "FINAL S3" -A 16 /mnt/workspace/output/dual-s3c.log
echo ALL_NIGHT_DONE
