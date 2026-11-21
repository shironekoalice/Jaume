#!/bin/bash
source /opt/home/developer/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1
source /opt/home/developer/Ascend/nnal/atb/set_env.sh >/dev/null 2>&1
SNAP=/home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master
S=/home/developer/.cache/modelscope/models/google--siglip2-base-patch16-224/snapshots/master
cd "$SNAP/code/repo" || exit 1
while pgrep -f "train_causal_lm" > /dev/null; do sleep 90; done
echo "== CLM done, CIFAR start $(date +%H:%M:%S) =="
python -u scripts/train_cifar_siglip.py --model /mnt/workspace/output/350m-causal-mlm/final \
  --siglip-path "$S" --cifar-root /mnt/workspace/data/cifar-10-batches-py \
  --n-train 10000 --epochs 8 --batch-size 64 --depth 16 --unfreeze-layers 4 \
  --img-size 224 --device npu --out /mnt/workspace/output/cifar-v1 \
  > /mnt/workspace/output/cifar-train.log 2>&1
echo "CIFAR_RC=$?"
grep -a "\[cf\]" /mnt/workspace/output/cifar-train.log | tail -10
while pgrep -f "train_cifar_siglip" > /dev/null; do sleep 30; done
echo "== char OCR start $(date +%H:%M:%S) =="
python -u scripts/train_ocr_char.py --model /mnt/workspace/output/350m-causal-mlm/final \
  --siglip-path "$S" --n-train 9000 --n-val 900 --epochs 12 --batch-size 16 \
  --depth 16 --unfreeze-layers 4 --img-size 224 --device npu \
  --out /mnt/workspace/output/ocr-charv1 \
  > /mnt/workspace/output/ocr-char.log 2>&1
echo "CHAR_RC=$?"
grep -a "FINAL" -A 4 /mnt/workspace/output/ocr-char.log
echo ALL_DONE
