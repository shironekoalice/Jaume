#!/bin/bash
source /opt/home/developer/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1
source /opt/home/developer/Ascend/nnal/atb/set_env.sh >/dev/null 2>&1
SNAP=/home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master
S=/home/developer/.cache/modelscope/models/google--siglip2-base-patch16-224/snapshots/master
cd "$SNAP/code/repo" || exit 1
echo "== waiting for CIFAR stage =="
while pgrep -f "cifar_after_clm\|train_cifar_siglip\|clm_pipeline\|train_causal_lm" > /dev/null; do sleep 120; done
echo "== starting char-head OCR at $(date +%H:%M:%S) =="
python -u scripts/train_ocr_char.py --model /mnt/workspace/output/350m-causal-mlm/final \
  --siglip-path "$S" --n-train 9000 --n-val 900 --epochs 12 --batch-size 16 \
  --depth 16 --unfreeze-layers 4 --img-size 224 --device npu \
  --out /mnt/workspace/output/ocr-charv1 \
  > /mnt/workspace/output/ocr-char.log 2>&1
echo "== char OCR rc=$? =="
grep -a "FINAL" -A 4 /mnt/workspace/output/ocr-char.log
echo CHAR_DONE
