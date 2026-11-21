#!/bin/bash
source /opt/home/developer/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1
source /opt/home/developer/Ascend/nnal/atb/set_env.sh >/dev/null 2>&1
SNAP=/home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master
S=/home/developer/.cache/modelscope/models/google--siglip2-base-patch16-224/snapshots/master
cd "$SNAP/code/repo" || exit 1
echo "== waiting for CLM pipeline =="
while pgrep -f "train_causal_lm\|clm_pipeline" > /dev/null; do sleep 120; done
echo "== starting CIFAR finetune at $(date +%H:%M:%S) =="
python -u scripts/train_cifar_siglip.py --model /mnt/workspace/output/causal-350m-v1 \
  --siglip-path "$S" --cifar-root /mnt/workspace/data/cifar-10-batches-py \
  --n-train 10000 --epochs 8 --batch-size 64 --depth 16 --unfreeze-layers 4 \
  --img-size 224 --device npu --out /mnt/workspace/output/cifar-v1 \
  > /mnt/workspace/output/cifar-train.log 2>&1
echo "== CIFAR rc=$? =="
grep -a "\[cf\]" /mnt/workspace/output/cifar-train.log | tail -10
echo CIFAR_DONE
