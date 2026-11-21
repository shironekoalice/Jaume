#!/bin/bash
source /opt/home/developer/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1
source /opt/home/developer/Ascend/nnal/atb/set_env.sh >/dev/null 2>&1
SNAP=/home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master
cd "$SNAP/code/repo" || exit 1
python -u scripts/train_causal_lm.py --model /mnt/workspace/output/350m-causal-mlm/final \
  --data "$SNAP/data/mlm_corpus_v2.jsonl" --output /mnt/workspace/output/causal-350m-v1 \
  --steps 2000 --batch-size 8 --grad-accum 2 --lr 1e-4 --device npu \
  > /mnt/workspace/output/clm-train.log 2>&1
echo "CLM_RC=$?"
python -u scripts/context_probe.py --model /mnt/workspace/output/causal-350m-v1 \
  --lengths 512,1024,2048,4096 --window 128 --device npu \
  > /mnt/workspace/output/ctx-probe.log 2>&1
echo "PROBE_RC=$?"
grep -a "\[ctx\]" /mnt/workspace/output/ctx-probe.log