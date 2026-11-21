#!/bin/bash
# Stage 1 of the loop programme: three matched configs.
#   A  base      : num_loops=1  (no loop; the reference)
#   B  loop      : num_loops=4  (the loop alone)
#   C  loopalign : num_loops=4  + LoopMTP-style latent soft alignment
# A separates "the loop helps" from "the alignment supervision helps".
SNAP=/home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master
CKPT=/mnt/workspace/output/350m-causal-mlm/final
DATA=$SNAP/data/mlm_corpus.jsonl
OUT=/mnt/workspace/output
cd "$SNAP/code/repo"

while pgrep -f plan_finetune > /dev/null; do sleep 20; done
echo "=== fine-tune finished, starting loop stage 1 at $(date +%H:%M:%S) ==="

common="--config configs/jaume-350m-loop.json --init-from $CKPT --data $DATA
        --epochs 1 --batch-size 16 --grad-accum 4 --max-length 256
        --lr 3e-4 --warmup-steps 200 --val-blocks 64 --eval-every 200
        --eval-batches 8 --log-every 100 --device npu"

echo "=== [A] base: num_loops=1 ==="
python -u scripts/train_lm.py $common --num-loops 1 \
  --output $OUT/jaume-loop-350m-base \
  2>&1 | tee $OUT/jaume-loop-350m-base.log
echo "=== [A] done $(date +%H:%M:%S) ==="

echo "=== [B] loop: num_loops=4, no alignment ==="
python -u scripts/train_lm.py $common --num-loops 4 \
  --output $OUT/jaume-loop-350m-loop4 \
  2>&1 | tee $OUT/jaume-loop-350m-loop4.log
echo "=== [B] done $(date +%H:%M:%S) ==="

echo "=== [C] loopalign: num_loops=4 + latent soft alignment ==="
python -u scripts/train_lm.py $common --num-loops 4 --loop-align-weight 0.1 \
  --output $OUT/jaume-loop-350m-loop4align \
  2>&1 | tee $OUT/jaume-loop-350m-loop4align.log
echo "=== [C] done $(date +%H:%M:%S) ==="

echo "=== [D] probe the interior of the looped model ==="
python -u scripts/probe_loop.py --model $OUT/jaume-loop-350m-loop4align \
  --num-loops 4 --sweep-loops 1 2 4 8 --data $DATA \
  --num-samples 32 --batch-size 4 --max-length 256 --device npu \
  --out $OUT/loop-probe-350m.json \
  2>&1 | tee $OUT/loop-probe-350m.log
echo "=== ALL DONE $(date +%H:%M:%S) ==="