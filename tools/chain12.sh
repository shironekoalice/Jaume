#!/bin/bash
# Is 16 layers already saturated on the two-stage task?
# If 32 layers does not beat 16, the base is at capacity for this task and
# extra depth (looped or stacked) has nothing to add -- which is the claim the
# loop experiments imply but have not tested directly.
SNAP=/home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master
B=/mnt/workspace/output/350m-causal-mlm/final
OUT=/mnt/workspace/output
cd "$SNAP/code/repo"

common="--init-from $B --n-items 6 --n-slots 4 --n-fixed 3 --n-same 2 --n-notsame 3
        --epochs 1 --batch-size 24 --grad-accum 2 --max-length 160
        --lr 3e-4 --warmup-steps 100 --val-docs 256
        --eval-every 300 --log-every 150 --device npu"

cfg_for () {
  # write a config with num_layers=$1, copying the loop config
  python3 - "$1" <<'PY'
import json, sys
n = int(sys.argv[1])
c = json.load(open("configs/jaume-350m-loop.json"))
c["num_layers"] = n
c["num_loops"] = 1
c["loop_form"] = "full"
json.dump(c, open(f"configs/_depth{n}.json", "w"), indent=2)
print(f"configs/_depth{n}.json")
PY
}

# 16L reference is D1 (already done): 0.1812
# 8L  -- is 16 needed at all?
cfg_for 8
python -u scripts/train_lm_derive.py --config configs/_depth8.json $common \
  --n-tasks 40000 --output $OUT/derive-L8 2>&1 | tee $OUT/derive-L8.log
grep -a "FINAL VAL" $OUT/derive-L8.log

# 32L -- does more depth help?  (warm start only fills matching keys, and
# interleaving new layers would bias the result, so 32L trains from the same
# 16L init by repeating the stack -- documented as a limitation)
cfg_for 32
python -u scripts/train_lm_derive.py --config configs/_depth32.json $common \
  --n-tasks 40000 --output $OUT/derive-L32 2>&1 | tee $OUT/derive-L32.log
grep -a "FINAL VAL" $OUT/derive-L32.log

echo "=== DEPTH SWEEP DONE $(date +%H:%M:%S) ==="