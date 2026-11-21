import io
lines = [
"#!/bin/bash",
"SNAP=/home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master",
"B=/mnt/workspace/output/350m-causal-mlm/final",
"CFG=/code/repo/configs/jaume-350m-causal.json",
"OUT=/mnt/workspace/output",
'cd "/code/repo" || exit 1',
"run () {",
"  local name=; shift",
'  echo "=== [] 20:11:56 ==="',
"  python -u scripts/train_carry.py --config  --init-from  \\",
"    --n-digits 8 --n-train 20000 --n-val 300 --epochs 8 \\",
"    --batch-size 32 --grad-accum 1 --lr 3e-4 --max-length 128 \\",
"    --eval-every 200 --log-every 100 \\",
'    --output /carry8- --device npu "" \',
'    2>&1 | tee /carry8-.log | grep -aE "\[carry\] (VAL_ACC|FINAL_ACC)"',
'  echo "=== [] rc= 20:11:56 ==="',
"}",
"run L8 --depth 8",
"run L16 --depth 16",
'echo "=== ALL DONE 20:11:56 ==="',
]
io.open("tools/chain23.sh", "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
print("ok")
