#!/bin/bash
S1=$(grep -a "step " /tmp/perf.log | tail -1 | grep -oP 'step \K\d+')
T1=$(date +%s)
sleep 90
S2=$(grep -a "step " /tmp/perf.log | tail -1 | grep -oP 'step \K\d+')
T2=$(date +%s)
DT=$((T2-T1)); DS=$((S2-S1))
echo "steps: $S1 -> $S2 in ${DT}s (delta $DS)"
python - <<PY
ds=$DS; dt=$DT
rate=ds/dt if dt else 0
print(f"rate={rate:.3f} step/s")
print(f"7492 steps -> {7492/rate/60:.1f} min")
PY
echo "=== AICore (3 samples) ==="
for i in 1 2 3; do npu-smi info 2>&1 | sed -n '8p'; sleep 3; done
echo "=== log ==="
tail -3 /tmp/perf.log
pkill -f train_lm_planning