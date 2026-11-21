#!/bin/bash
S1=$(grep -a "step " /tmp/perf2.log | tail -1 | grep -oP 'step \K\d+')
T1=$(date +%s)
sleep 80
S2=$(grep -a "step " /tmp/perf2.log | tail -1 | grep -oP 'step \K\d+')
T2=$(date +%s)
DT=$((T2-T1)); DS=$((S2-S1))
echo "delta_steps=$DS delta_sec=$DT"
python - <<PY
ds=$DS; dt=$DT
rate=ds/dt if dt else 0
print(f"rate={rate:.3f} step/s")
print(f"7492 steps -> {7492/rate/60:.0f} min")
PY
tail -2 /tmp/perf2.log
pkill -f train_lm_planning
echo killed