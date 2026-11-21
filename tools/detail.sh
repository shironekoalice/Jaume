#!/bin/bash
echo "=== P2 tail ==="
tail -5 /mnt/workspace/output/jaume-plan-P2-loop.log
echo "=== P3 tail ==="
tail -5 /mnt/workspace/output/jaume-plan-P3-loop-wc.log
echo "=== P4 tail ==="
tail -6 /mnt/workspace/output/jaume-plan-P4-gate.log
echo "=== probe ==="
python3 - <<'PY'
import json
d = json.load(open('/mnt/workspace/output/loop-probe-planning.json'))
for k, v in sorted(d['runs'].items()):
    print(k, 'dist=', [round(x,4) for x in v['round_distance']])
    print('   ovl_consec=', [round(x,3) for x in v['subspace_overlap_consecutive']])
    print('   ovl_first =', [round(x,3) for x in v['subspace_overlap_vs_first']])
PY