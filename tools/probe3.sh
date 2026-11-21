#!/bin/bash
echo "=== probe log ==="
grep -aE "T=[0-9]:" /mnt/workspace/output/loop-probe-derive.log
echo
echo "=== side by side ==="
python3 - <<'PY'
import json
for tag, path in [("generic text", "/mnt/workspace/output/loop-probe-350m.json"),
                  ("planning single-step", "/mnt/workspace/output/loop-probe-planning.json"),
                  ("derive two-stage", "/mnt/workspace/output/loop-probe-derive.json")]:
    try:
        d = json.load(open(path))
    except Exception as e:
        print(tag, "MISSING"); continue
    v = d["runs"].get("T=8")
    if not v:
        print(tag, "no T=8"); continue
    dist = [round(x,4) for x in v["round_distance"]]
    print(tag, dist)
PY