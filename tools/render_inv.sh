#!/bin/bash
cd /home/developer/.cache/modelscope/models/LightNeko--Jaume/snapshots/master/code/repo || exit 1
python - << PYEOF
import sys
sys.path.insert(0, ".")
from docvqa_env import make_invoice
img, gt = make_invoice(12345)
img.save("/mnt/workspace/output/invoice_sample.png")
print("saved", img.size)
for k, v in gt.items():
    print(" ", k, "=", v)
PYEOF
