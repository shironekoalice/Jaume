#!/usr/bin/env bash
# Patch funasr's register.py so a failing source inspection cannot hang import.
# This only affects metadata (file:line shown in registries); registrations and
# model loading are unchanged.
set -euo pipefail
ENV_DIR="/mnt/workspace/venvs/jaume-omni"
python - "" <<'PY'
import sys
from pathlib import Path
env = Path(sys.argv[1])
p = env / "lib/python3.11/site-packages/funasr/register.py"
if not p.exists():
    raise SystemExit(f"not found: {p}")
s = p.read_text(encoding="utf-8")
old = '''            class_file = inspect.getfile(target_class)
            class_line = inspect.getsourcelines(target_class)[1]
'''
new = '''            try:
                class_file = inspect.getfile(target_class)
                class_line = inspect.getsourcelines(target_class)[1]
            except Exception:
                class_file = getattr(target_class, "__module__", "unknown")
                class_line = 0
'''
if old in s:
    p.write_text(s.replace(old, new, 1), encoding="utf-8")
    print(f"[funasr-patch] patched {p}")
elif "class_line = 0" in s:
    print(f"[funasr-patch] already patched {p}")
else:
    raise SystemExit("anchor not found; refusing to guess")
PY
