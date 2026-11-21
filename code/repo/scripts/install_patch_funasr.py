#!/usr/bin/env python3
"""Make funasr/register.py tolerate source-inspection failures.

funasr 1.4.24 recursively inspects every registered class. On the Ascend image
this can hang inside inspect.getsourcelines. The metadata is only file:line for
debugging, so falling back to module name and line 0 is safe.
"""
import sys
from pathlib import Path

env = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/workspace/venvs/jaume-omni")
p = env / "lib/python3.11/site-packages/funasr/register.py"
if not p.exists():
    raise SystemExit("not found: " + str(p))
s = p.read_text(encoding="utf-8")
old = "            class_file = inspect.getfile(target_class)\n            class_line = inspect.getsourcelines(target_class)[1]\n"
new = (
    "            try:\n"
    "                class_file = inspect.getfile(target_class)\n"
    "                class_line = inspect.getsourcelines(target_class)[1]\n"
    "            except Exception:\n"
    "                class_file = getattr(target_class, \"__module__\", \"unknown\")\n"
    "                class_line = 0\n"
)
if old in s:
    p.write_text(s.replace(old, new, 1), encoding="utf-8")
    print("[funasr-patch] patched " + str(p))
elif "class_line = 0" in s:
    print("[funasr-patch] already patched " + str(p))
else:
    raise SystemExit("anchor not found; refusing to guess")
