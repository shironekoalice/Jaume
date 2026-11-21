"""Delete specific files from the ModelScope repo (targeted, never blind).

Usage:
    python ms_delete.py LightNeko/Jaume output/mem/B1-cls-s1/final/pytorch_model.bin
"""
import sys
import os

from modelscope.hub.api import HubApi

token = open(os.path.expanduser("~/.modelscope_token"), encoding="utf-8").read().strip()
api = HubApi()
api.login(token)

repo, *paths = sys.argv[1:]
for p in paths:
    try:
        api.delete_files(repo, delete_patterns=p)
        print(f"deleted {p}")
    except Exception as exc:  # noqa: BLE001 - report and continue
        print(f"FAILED {p}: {type(exc).__name__}: {exc}")
