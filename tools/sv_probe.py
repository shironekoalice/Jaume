import sys, torch, json, os
# Avoid funasr/__init__.py auto-submodule scanning; load the model module directly.
sys.path.insert(0, "/mnt/workspace/venvs/jaume-omni/lib/python3.11/site-packages")
print("torch", torch.__version__)
import importlib
# touch needed heavy deps first
for m in ("torch", "torchaudio", "transformers"):
    try:
        importlib.import_module(m)
    except Exception as e:
        print("dep err", m, repr(e)[:120])
# load funasr submodules lazily
import funasr.models.sense_voice.model as svm
print("sense_voice model module ok")