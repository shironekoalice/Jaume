import sys, importlib, faulthandler, torch
sys.path.insert(0, "/mnt/workspace/venvs/jaume-omni/lib/python3.11/site-packages")
faulthandler.dump_traceback_later(20, exit=True)
print("torch", torch.__version__)
# 1. load funasr.register alone WITHOUT triggering funasr/__init__ auto-scan
from funasr.register import tables, RegisterTables
print("register loaded")
# 2. neutralize the source-inspection in register
def fast_register(self, register_tables_key, key=None):
    def decorator(target_class):
        registry_key = key if key is not None else target_class.__name__
        registry = getattr(self, register_tables_key)
        registry[registry_key] = target_class
        return target_class
    return decorator
tables.register = fast_register.__get__(tables, RegisterTables)
print("register patched")
# 3. now full funasr import scans submodules but decorator is no-op
import funasr
print("funasr imported", funasr.__version__)
# 4. load SenseVoice without AutoModel submodule scan
from funasr.models.sense_voice.model import SenseVoiceEncoderSmall, SenseVoiceSmall
print("classes ok", SenseVoiceEncoderSmall, SenseVoiceSmall)