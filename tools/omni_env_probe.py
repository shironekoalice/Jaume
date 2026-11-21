"""Read-only audio environment probe. Run on the remote NPU host."""
import importlib
import importlib.metadata
import json
import platform
import sys

report = {"python": sys.version, "platform": platform.platform(), "modules": {}}
for name in ("torch", "torch_npu", "torchaudio", "transformers", "funasr", "soundfile", "numpy", "peft", "accelerate"):
    try:
        module = importlib.import_module(name)
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = str(getattr(module, "__version__", "unknown"))
        report["modules"][name] = {"ok": True, "version": version}
    except Exception as exc:
        report["modules"][name] = {"ok": False, "error": repr(exc)}

torch = sys.modules.get("torch")
if torch is not None:
    npu = getattr(torch, "npu", None)
    try:
        report["npu_available"] = bool(npu and npu.is_available())
        if report["npu_available"]:
            report["device_count"] = npu.device_count()
            report["device_name"] = npu.get_device_name(0)
    except Exception as exc:
        report["npu_error"] = repr(exc)
print(json.dumps(report, ensure_ascii=False, indent=2))
