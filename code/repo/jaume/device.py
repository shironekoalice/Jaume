"""Device resolution and backend-agnostic helpers shared by all scripts.

Priority: cuda > npu > xpu > mps > cpu.

  * 'cuda' -- NVIDIA (rented boxes).
  * 'npu'  -- Ascend, via the `torch_npu` package (CANN). Requires
              `import torch_npu` before `torch.npu` is usable; this module
              imports it lazily and tolerates its absence.
  * 'xpu'  -- Intel GPUs through the official torch XPU builds (torch 2.6+
              ships Windows client-GPU support).
  * 'mps'  -- Apple Silicon.

The same commands therefore run unchanged on NVIDIA, Ascend and Intel boxes.
Scripts pick a device with `resolve_device()` and use the helpers below
instead of calling torch.cuda.* / torch.npu.* directly.
"""
import contextlib

import torch

try:  # Ascend CANN backend. Absent on non-Ascend hosts.
    import torch_npu  # noqa: F401
    _HAS_NPU = True
except Exception:  # pragma: no cover - import guard
    _HAS_NPU = False

ACCEL_DEVICES = ("cuda", "npu", "xpu")


def resolve_device(preference=None):
    """Return a torch device string: 'cuda' | 'xpu' | 'mps' | 'cpu'.

    An explicit preference is honoured verbatim (so --device cpu keeps
    working for debugging); 'auto'/None falls through the priority list.
    """
    if preference not in (None, "", "auto"):
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if _HAS_NPU and hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def has_amp(device):
    """bf16 autocast is active on GPU backends only; cpu keeps fp32."""
    return device in ACCEL_DEVICES


def accelerator(device=None):
    """The active accelerator string, or None when running on cpu."""
    device = device or resolve_device()
    return device if device in ACCEL_DEVICES else None


def seed_all(device, seed):
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif device == "npu":
        torch.npu.manual_seed_all(seed)
    elif device == "xpu":
        torch.xpu.manual_seed_all(seed)


def make_scaler(device, enabled=True):
    return torch.amp.GradScaler(device, enabled=enabled and has_amp(device))


def autocast_ctx(device, enabled=True):
    """bf16 autocast on cuda/xpu; a no-op context on cpu."""
    if has_amp(device) and enabled:
        return torch.autocast(device, dtype=torch.bfloat16)
    return contextlib.nullcontext()


def empty_cache(device=None):
    device = device or accelerator()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "npu":
        torch.npu.empty_cache()
    elif device == "xpu":
        torch.xpu.empty_cache()


def max_memory_allocated(device=None):
    device = device or accelerator()
    if device == "cuda":
        return torch.cuda.max_memory_allocated()
    if device == "npu":
        return torch.npu.max_memory_allocated()
    if device == "xpu":
        return torch.xpu.max_memory_allocated()
    return 0


def reset_peak_memory_stats(device=None):
    device = device or accelerator()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    elif device == "npu":
        torch.npu.reset_peak_memory_stats()
    elif device == "xpu":
        torch.xpu.reset_peak_memory_stats()


def total_memory(device=None):
    device = device or accelerator()
    if device == "cuda":
        return torch.cuda.get_device_properties(0).total_memory
    if device == "npu":
        return torch.npu.get_device_properties(0).total_memory
    if device == "xpu":
        return torch.xpu.get_device_properties(0).total_memory
    return 0


def synchronize(device=None):
    device = device or accelerator()
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "npu":
        torch.npu.synchronize()
    elif device == "xpu":
        torch.xpu.synchronize()


def describe(device=None):
    device = device or resolve_device()
    if device == "cuda":
        p = torch.cuda.get_device_properties(0)
        return f"{p.name} ({p.total_memory / 1024 ** 3:.1f} GiB)"
    if device == "npu":
        p = torch.npu.get_device_properties(0)
        return f"{p.name} ({p.total_memory / 1024 ** 3:.1f} GiB)"
    if device == "xpu":
        p = torch.xpu.get_device_properties(0)
        return f"{p.name} ({p.total_memory / 1024 ** 3:.1f} GiB)"
    return "cpu"
