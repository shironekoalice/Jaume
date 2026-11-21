"""Checkpoint loading that works across the torch versions this project meets.

Why this exists
---------------
`torch.load(..., weights_only=True)` is not portable across the versions in
use here:

  * torch 2.14 + xpu  : loads these checkpoints fine
  * torch 2.7  + cpu  : raises
        _pickle.UnpicklingError: Unsupported global: GLOBAL getattr was not an
        allowed global by default
    even though the archive itself contains only OrderedDict,
    _rebuild_tensor_v2 and FloatStorage -- the restriction is hit by the
    unpickler's internal machinery, not by project code.

The checkpoints this project writes are its own training outputs (written by
save_pretrained() from an nn.Module state_dict), never third-party
downloads, so the trade-off `weights_only=False` is the accepted one. This
helper keeps that decision in ONE place instead of spread across 15 scripts,
and falls back gracefully when a torch is too old to know the kwarg.
"""

import io
import sys
import types

import torch


def _install_torch_npu_stub():
    """Make an NPU-saved checkpoint loadable on a machine without torch_npu.

    Checkpoints written on an Ascend box pickle two torch_npu symbols:

        torch_npu.utils.storage._rebuild_npu_tensor
        torch_npu.npu._format.Format        (used via getattr(Format, 'ND'))

    The tensors themselves are ordinary CPU storages; the NPU symbols only
    tag them.  When torch_npu is genuinely installed we must NOT shadow it
    (it carries real behaviour), so the stub is only installed when the
    import fails.  With the stub the archive unpickles into plain CPU
    tensors and the weights load unchanged.

    Returns True if a stub was installed, False if the real torch_npu is
    present (or could not be stubbed).
    """
    if "torch_npu" in sys.modules:
        return False
    try:
        import torch_npu  # noqa: F401
        return False
    except Exception:
        pass

    def _rebuild_npu_tensor(storage, storage_offset, size, stride,
                            requires_grad=None, backward_hooks=None,
                            *args, **kwargs):
        # torch._utils._rebuild_tensor changed signature between torch
        # versions (4 args in >=2.14, 6 args before), so rebuild by hand.
        t = torch.empty((0,), dtype=storage.dtype,
                        device=storage._untyped_storage.device)
        t = t.set_(storage._untyped_storage, storage_offset, size, stride)
        if requires_grad:
            t.requires_grad_(True)
        return t

    class Format:
        ND = 2
        FRACTAL_ND = 3
        ND_NZ = 4

    npu = types.ModuleType("torch_npu")
    utils = types.ModuleType("torch_npu.utils")
    storage = types.ModuleType("torch_npu.utils.storage")
    npu_mod = types.ModuleType("torch_npu.npu")
    fmt = types.ModuleType("torch_npu.npu._format")

    storage._rebuild_npu_tensor = _rebuild_npu_tensor
    fmt.Format = Format
    utils.storage = storage
    npu.utils = utils
    npu.npu = npu_mod
    npu_mod._format = fmt

    sys.modules["torch_npu"] = npu
    sys.modules["torch_npu.utils"] = utils
    sys.modules["torch_npu.utils.storage"] = storage
    sys.modules["torch_npu.npu"] = npu_mod
    sys.modules["torch_npu.npu._format"] = fmt
    return True


def load_state_dict(path, map_location="cpu"):
    """Load a state_dict written by this project, across torch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except ModuleNotFoundError as exc:
        if "torch_npu" not in str(exc):
            raise
        if not _install_torch_npu_stub():
            raise
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # torch < 1.13 has no weights_only kwarg
        return torch.load(path, map_location=map_location)
