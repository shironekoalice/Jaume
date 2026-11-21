"""Backend-compatibility helpers for scaled dot-product attention.

Why this exists
---------------
The causal twin needs a *boolean* attention mask (True = attend), because a
float additive mask overflows under bfloat16 autocast when two masks are
added together (finfo.min + finfo.min). The bidirectional encoder uses a
single float mask and is therefore unaffected.

Not every backend accepts a boolean mask in
`torch.nn.functional.scaled_dot_product_attention`. Some Ascend CANN
builds in particular reject it. `sdpa_masked` therefore probes the backend
once and caches the result, falling back to an equivalent float mask.

The two encodings are semantically identical:

    bool mask   : True  -> attend,  False -> blocked
    float mask  : 0     -> attend,  -inf  -> blocked

The float fallback is built with a single `masked_fill_`, never by adding
two masks, so it cannot overflow in bfloat16.
"""

import torch
import torch.nn.functional as F

# Cache of backend capability, keyed by (mask dtype, device type).
_BOOL_MASK_OK = {}


def _explicit_attention(q, k, v, allowed):
    """Reference attention: matmul + mask + softmax + matmul.

    Used instead of SDPA where the fused kernel is unavailable or faulty.
    Same math as scaled_dot_product_attention for a bool mask:

        scores = q @ k^T / sqrt(d)
        scores[~allowed] = -inf
        out = softmax(scores) @ v

    Numerically it differs from the fused kernel only in fp32 accumulation
    order, so it is a drop-in replacement, not an approximation.
    """
    d = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) * (d ** -0.5)
    # single masked_fill_, never an addition of two masks: adding two
    # finfo.min values overflows in bfloat16
    neg = torch.finfo(scores.dtype).min
    scores = scores.masked_fill(~allowed, neg)
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def sdpa_masked(q, k, v, allowed):
    """SDPA with a boolean mask, with an equivalent float-mask fallback.

    `allowed` is a bool tensor broadcastable to [B, H, T, T]; True means the
    query may attend to that key.
    """
    # Ascend's fused FlashAttention (aclnnFlashAttentionScore) faults on the
    # vector core for this mask/head-dim combination. The bool-mask probe in
    # check_npu_env.py passes because it calls SDPA on a tiny 16x16 tensor;
    # the real model uses T=128 with a causal mask and head_dim=64, which
    # takes a different backend path. Rather than try to find a shape that
    # happens to work, skip the fused kernel entirely on NPU.
    #
    # EXCEPT for long sequences: explicit attention materialises a
    # [B, H, T, T] score matrix, which at T=4096 costs 1-2 GiB per layer and
    # OOMs a 61 GiB card at T=8192. For long contexts the fused kernel is the
    # only option, so we try it first and fall back to explicit attention if
    # the backend rejects it (the historical vector-core fault).
    if str(q.device.type) == "npu":
        T = q.shape[-2]
        if T >= 2048:
            key = ("long", T, allowed.dtype)
            cached = _BOOL_MASK_OK.get(key)
            if cached is False:
                return _explicit_attention(q, k, v, allowed)
            try:
                # float additive mask path is the one CANN's fused kernel
                # accepts most reliably; bool masks are rejected on some builds
                neg = torch.finfo(q.dtype).min
                fmask = torch.zeros(allowed.shape, dtype=q.dtype,
                                    device=q.device).masked_fill(~allowed, neg)
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=fmask)
                _BOOL_MASK_OK[key] = True
                return out
            except Exception:
                _BOOL_MASK_OK[key] = False
                return _explicit_attention(q, k, v, allowed)
        return _explicit_attention(q, k, v, allowed)

    key = (allowed.dtype, str(q.device.type))
    ok = _BOOL_MASK_OK.get(key)

    if ok is None:
        try:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
            _BOOL_MASK_OK[key] = True
            return out
        except Exception:
            _BOOL_MASK_OK[key] = False
            ok = False

    if ok:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)

    # float-mask fallback: one masked_fill_, never an addition of two masks
    m = torch.zeros(allowed.shape, dtype=q.dtype, device=q.device)
    m.masked_fill_(~allowed, torch.finfo(q.dtype).min)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=m)


def bool_mask_supported(q, allowed):
    """Probe once whether this backend accepts a bool mask for SDPA."""
    key = (allowed.dtype, str(q.device.type))
    if key not in _BOOL_MASK_OK:
        try:
            F.scaled_dot_product_attention(q, q, q, attn_mask=allowed)
            _BOOL_MASK_OK[key] = True
        except Exception:
            _BOOL_MASK_OK[key] = False
    return _BOOL_MASK_OK[key]
