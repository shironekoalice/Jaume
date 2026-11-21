"""Incremental KV cache for Jaume-Causal.

Design choices (what "best current practice" means here)
-------------------------------------------------------
1. **Cache the KV *before* GQA expansion.** The model is GQA (8 query heads,
   4 KV heads), so storing expanded K/V would waste exactly 2x memory. We
   store [B, n_kv, T, head_dim] and expand only at attention time.

2. **RoPE is applied at write time, not read time.** Each new token's K is
   rotated with its *absolute* position (past_len + i) before being stored.
   Reading then needs no re-rotation, which is what makes the cache O(1)
   per step instead of O(T).

3. **Static pre-allocation with a write cursor.** Reallocating a growing
   tensor every decode step is O(T) per step and fragments the NPU allocator.
   We allocate [B, n_kv, max_len, head_dim] once and write in place.

4. **Append-only, no eviction.** Sliding-window / eviction policies are a
   separate concern (and interact with the depth-saturation findings); this
   layer only guarantees correctness and speed for sequences up to max_len.

5. **Explicit opt-in.**  reproduces the previous behaviour
   bit-for-bit, so every existing experiment stays comparable.

The mask in incremental mode is a single row: query at absolute position
 may attend to keys [0, past_len + i].
"""

import torch


class KVCache:
    """Pre-allocated KV cache for one attention layer."""

    def __init__(self, batch_size, num_kv_heads, head_dim, max_len,
                 device, dtype):
        self.max_len = max_len
        self.seen = 0
        shape = (batch_size, num_kv_heads, max_len, head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)

    @property
    def length(self):
        return self.seen

    def append(self, k_new, v_new):
        """k_new/v_new: [B, n_kv, T_new, head_dim] already RoPE-rotated.

        Returns views into the pre-allocated buffer (no copy) so the caller
        can attend over the whole history without materialising a new tensor
        each step.
        """
        t_new = k_new.shape[2]
        end = self.seen + t_new
        if end > self.max_len:
            raise RuntimeError(
                "KV cache overflow: %d + %d > %d" % (self.seen, t_new,
                                                     self.max_len))
        self.k[:, :, self.seen:end] = k_new
        self.v[:, :, self.seen:end] = v_new
        self.seen = end
        return self.k[:, :, :end], self.v[:, :, :end]

    def reset(self):
        self.seen = 0

    def memory_bytes(self):
        return self.k.numel() * self.k.element_size() * 2


def make_caches(config, batch_size, max_len, device, dtype):
    """One cache per layer, matching the backbone's layer count."""
    head_dim = config.hidden_size // config.num_attention_heads
    return [KVCache(batch_size, config.num_key_value_heads, head_dim,
                    max_len, device, dtype)
            for _ in range(config.num_layers)]


def cache_report(caches):
    total = sum(c.memory_bytes() for c in caches)
    return {
        "layers": len(caches),
        "seen": caches[0].length if caches else 0,
        "total_mib": total / (1024 ** 2),
    }
