"""Causal (decoder-style) attention for the Jaume-Causal twin.

Physically separate from `attention.BidirectionalGQA`. This module is imported
only by `backbone_causal.CausalBackbone`, which is selected only when the
loaded config carries `"causal": true`. The two attention forms therefore
cannot be mixed by accident: no code path reaches both.

Difference from the bidirectional implementation, and why it matters for
Jaume-TTC: position i may attend only to positions <= i. In a bidirectional
encoder every token sees the whole sequence, so re-running a layer segment
(REPEAT) is roughly equivalent to adding depth. Under a causal mask the
second pass can only see the prefix, so it necessarily performs new
computation -- the "extra compute" that the test-time-compute literature
refers to.

The mask is built as a boolean mask (True = attend) rather than an additive
float mask: additive masks overflow under bfloat16 autocast (min + min),
whereas a boolean mask composes with `&` and cannot overflow.

RoPE is relative, so it is unaffected by the mask change.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import build_rope, apply_rope
from .nn_compat import sdpa_masked


class CausalGQA(nn.Module):
    """GQA with a causal mask: every token sees only itself and the past."""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def repeat_kv(self, x, n_rep, materialise=True):
        """Expand KV heads to the query-head count.

        materialise=False returns a broadcast *view* instead of a copy. With a
        KV cache the input is the entire history, so copying here would make
        every decode step O(T) in memory traffic and erase the cache's whole
        benefit. SDPA broadcasts the extra head dimension, so a view suffices.
        """
        batch_size, n_kv, seq_len, head_dim = x.shape
        if n_rep == 1:
            return x
        expanded = x[:, :, None, :, :].expand(
            batch_size, n_kv, n_rep, seq_len, head_dim)
        out = expanded.reshape(batch_size, n_kv * n_rep, seq_len, head_dim)
        if not materialise:
            return out          # reshape of an expand() is a view when possible
        return out.contiguous()

    def forward(self, x, attention_mask=None, kv_cache=None,
                position_offset=None):
        """Causal GQA forward, optionally with incremental KV caching.

        kv_cache: a KVCache for THIS layer. When provided:
          * k/v for the new tokens are appended to the cache,
          * RoPE uses absolute positions (past_len + i), which is why the
            cache stores already-rotated keys,
          * attention runs against the whole cached history, with a single
            causal row per new query.
        position_offset: absolute position of x[:, 0]. Defaults to the
          cache length when a cache is supplied, else 0.
        """
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # RoPE at ABSOLUTE positions. For incremental decoding only the new
        # tokens need rotation; cached keys were rotated when written.
        if position_offset is None:
            position_offset = kv_cache.length if kv_cache is not None else 0
        cos, sin = build_rope(self.head_dim, position_offset + seq_len,
                              1000000.0, x.device, x.dtype)
        cos = cos[position_offset:position_offset + seq_len]
        sin = sin[position_offset:position_offset + seq_len]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if kv_cache is not None:
            # store UNexpanded (n_kv heads) and only expand for the matmul
            k_all, v_all = kv_cache.append(k, v)
            kv_len = k_all.shape[2]
            past_len = position_offset
        else:
            k_all, v_all = k, v
            kv_len = seq_len
            past_len = 0

        n_rep = self.num_heads // self.num_key_value_heads
        k_all = self.repeat_kv(k_all, n_rep, materialise=(kv_cache is None))
        v_all = self.repeat_kv(v_all, n_rep, materialise=(kv_cache is None))

        # allowed[b, 0, i, j]: query i (absolute pos past_len+i) sees key j
        # for all j <= past_len+i.
        qi = torch.arange(past_len, past_len + seq_len, device=x.device)
        kj = torch.arange(kv_len, device=x.device)
        allowed = (kj[None, :] <= qi[:, None])[None, None, :, :]
        if attention_mask is not None:
            keep = attention_mask.to(torch.bool)[:, None, None, :]
            if keep.shape[-1] == kv_len:
                allowed = allowed & keep
            else:
                # mask covers only the new tokens; history is assumed valid
                allowed = allowed & torch.cat([
                    torch.ones(batch_size, 1, 1, kv_len - keep.shape[-1],
                               dtype=torch.bool, device=x.device), keep], dim=-1)

        # Always let a position attend to itself (see module docstring).
        idx = torch.arange(seq_len, device=x.device)
        allowed = allowed.clone()
        allowed[:, :, idx, past_len + idx] = True

        output = sdpa_masked(q, k_all, v_all, allowed)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(output)
