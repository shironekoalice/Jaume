"""Autoregressive generation for Jaume-Causal with incremental KV caching.

Without a cache, producing N tokens costs sum_{i=1..N} O((P+i)^2) attention
work -- the whole prefix is recomputed every step. With the cache it is one
O(P^2) prefill plus N steps of O(P+i), which is the standard 50-100x win at
interactive lengths.

Two entry points:
  *             -- greedy / temperature / top-k sampling
  *  -- asserts cached decoding reproduces the
    uncached logits, so we can prove the cache is a pure optimisation.
"""

import torch
import torch.nn.functional as F

from .kv_cache import make_caches


@torch.no_grad()
def generate(model, input_ids, max_new_tokens=64, eos_token_id=None,
             temperature=0.0, top_k=0, use_cache=True, max_len=None):
    """Greedy / sampled decoding with incremental KV cache.

    model: a module exposing  (CausalBackbone) and .
    Returns the full sequence [B, P + N].
    """
    device = input_ids.device
    B, P = input_ids.shape
    if max_len is None:
        max_len = P + max_new_tokens
    caches = None
    if use_cache:
        cfg = model.backbone.config
        caches = make_caches(cfg, B, max_len, device, model.backbone.embed_tokens.weight.dtype)

    # prefill
    h, _ = model.backbone(input_ids, torch.ones_like(input_ids),
                          kv_caches=caches)
    logits = model.lm_head(h[:, -1, :].float())

    out = input_ids
    for _ in range(max_new_tokens):
        if temperature <= 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            lg = logits / temperature
            if top_k > 0:
                k = min(top_k, lg.shape[-1])
                thresh = torch.topk(lg, k, dim=-1).values[:, -1:]
                lg = lg.masked_fill(lg < thresh, float("-inf"))
            probs = F.softmax(lg, dim=-1)
            nxt = torch.multinomial(probs, 1)
        out = torch.cat([out, nxt], dim=1)
        if eos_token_id is not None and bool((nxt == eos_token_id).all()):
            break
        if not use_cache:
            h, _ = model.backbone(out, torch.ones_like(out))
            logits = model.lm_head(h[:, -1, :].float())
        else:
            h, _ = model.backbone(nxt, torch.ones_like(nxt), kv_caches=caches)
            logits = model.lm_head(h[:, -1, :].float())
    return out


@torch.no_grad()
def verify_cache_equivalence(model, input_ids, atol=1e-3, verbose=True):
    """Prove cached decoding == uncached decoding on the same prefix.

    Runs a full forward to get reference logits for positions [0..P-1], then
    replays the same tokens one at a time through the cache and compares the
    logits at every position. A mismatch means the RoPE offset or the mask is
    wrong, which is exactly the class of bug a cache implementation invites.
    """
    B, P = input_ids.shape
    cfg = model.backbone.config
    dtype = model.backbone.embed_tokens.weight.dtype

    h_ref, _ = model.backbone(input_ids, torch.ones_like(input_ids))
    ref = model.lm_head(h_ref.float())              # [B, P, V]

    caches = make_caches(cfg, B, P, input_ids.device, dtype)
    max_diff = 0.0
    for i in range(P):
        tok = input_ids[:, i:i + 1]
        h, _ = model.backbone(tok, torch.ones_like(tok), kv_caches=caches)
        cur = model.lm_head(h[:, -1, :].float())
        d = float((cur - ref[:, i, :]).abs().max())
        max_diff = max(max_diff, d)

    ok = max_diff <= atol
    if verbose:
        print("[kv] cache equivalence: max|diff|=%.6f atol=%.4f -> %s"
              % (max_diff, atol, "PASS" if ok else "FAIL"), flush=True)
    return ok, max_diff


@torch.no_grad()
def benchmark_decode(model, prompt_ids, n_new=32):
    """Wall-clock comparison of cached vs uncached decoding."""
    import time
    t0 = time.time()
    generate(model, prompt_ids, n_new, use_cache=False)
    t_uncached = time.time() - t0
    t0 = time.time()
    generate(model, prompt_ids, n_new, use_cache=True)
    t_cached = time.time() - t0
    return {"uncached_s": t_uncached, "cached_s": t_cached,
            "speedup": t_uncached / max(t_cached, 1e-9)}
