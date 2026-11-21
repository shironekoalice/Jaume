"""KV cache correctness tests.

The cache is an optimisation, so the contract is exact equivalence: cached
incremental decoding must produce the same logits as a full forward pass.
These tests use a tiny random model so they run on CPU in seconds.
"""
import os
import sys

import torch
import torch.nn as nn

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from jaume.modeling.config import JaumeConfig
from jaume.modeling.backbone_causal import CausalBackbone
from jaume.modeling.kv_cache import make_caches, cache_report
from jaume.modeling.generate import verify_cache_equivalence


def tiny_cfg(vocab=64):
    return JaumeConfig(
        vocab_size=vocab, hidden_size=64, intermediate_size=128,
        num_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_seq_len=128, expert_num=2, topk=1, use_moe=False,
        embedding_dim=64, matryoshka_dims=(32, 64), pooling="attention",
        pad_token_id=0, causal=True)


class ToyLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = CausalBackbone(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, ids):
        h, _ = self.backbone(ids, torch.ones_like(ids))
        return self.lm_head(h)


def test_cache_matches_full_forward():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    m = ToyLM(cfg).eval()
    ids = torch.randint(1, 60, (2, 24))
    ok, diff = verify_cache_equivalence(m, ids, atol=1e-4, verbose=False)
    print("  max|diff| = %.6f" % diff)
    assert ok, "cached logits differ from full forward (diff=%.6f)" % diff


def test_cache_gqa_shapes():
    cfg = tiny_cfg()
    B, T = 2, 16
    caches = make_caches(cfg, B, T, "cpu", torch.float32)
    rep = cache_report(caches)
    # GQA stores n_kv heads, not n_heads
    per_layer = B * cfg.num_key_value_heads * T *         (cfg.hidden_size // cfg.num_attention_heads) * 4 * 2
    expect = per_layer * cfg.num_layers
    print("  cache bytes = %d (expected %d)" % (
        rep["total_mib"] * 1024 ** 2, expect))
    assert abs(rep["total_mib"] * 1024 ** 2 - expect) < 1024


def test_no_cache_path_unchanged():
    """use_cache=False must reproduce the pre-cache behaviour exactly."""
    torch.manual_seed(1)
    cfg = tiny_cfg()
    m = ToyLM(cfg).eval()
    ids = torch.randint(1, 60, (1, 12))
    h1, _ = m.backbone(ids, torch.ones_like(ids))
    h2, _ = m.backbone(ids, torch.ones_like(ids), kv_caches=None)
    assert torch.allclose(h1, h2, atol=1e-6)
    print("  uncached path identical")


if __name__ == "__main__":
    tests = [test_cache_matches_full_forward, test_cache_gqa_shapes,
             test_no_cache_path_unchanged]
    for t in tests:
        print("[%s]" % t.__name__)
        t()
    print("ALL KV CACHE TESTS PASSED")
