"""Diagnose why KV cache shows no speedup: where does decode time go?"""
import os, sys, time
import torch, torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code", "repo"))

from jaume.device import resolve_device
from jaume.modeling.config import JaumeConfig
from jaume.modeling.embedding import JaumeEmbeddingModel
from jaume.modeling.checkpoint import load_state_dict
from jaume.modeling.kv_cache import make_caches


def main():
    dev = resolve_device("npu")
    model_dir = "/mnt/workspace/output/350m-causal-mlm/final"
    cfg = JaumeConfig.load(os.path.join(model_dir, "config.json"))
    enc = JaumeEmbeddingModel(cfg).to(dev)
    enc.load_state_dict(load_state_dict(os.path.join(model_dir, "pytorch_model.bin")),
                        strict=True)
    enc.eval()

    def sync():
        torch.npu.synchronize()

    # 1) per-op launch overhead
    x = torch.randn(1, 64, 512, device=dev)
    w = torch.randn(512, 512, device=dev)
    sync(); t = time.time()
    for _ in range(50):
        _ = x @ w
    sync(); print("[diag] 50x matmul(1,64,512)@(512,512): %.2fs" % (time.time() - t))

    sync(); t = time.time()
    for _ in range(50):
        _ = x * 1.0001
    sync(); print("[diag] 50x elementwise: %.2fs" % (time.time() - t))

    # 2) full model, one token vs 64 tokens
    ids64 = torch.randint(100, 1000, (1, 64), device=dev)
    ids1 = ids64[:, :1]
    sync(); t = time.time()
    for _ in range(10):
        enc.backbone(ids64, torch.ones_like(ids64))
    sync(); print("[diag] 10x forward(T=64): %.2fs" % (time.time() - t))

    sync(); t = time.time()
    for _ in range(10):
        enc.backbone(ids1, torch.ones_like(ids1))
    sync(); print("[diag] 10x forward(T=1): %.2fs" % (time.time() - t))

    # 3) cached decode: 1 token against a grown cache
    caches = make_caches(cfg, 1, 96, dev, enc.backbone.embed_tokens.weight.dtype)
    h, _ = enc.backbone(ids64, torch.ones_like(ids64), kv_caches=caches)
    tok = ids64[:, :1]
    sync(); t = time.time()
    for _ in range(10):
        enc.backbone(tok, torch.ones_like(tok), kv_caches=caches)
    sync(); print("[diag] 10x cached step (cache=64): %.2fs" % (time.time() - t))

    # 4) how much of a step is MoE?
    xh = torch.randn(1, 1, 512, device=dev)
    layer = enc.backbone.layers[0]
    sync(); t = time.time()
    for _ in range(10):
        layer.mlp(xh)
    sync(); print("[diag] 10x MoE(1 token): %.2fs" % (time.time() - t))


if __name__ == "__main__":
    main()
