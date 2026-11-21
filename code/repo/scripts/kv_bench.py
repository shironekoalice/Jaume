"""Benchmark KV cache on the real 350M model: correctness + speedup."""
import os
import sys
import time

import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
REPO = os.path.join(PROJECT_ROOT, "code", "repo")
sys.path.insert(0, REPO)

from jaume.device import resolve_device, describe
from jaume.modeling.config import JaumeConfig
from jaume.modeling.embedding import JaumeEmbeddingModel
from jaume.modeling.checkpoint import load_state_dict
from jaume.modeling.generate import verify_cache_equivalence, benchmark_decode
from jaume.modeling.kv_cache import make_caches, cache_report


class LM(nn.Module):
    def __init__(self, enc, cfg, device):
        super().__init__()
        self.backbone = enc.backbone
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False).to(device)
        with torch.no_grad():
            self.lm_head.weight.copy_(enc.backbone.embed_tokens.weight)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--n-new", type=int, default=32)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)
    print("[kv] device=%s (%s)" % (device, describe(device)), flush=True)

    cfg = JaumeConfig.load(os.path.join(args.model, "config.json"))
    enc = JaumeEmbeddingModel(cfg).to(device)
    enc.load_state_dict(load_state_dict(
        os.path.join(args.model, "pytorch_model.bin")), strict=True)
    enc.eval()
    model = LM(enc, cfg, device).eval()

    ids = torch.randint(100, 1000, (1, args.prompt_len), device=device)

    ok, diff = verify_cache_equivalence(model, ids, atol=2e-2)
    if not ok:
        print("[kv] WARNING: equivalence failed on the real model", flush=True)

    caches = make_caches(cfg, 1, args.prompt_len + args.n_new, device,
                         enc.backbone.embed_tokens.weight.dtype)
    rep = cache_report(caches)
    print("[kv] cache: %d layers, %.1f MiB total (GQA: %d kv heads)"
          % (rep["layers"], rep["total_mib"], cfg.num_key_value_heads),
          flush=True)

    r = benchmark_decode(model, ids, args.n_new)
    print("[kv] decode %d tokens: uncached %.2fs  cached %.2fs  speedup %.1fx"
          % (args.n_new, r["uncached_s"], r["cached_s"], r["speedup"]),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
