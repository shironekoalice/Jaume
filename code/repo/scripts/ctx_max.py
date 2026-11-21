"""Find the real maximum context: run the causal checkpoint at increasing
lengths on CPU and report where it breaks (OOM / numeric / time)."""
import os, sys, time, torch, torch.nn as nn, torch.nn.functional as F
PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code", "repo"))
from jaume.device import resolve_device
from jaume.modeling.config import JaumeConfig
from jaume.modeling.embedding import JaumeEmbeddingModel
from jaume.modeling.checkpoint import load_state_dict

def main():
    dev = resolve_device("cpu")
    md = "/mnt/workspace/output/causal-350m-v1"
    cfg = JaumeConfig.load(os.path.join(md, "config.json"))
    enc = JaumeEmbeddingModel(cfg).to(dev)
    sd = load_state_dict(os.path.join(md, "pytorch_model.bin"))
    own = enc.state_dict(); merged = {}
    for k in own:
        merged[k] = sd[k] if (k in sd and sd[k].shape == own[k].shape) else own[k]
    enc.load_state_dict(merged, strict=True)
    enc.eval()
    w = sd.get("lm_head.weight")
    head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False).to(dev)
    with torch.no_grad():
        head.weight.copy_(w if w is not None else sd["backbone.embed_tokens.weight"])
    head.eval()
    print("[ctxmax] advertised max_seq_len = %d" % cfg.max_seq_len, flush=True)
    for L in (1024, 2048, 4096, 8192, 16384):
        ids = torch.randint(100, 1000, (1, L), device=dev)
        t0 = time.time()
        try:
            with torch.no_grad():
                h, _ = enc.backbone(ids, torch.ones_like(ids))
                logits = head(h.float())
                loss = F.cross_entropy(logits[0, :-1], ids[0, 1:])
            ok = torch.isfinite(loss).item()
            print("[ctxmax] L=%5d  loss=%8.3f  finite=%s  %.1fs" % (
                L, float(loss), ok, time.time() - t0), flush=True)
        except Exception as e:
            print("[ctxmax] L=%5d  FAILED: %s: %s (%.1fs)" % (
                L, type(e).__name__, str(e)[:80], time.time() - t0), flush=True)
            break

if __name__ == "__main__":
    main()
