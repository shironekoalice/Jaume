"""Pre-flight check for the Ascend (CANN / torch_npu) environment.

Run this INSIDE the Ascend environment before starting the causal twin
training. It answers, in order:

  1. is torch_npu importable and is a device visible?
  2. what device / memory / CANN version are we on?
  3. does the installed torch_npu accept a BOOLEAN attention mask in
     scaled_dot_product_attention? (some CANN builds do not)
  4. does bfloat16 autocast work on the NPU?
  5. does the full 350M causal model do a forward + backward at the
     requested batch size, and what is the measured tokens/s?
  6. does gradient checkpointing fit as well?

It prints an explicit VERDICT line. Exit code 0 = ready to train.

Usage:
    python scripts/check_npu_env.py --batch-size 32 --max-length 128
"""

import argparse
import os
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

FAILS = []
NOTES = []


def check(label, ok, detail=""):
    print(("  [PASS] " if ok else "  [FAIL] ") + label + ("   " + detail if detail else ""))
    if not ok:
        FAILS.append(label)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--config", default=os.path.join(REPO, "configs", "jaume-350m-causal.json"))
    ap.add_argument("--skip-throughput", action="store_true")
    args = ap.parse_args()

    print("=" * 72)
    print("Ascend CANN environment pre-flight")
    print("=" * 72)

    print("")
    print("[1] torch_npu availability")
    try:
        import torch_npu
        print("  torch_npu version:", getattr(torch_npu, "__version__", "unknown"))
        check("import torch_npu", True)
    except Exception as e:
        check("import torch_npu", False, repr(e)[:120])
        print("")
        print("  torch_npu is not importable -- this host is not an Ascend environment.")
        print("  VERDICT: NOT READY")
        return 1

    print("")
    print("[2] device")
    has_npu = hasattr(torch, "npu") and torch.npu.is_available()
    check("torch.npu.is_available()", has_npu)
    if not has_npu:
        print("")
        print("  VERDICT: NOT READY (no NPU visible)")
        return 1

    n = torch.npu.device_count()
    check("device_count >= 1", n >= 1, "count=%d" % n)
    props = torch.npu.get_device_properties(0)
    mem_gib = props.total_memory / 1024 ** 3
    print("  device 0: %s  %.1f GiB" % (props.name, mem_gib))
    torch.npu.set_device(0)

    try:
        print("  CANN:", getattr(torch_npu, "cann_version", lambda: "unknown")())
    except Exception:
        pass

    from jaume.device import resolve_device, describe
    dev = resolve_device()
    check("resolve_device() == npu", dev == "npu", "got %r" % dev)
    print("  describe():", describe(dev))

    print("")
    print("[3] SDPA boolean-mask support")
    from jaume.modeling.nn_compat import sdpa_masked, bool_mask_supported
    q = torch.randn(1, 4, 16, 16, device="npu", dtype=torch.float32)
    k = torch.randn(1, 4, 16, 16, device="npu", dtype=torch.float32)
    v = torch.randn(1, 4, 16, 16, device="npu", dtype=torch.float32)
    allowed = torch.ones(1, 1, 16, 16, dtype=torch.bool, device="npu").tril()
    try:
        out = sdpa_masked(q, k, v, allowed)
        torch.npu.synchronize()
        detail = "bool mask accepted" if bool_mask_supported(q, allowed) else "bool mask REJECTED, float fallback used"
        check("sdpa_masked (bool mask)", bool(torch.isfinite(out).all().item()), detail)
    except Exception as e:
        check("sdpa_masked (bool mask)", False, repr(e)[:140])

    print("")
    print("[4] bfloat16 autocast")
    try:
        with torch.autocast("npu", dtype=torch.bfloat16):
            a = torch.randn(64, 64, device="npu")
            b = (a @ a).float()
        torch.npu.synchronize()
        check("autocast(npu, bfloat16)", bool(torch.isfinite(b).all().item()))
    except Exception as e:
        check("autocast(npu, bfloat16)", False, repr(e)[:140])

    print("")
    print("[4b] primitive operator probe")
    def probe_op(name, fn):
        try:
            fn()
            torch.npu.synchronize()
            print("  [OK]   " + name)
        except Exception as e:
            print("  [FAIL] " + name + "   " + repr(e)[:110])

    dev = "npu"
    xp = torch.randn(64, 32, device=dev)
    mp = torch.rand(64, device=dev) > 0.5
    idxp = torch.tensor([0, 2, 5, 9], device=dev)

    probe_op("boolean index x[mask]", lambda: xp[mp])
    probe_op("sum().item()", lambda: mp.sum().item())
    probe_op("nonzero", lambda: mp.nonzero())
    probe_op("argsort", lambda: torch.argsort(mp.to(torch.int8), descending=True))
    probe_op("gather", lambda: xp.gather(0, idxp.unsqueeze(1).expand(-1, 32)))
    probe_op("index_select", lambda: torch.index_select(xp, 0, idxp))
    probe_op("index_put_", lambda: torch.index_put_(
        torch.zeros_like(xp), (idxp,), torch.ones(4, 32, device=dev)))
    probe_op("masked_fill", lambda: xp.masked_fill(mp.unsqueeze(1), 0.0))
    probe_op("topk", lambda: torch.topk(xp, 2, dim=-1))
    probe_op("cross_entropy", lambda: torch.nn.functional.cross_entropy(
        torch.randn(16, 1000, device=dev), torch.randint(0, 1000, (16,), device=dev)))
    probe_op("silu", lambda: torch.nn.functional.silu(xp))
    probe_op("layer_norm", lambda: torch.nn.functional.layer_norm(xp, (32,)))
    probe_op("softmax", lambda: torch.softmax(xp, dim=-1))
    probe_op("embedding", lambda: torch.nn.functional.embedding(
        torch.tensor([1, 5, 9], device=dev), torch.randn(16, 32, device=dev)))
    # the exact 2-D gather the MLM loss does: hidden.view(-1,H)[sel]
    xh = torch.randn(128, 32, device=dev)
    mh = torch.rand(128, device=dev) > 0.85
    probe_op("2-D boolean row index xh[mh]", lambda: xh[mh])
    probe_op("bool->int8 argsort pack", lambda: xh[
        torch.argsort(mh.to(torch.int8), descending=True)][:int(mh.sum().item())])
    if not args.skip_throughput:
        print("")
        print("[5] full 350M causal model: forward + backward")
        from jaume.modeling.config import JaumeConfig
        from jaume.modeling.embedding import JaumeEmbeddingModel
        from jaume.modeling.backbone_causal import CausalBackbone
        sys.path.insert(0, os.path.join(REPO, "scripts"))
        from train_mlm import MLMHead

        config = JaumeConfig.load(args.config)
        config.vocab_size = 151647
        model = JaumeEmbeddingModel(config).to("npu")
        check("backbone is CausalBackbone", isinstance(model.backbone, CausalBackbone), type(model.backbone).__name__)
        model.train()
        # The real training objective projects hidden states through an MLM
        # head onto the 151647-token vocabulary. Without it this check fed
        # [N, 512] straight into cross_entropy -- a 512-class problem that
        # has nothing to do with the real run.
        mlm_head = MLMHead(config.hidden_size, config.vocab_size).to("npu")
        mlm_head.train()

        B, T = args.batch_size, args.max_length
        ids = torch.randint(1, 150000, (B, T), device="npu")
        mask = torch.ones(B, T, dtype=torch.long, device="npu")
        labels = torch.randint(0, 151647, (B, T), device="npu")

        torch.npu.reset_peak_memory_stats()
        try:
            t0 = time.time()
            with torch.autocast("npu", dtype=torch.bfloat16):
                hidden, aux = model.backbone(ids, mask)
                # No boolean indexing here: Ascend faults on aclnnNonzeroV2.
                # The training script packs the masked rows with argsort; this
                # check mirrors that so it exercises the same code path.
                sel = labels.view(-1) != -100
                n_sel = int(sel.sum().item())
                order = torch.argsort(sel.to(torch.int8), descending=True)
                h_packed = hidden.view(-1, config.hidden_size)[order][:n_sel]
                l_packed = labels.view(-1)[order][:n_sel]
                logits = mlm_head(h_packed)
                loss = torch.nn.functional.cross_entropy(logits.float(), l_packed)
                del logits
            loss.backward()
            torch.npu.synchronize()
            dt = time.time() - t0
            peak = torch.npu.max_memory_allocated() / 1024 ** 3
            tok_s = (B * T) / dt
            print("  batch=%d len=%d  step %.2fs  ~%.0f tok/s  peak %.1f GiB" % (B, T, dt, tok_s, peak))
            check("forward+backward on NPU", True)
            print("  >> rough ETA for 300k sentences x %d tokens = %.1f h on this batch size" % (T, 300000 * T / tok_s / 3600))
        except Exception as e:
            check("forward+backward on NPU", False, repr(e)[:160])
            NOTES.append("OOM here? lower --batch-size, or enable --grad-ckpt")

        print("")
        print("[6] gradient checkpointing")
        try:
            model.backbone.gradient_checkpointing_enable()
            model.zero_grad(set_to_none=True)
            with torch.autocast("npu", dtype=torch.bfloat16):
                hidden, aux = model.backbone(ids, mask)
                loss = hidden.float().pow(2).mean()
            loss.backward()
            torch.npu.synchronize()
            check("gradient checkpointing", True)
        except Exception as e:
            check("gradient checkpointing", False, repr(e)[:160])

    print("")
    print("=" * 72)
    for note in NOTES:
        print("NOTE:", note)
    if FAILS:
        print("VERDICT: NOT READY -- failing checks: " + ", ".join(FAILS))
        return 1
    print("VERDICT: READY -- environment can train the causal twin.")
    print("Next:")
    print("  python scripts/train_mlm_causal.py " + chr(92))
    print("      --config configs/jaume-350m-causal.json " + chr(92))
    print("      --data <path>/mlm_corpus.jsonl " + chr(92))
    print("      --output <path>/output/350m-causal-mlm --max-samples 300000")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
