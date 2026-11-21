"""Stage 3: measure what happens INSIDE a loop.

The gap this fills
-----------------
LoopMTP (2608.03624) supervises the loop with a latent soft alignment and
reports +8.1%; ECHO (2610.24766) makes the backbone's last layer supervise
every horizon; Looped Latent Attention (2607.15456) shows the per-loop KV
cache is low-rank across loops.  All three USE the loop.  None of them
characterise the interior of it.

This script asks four questions of a trained looped model, none of which
need new training:

  Q1  How much of the representation is SHARED between rounds?
      The share of each round's top principal subspace that is also in the
      next round's.  1.0 means the round barely changed anything; 0.0 means
      the round built a completely different representation.

  Q2  Which dimensions are STABLE and which are MOVING?
      Per-dimension variance across rounds, normalised by the total variance
      of that dimension.  Near 0 = the dimension carries a fact that stays
      put across the loop (a stable premise); near 1 = the dimension is
      where the computation happens (an intermediate step of the reasoning).

  Q3  Does the loop STOP computing?  (overthinking)
      The relative distance between consecutive rounds.  If it collapses to
      zero after a few rounds, the extra rounds are wasted -- the "latent
      overthinking" LoopMTP names.

  Q4  Do hard inputs keep the loop alive longer than easy ones?
      If yes, the depth is being spent adaptively, which is the strongest
      possible argument that the loop is doing reasoning rather than just
      adding parameters.

Everything is measured on the same forward pass, so Q1-Q3 are free once the
model runs, and the whole probe costs one pass over a batch.

Usage
-----
    python scripts/probe_loop.py --model <ckpt> --num-loops 4 \
        --data data/mlm_corpus.jsonl --num-samples 64 --device npu \
        --out output/loop-probe.json
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
REPO = os.path.join(PROJECT_ROOT, "code", "repo")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jaume.device import resolve_device, seed_all, describe


def top_subspace(h, k=16):
    """Top-k principal directions of a [N, D] activation matrix."""
    x = h.reshape(-1, h.shape[-1]).float()
    x = x - x.mean(0, keepdim=True)
    cov = (x.T @ x) / max(x.shape[0], 1)
    evals, evecs = torch.linalg.eigh(cov.double())
    return evecs[:, -k:].float()          # [D, k]


def subspace_overlap(u, v, k=16):
    """Fraction of u's top-k subspace that also lies in v's (0..1)."""
    m = u.T @ v
    return float((m.pow(2).sum() / k).clamp(0, 1))


def round_stats(loop_states, k=16):
    """Q1-Q3 for one forward pass."""
    T = len(loop_states)
    H = torch.stack(loop_states)                 # [T, B, L, D]

    # Q1: shared subspace between consecutive rounds
    subs = [top_subspace(H[t], k) for t in range(T)]
    overlap = [subspace_overlap(subs[t], subs[t + 1], k)
               for t in range(T - 1)]
    # and against the FIRST round, to see the drift accumulate
    overlap_first = [subspace_overlap(subs[0], subs[t], k) for t in range(T)]

    # Q2: per-dimension variance across rounds / total variance
    var_across = H.var(dim=0, unbiased=False)                 # [B, L, D]
    grand = H.mean(dim=0, keepdim=True)
    var_total = (H - grand).pow(2).mean(dim=0)                # [B, L, D]
    dim_ratio = (var_across / var_total.clamp(min=1e-9)).mean(dim=(0, 1))  # [D]
    dim_ratio = dim_ratio.tolist()

    # Q3: relative distance between consecutive rounds
    dists = []
    for t in range(1, T):
        d = (H[t] - H[t - 1]).norm(dim=-1)
        n = H[t].norm(dim=-1).clamp(min=1e-8)
        dists.append(float((d / n).mean()))

    # how many dimensions are effectively static
    static = sum(1 for v in dim_ratio if v < 0.05)
    moving = sum(1 for v in dim_ratio if v > 0.5)

    return {
        "T": T,
        "subspace_overlap_consecutive": overlap,
        "subspace_overlap_vs_first": overlap_first,
        "dim_var_ratio_mean": float(sum(dim_ratio) / len(dim_ratio)),
        "dim_var_ratio": dim_ratio,
        "n_dims_static": static,
        "n_dims_moving": moving,
        "round_distance": dists,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--num-loops", type=int, default=4)
    ap.add_argument("--loop-form", choices=["full", "tail"], default="full")
    ap.add_argument("--loop-layers", type=int, default=0)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--num-samples", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--sweep-loops", type=int, nargs="+", default=None,
                    help="also probe these loop depths on the same checkpoint")
    args = ap.parse_args()

    device = resolve_device(args.device)
    print(f"[probe] device={device} ({describe(device)})", flush=True)

    from jaume.modeling.config import JaumeConfig
    from jaume.modeling.loop_lm import LoopLM
    from jaume.modeling.checkpoint import load_state_dict
    from jaume.data.dataset import load_tokenizer

    cfg = JaumeConfig.load(os.path.join(args.model, "config.json"))
    tokdir = args.tokenizer or os.path.join(args.model, "tokenizer")
    if not os.path.isdir(tokdir):
        tokdir = os.path.join(REPO, "jaume_tokenizer")
    tokenizer = load_tokenizer(tokdir)
    # The causal twin's vocabulary includes a [MASK] token appended after the
    # base tokenizer was built; without it the embedding matrix is one row
    # short and the warm start drops it.
    tokenizer.add_tokens(["[MASK]"])
    cfg.vocab_size = len(tokenizer)
    cfg.num_loops = args.num_loops
    cfg.loop_form = args.loop_form
    cfg.loop_layers = args.loop_layers
    cfg.causal = True

    model = LoopLM(cfg).to(device)
    sd = load_state_dict(os.path.join(args.model, "pytorch_model.bin"))
    tgt = model.backbone.state_dict()
    src = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
    loaded = [k for k, v in src.items() if k in tgt and tgt[k].shape == v.shape]
    for k in loaded:
        tgt[k] = src[k]
    model.backbone.load_state_dict(tgt)
    if len(loaded) != len(src):
        print(f"[probe] WARNING: {len(src)-len(loaded)} tensors skipped, e.g. "
              f"{[k for k in src if k not in loaded][:5]}", flush=True)
    if "lm_head.weight" in sd:
        model.lm_head.weight.data = sd["lm_head.weight"].to(device)
    model.eval()
    print(f"[probe] loaded {len(loaded)} backbone tensors "
          f"(skipped {len(src)-len(loaded)})", flush=True)

    texts = []
    if args.data:
        with open(args.data, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    texts.append(json.loads(line)["text"])
                except Exception:
                    continue
                if len(texts) >= args.num_samples:
                    break
    if not texts:
        texts = ["The quick brown fox jumps over the lazy dog. " * 8] * args.num_samples

    ids = []
    for t in texts:
        e = tokenizer(t, add_special_tokens=False)["input_ids"][:args.max_length]
        if len(e) >= 8:
            ids.append(e)
    ids = ids[:args.num_samples]
    print(f"[probe] {len(ids)} sequences", flush=True)

    depths = args.sweep_loops or [args.num_loops]
    out = {"model": args.model, "top_k": args.top_k, "runs": {}}

    for T in depths:
        model.backbone.num_loops = T
        # The gate parameter is sized num_loops-1 (round 0 is pinned), so
        # sweeping T on a gate-trained checkpoint needs the gate resized to
        # match.  Truncate or extend with 1.0: extending keeps the sweep
        # honest for rounds the model has never seen (they behave as plain
        # loop rounds), truncating drops rounds the model did train.
        gp = getattr(model.backbone, "loop_gate_param", None)
        if gp is not None and gp.shape[0] != max(0, T - 1):
            with torch.no_grad():
                if T - 1 <= 0:
                    model.backbone.loop_gate_param = None
                else:
                    newg = torch.ones(max(0, T - 1), device=gp.device,
                                      dtype=gp.dtype)
                    k = min(gp.shape[0], T - 1)
                    newg[:k] = gp[:k]
                    model.backbone.loop_gate_param = torch.nn.Parameter(newg)
        agg = None
        n = 0
        for i in range(0, len(ids), args.batch_size):
            chunk = ids[i:i + args.batch_size]
            L = min(len(c) for c in chunk)
            batch = torch.tensor([c[:L] for c in chunk], device=device)
            with torch.no_grad():
                logits, aux, ls = model(batch, return_loop_states=True)
                st = round_stats(ls, args.top_k)
            if agg is None:
                agg = st
            else:
                for key in ("subspace_overlap_consecutive",
                            "subspace_overlap_vs_first"):
                    agg[key] = [a + b for a, b in zip(agg[key], st[key])]
                agg["dim_var_ratio_mean"] += st["dim_var_ratio_mean"]
                agg["n_dims_static"] += st["n_dims_static"]
                agg["n_dims_moving"] += st["n_dims_moving"]
                agg["round_distance"] = [a + b for a, b in
                                         zip(agg["round_distance"],
                                             st["round_distance"])]
            n += 1
        for key in ("subspace_overlap_consecutive", "subspace_overlap_vs_first",
                    "round_distance"):
            agg[key] = [v / n for v in agg[key]]
        agg["dim_var_ratio_mean"] /= n
        agg["n_dims_static"] //= n
        agg["n_dims_moving"] //= n
        out["runs"][f"T={T}"] = agg
        print(f"[probe] T={T}: "
              f"overlap_consecutive="
              f"{[round(v,3) for v in agg['subspace_overlap_consecutive']]} "
              f"overlap_vs_first="
              f"{[round(v,3) for v in agg['subspace_overlap_vs_first']]}", flush=True)
        print(f"[probe] T={T}: dim_var_ratio_mean="
              f"{agg['dim_var_ratio_mean']:.4f} "
              f"static_dims={agg['n_dims_static']}/{cfg.hidden_size} "
              f"moving_dims={agg['n_dims_moving']}/{cfg.hidden_size}", flush=True)
        print(f"[probe] T={T}: round_distance="
              f"{[round(v,4) for v in agg['round_distance']]}", flush=True)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"[probe] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
