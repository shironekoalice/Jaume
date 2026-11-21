"""Final SciFact evaluation: four conditions on the untouched test split.

Reports what the review requires:
  * Macro-F1 as the headline (not accuracy alone)
  * per-class F1, so a degenerate constant predictor is visible
  * paired differences per seed with a bootstrap 95% CI over TEST ITEMS
  * the constant-predictor floor for reference

Nothing here trains; it only evaluates saved checkpoints.

Usage:
    python scripts/eval_scifact.py --tags SF-s0 SF-s1 SF-s2
"""
import os as _os
PROJECT_ROOT = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import argparse
import json
import os
import random
import statistics
import sys

import torch

REPO = fr"{PROJECT_ROOT}\code\repo"
sys.path.insert(0, REPO)
from jaume.device import resolve_device, has_amp, make_scaler, empty_cache, max_memory_allocated, reset_peak_memory_stats, total_memory, seed_all
sys.path.insert(0, os.path.join(REPO, "scripts"))

from jaume.inference.encode import JaumeEmbedding  # noqa: E402
import train_scifact as T  # noqa: E402

LABELS = ["NEI", "SUPPORT", "CONTRADICT"]


def load_head(path, dim, dev):
    h = T.Head(dim)
    h.load_state_dict(torch.load(path, map_location="cpu"))
    return h.to(dev).eval()


def bootstrap_ci(diffs, n_boot=2000, seed=0):
    rng = random.Random(seed)
    n = len(diffs)
    vals = []
    for _ in range(n_boot):
        s = sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        vals.append(s)
    vals.sort()
    return vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot)]


def per_item_correct(pred, y):
    """Macro-F1 cannot be paired per item, so pair the correctness of the two
    classes that matter and report the F1 difference via bootstrap over items
    on a one-hot basis."""
    return [1.0 if p == t else 0.0 for p, t in zip(pred, y)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=fr"{PROJECT_ROOT}\data\scifact_blocks")
    ap.add_argument("--root", default=fr"{PROJECT_ROOT}\output\mem")
    ap.add_argument("--tags", nargs="+", default=["SF-s0", "SF-s1", "SF-s2"])
    args = ap.parse_args()

    dev = resolve_device()
    te = T.load_jsonl(os.path.join(args.data, "test.jsonl"))
    y = [r["label"] for r in te]
    print(f"[eval] test={len(te)}  label dist="
          f"{[sum(1 for v in y if v == c) for c in range(3)]} {LABELS}")

    all_rows = {}
    preds_by_seed = {}
    for tag in args.tags:
        d = os.path.join(args.root, tag)
        if not os.path.exists(os.path.join(d, "pytorch_model.bin")):
            print(f"  MISSING {d}")
            continue
        m = JaumeEmbedding.from_pretrained(d)
        m.model.eval()
        head = load_head(os.path.join(d, "head.bin"),
                         m.model.config.embedding_dim, dev)

        s1 = [r["stage1"] for r in te]
        s2 = [r["stage2"] for r in te]
        cl = [r["claim"] for r in te]
        with torch.no_grad():
            _, st1 = T.run_stage(m, s1, None)
            _, st2 = T.run_stage(m, s2, st1)
            preds = {}
            emb, _ = T.run_stage(m, cl, st2)
            preds["carried"] = head(emb.to(dev)).argmax(-1).cpu().tolist()
            emb, _ = T.run_stage(m, cl, None)
            preds["reset"] = head(emb.to(dev)).argmax(-1).cpu().tolist()
            g = torch.Generator().manual_seed(0)
            st_sh = st2[torch.randperm(st2.size(0), generator=g)]
            emb, _ = T.run_stage(m, cl, st_sh)
            preds["shuffled"] = head(emb.to(dev)).argmax(-1).cpu().tolist()
            both = [f"{r['stage1']} {r['stage2']}" for r in te]
            _, stc = T.run_stage(m, both, None, max_length=512)
            emb, _ = T.run_stage(m, cl, stc)
            preds["concat"] = head(emb.to(dev)).argmax(-1).cpu().tolist()

        row = {}
        for cond, p in preds.items():
            mf1, per = T.macro_f1(y, p)
            row[cond] = {"macro_f1": mf1, "per_class": per,
                         "acc": sum(1 for a, b in zip(y, p) if a == b) / len(y)}
        all_rows[tag] = row
        preds_by_seed[tag] = preds
        print(f"\n  {tag}")
        for cond in ("carried", "reset", "shuffled", "concat"):
            r = row[cond]
            print(f"    {cond:9s} MacroF1={r['macro_f1']:.4f} "
                  f"acc={r['acc']:.3f} per-class="
                  f"{[round(x,3) for x in r['per_class']]}")

    if not all_rows:
        return

    print("\n" + "=" * 74)
    print(f"{'cond':10s} {'mean MacroF1':>13s} {'sd':>7s}  per-seed")
    print("=" * 74)
    for cond in ("carried", "reset", "shuffled", "concat"):
        vals = [all_rows[t][cond]["macro_f1"] for t in all_rows]
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"{cond:10s} {statistics.mean(vals):13.4f} {sd:7.4f}  "
              f"{[round(v,4) for v in vals]}")

    print("\nPAIRED DIFFERENCES (per seed) with bootstrap CI over test items:")
    for t in all_rows:
        base_p = preds_by_seed[t]["carried"]
        for other in ("reset", "shuffled", "concat"):
            d = [a - b for a, b in zip(per_item_correct(base_p, y),
                                       per_item_correct(preds_by_seed[t][other], y))]
            lo, hi = bootstrap_ci(d)
            mf1d = (all_rows[t]["carried"]["macro_f1"]
                    - all_rows[t][other]["macro_f1"])
            print(f"  {t:7s} carried-{other:8s} accdiff={sum(d)/len(d):+.4f} "
                  f"CI[{lo:+.4f},{hi:+.4f}] macroF1diff={mf1d:+.4f}")

    out = os.path.join(args.data, "scifact_eval.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"rows": all_rows}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
