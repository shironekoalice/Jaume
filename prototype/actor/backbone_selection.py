"""Backbone selection on REAL GUI data (ScreenSpot).

THE QUESTION. The synthetic UI benchmark scores 1.000 for every ablation, so
it cannot rank backbones. ScreenSpot can: 1272 real screenshots across
iOS / Android / macOS / Windows / Web, each with a natural instruction and
the ground-truth box of the element it refers to (Apache-2.0, from SeeClick).

THE METRIC. Not a vendor leaderboard. Published ScreenSpot numbers are all in
COORDINATE-REGRESSION mode (predict the box directly), which is a different
action space from the one this project uses. So every backbone is measured
here in the mode the project will actually use:

    candidate scoring — score K candidate boxes, take the argmax.

The candidate set is CONSTRUCTED (GT box + size-matched distractors) because
ScreenSpot annotates only the target. Distractors are size-matched so the
answer cannot be found by picking the biggest box.

K=2 gives a floor of 0.50 and isolates "did the model find the right crop";
K=6 is the actor's operating regime (floor 0.167); K=20 is what a real
page's DOM shortlist looks like (floor 0.05).

Every backbone is used FROZEN, so the comparison measures the
representation rather than how much compute each was given. If all frozen
numbers are poor, that is itself the finding — the actor needs SOME
training, and the ranking says where to spend it.

Reported per backbone:
  acc@K        candidate-scoring accuracy at each K
  by_type      icon vs text breakdown (icons are the hard case)
  latency      ms per decision
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

import numpy as np
import torch

from backbone_registry import CANDIDATES
from screenspot_data import build_eval_set


def evaluate(scorer, data, device="cpu"):
    """One pass over the data.

    The candidate list has K boxes (GT + size-matched distractors). Two
    numbers are reported:

      top1      the gold is the highest-scoring candidate. Floor = 1/K.
      topK@6    the gold is within the best 6 of the K candidates. This is
                "a real page's DOM shortlist was 6 long and the answer was
                in it" — the actor's actual operating regime. Floor = 6/K.

    Reporting the floor alongside the number is the point: without it a
    0.30 on a 20-candidate set looks impressive and is actually near
    chance (1/20 = 0.05, and 6/20 = 0.30 for the top-6 measure).
    """
    n = len(data)
    K = len(data[0]["candidates"])
    top1 = top6 = 0
    by_type = defaultdict(lambda: [0, 0])
    lat = []
    for d in data:
        t0 = time.time()
        scores = scorer.score(d["image"], d["instruction"], d["candidates"])
        lat.append((time.time() - t0) * 1000)
        order = np.argsort(-scores)
        rank = int(np.where(order == d["gold"])[0][0])
        top1 += int(rank == 0)
        top6 += int(rank < 6)
        by_type[d["data_type"]][0] += int(rank == 0)
        by_type[d["data_type"]][1] += 1
    return dict(
        top1=top1 / n,
        top1_floor=1.0 / K,
        top6=top6 / n,
        top6_floor=min(6, K) / K,
        by_type={t: dict(top1=c / max(v, 1), n=v) for t, (c, v) in by_type.items()},
        latency_ms=float(np.mean(lat)),
        n=n, K=K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=300)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--backbones", default="")
    ap.add_argument("--out", default="backbone_selection.json")
    args = ap.parse_args()

    print(f"building ScreenSpot eval set ({args.rows} samples, K={args.k})...", flush=True)
    data = build_eval_set(max_rows=args.rows, k=args.k, seed=0)
    print(f"  {len(data)} samples | sources: "
          f"{dict((s, sum(1 for d in data if d['source']==s)) for s in set(d['source'] for d in data))}",
          flush=True)
    print(f"  types: {dict((t, sum(1 for d in data if d['data_type']==t)) for t in set(d['data_type'] for d in data))}",
          flush=True)

    names = args.backbones.split(",") if args.backbones else list(CANDIDATES)
    results = {}
    for name in names:
        spec = CANDIDATES[name]
        print(f"\n=== {name} ({spec['id']}) ===", flush=True)
        t0 = time.time()
        try:
            scorer = spec["build"]()
        except Exception as e:
            print(f"  LOAD FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
            results[name] = dict(error=f"{type(e).__name__}: {str(e)[:200]}")
            continue
        print(f"  loaded in {time.time()-t0:.1f}s", flush=True)
        try:
            r = evaluate(scorer, data)
            results[name] = r
            print(f"  top1    {r['top1']:.4f}  (floor {r['top1_floor']:.4f})", flush=True)
            print(f"  top6    {r['top6']:.4f}  (floor {r['top6_floor']:.4f})", flush=True)
            print(f"  by type: {r['by_type']}", flush=True)
            print(f"  {r['latency_ms']:.0f} ms/decision", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  EVAL FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
            results[name] = dict(error=f"{type(e).__name__}: {str(e)[:200]}")
        finally:
            try:
                scorer.unload()
            except Exception:
                pass

    json.dump(results, open(args.out, "w"), indent=2)
    print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
