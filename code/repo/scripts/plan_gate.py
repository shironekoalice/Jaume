"""P1 gate: does a learned score beat the hand-written heuristic, and is the
gain bigger than what an input-blind control gets?

Why the controls are mandatory
------------------------------
The TTC direction failed in exactly this way: an oracle showed +0.36 macro-F1
of headroom, while every input-blind combination (fixed path, top-k
ensemble, full ensemble) landed inside the seed noise.  A learned scorer
that beats greedy would be the same shape of claim, so it has to survive the
same controls:

  greedy          the hand-written constraint-progress heuristic
  beam-learned    the trained ScoreHead (the claim)
  beam-shuffled   SAME architecture and SAME training budget, but the
                  (state, action) -> label mapping is randomised: if the
                  gain survives this, the gain is not about the task
  beam-random     an untrained head of the same architecture: the floor
                  for "any score function, any ordering"

The claim passes only if

  beam-learned  >  greedy + 2 * binomial_noise
  beam-learned  >  beam-shuffled + 2 * binomial_noise
  beam-learned  >  beam-random  + 2 * binomial_noise

Usage
-----
    python scripts/plan_gate.py --model <ckpt> --seeds 0 1 2 \
        --n-constraints 8 10 12 --train-tasks 300 --test-tasks 100 \
        --budget 30 --epochs 30 --device xpu --out output/plan-gate.json
"""

import argparse
import json
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
REPO = os.path.join(PROJECT_ROOT, "code", "repo")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jaume.device import resolve_device, seed_all, describe
from jaume.modeling.checkpoint import load_state_dict

from planning_env import PlanningEnv, generate_task
from plan_search import (state_text, action_text, embed, embed_pair,
                         ScoreModel, run_episode, choose_greedy, choose_random)
from plan_train import collect, beam_search


def train_head(S, A, y, device, epochs, lr, hidden=None):
    d = S.shape[1]
    head = ScoreModel(d, d).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    Sd, Ad, yd = S.to(device), A.to(device), y.to(device)
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(len(Sd), device=device)
        tot, n = 0.0, 0
        for i in range(0, len(perm), 128):
            idx = perm[i:i + 128]
            logit = head(Sd[idx], Ad[idx])
            loss = F.binary_cross_entropy_with_logits(logit, yd[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            n += 1
        if (ep + 1) % 10 == 0:
            print(f"      epoch {ep+1}/{epochs} loss={tot/max(n,1):.4f}", flush=True)
    head.eval()
    return head


def eval_all(enc, tok, head, device, tasks, budget, max_length, rng):
    """Greedy / random baselines plus the model-guided beam search."""
    out = {"greedy": [], "random": [], "beam-learned": []}
    cache = {}
    for (n, m, cons, sd, opt) in tasks:
        env = PlanningEnv(n, m, cons, sd)
        ok, prog, _ = run_episode(env, choose_greedy, budget)
        out["greedy"].append((ok, prog))

        env = PlanningEnv(n, m, cons, sd)
        ok, prog, _ = run_episode(env, lambda e, a: choose_random(e, a, rng), budget)
        out["random"].append((ok, prog))

        env = PlanningEnv(n, m, cons, sd)
        ok, prog, _ = beam_search(enc, tok, head, device, env, budget,
                                  max_length=max_length, action_cache=cache)
        out["beam-learned"].append((ok, prog))
    return {k: {"success": sum(1 for r in v if r[0]) / max(len(v), 1),
                "progress": sum(r[1] for r in v) / max(len(v), 1)}
            for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-items", type=int, default=6)
    ap.add_argument("--n-slots", type=int, default=4)
    ap.add_argument("--n-constraints", type=int, nargs="+", default=[8])
    ap.add_argument("--train-tasks", type=int, default=300)
    ap.add_argument("--test-tasks", type=int, default=100)
    ap.add_argument("--budget", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)
    print(f"[pg] device={device} ({describe(device)})", flush=True)

    from jaume.modeling.config import JaumeConfig
    from jaume.modeling.embedding import JaumeEmbeddingModel
    from jaume.data.dataset import load_tokenizer

    cfg = JaumeConfig.load(os.path.join(args.model, "config.json"))
    tokdir = os.path.join(args.model, "tokenizer")
    if not os.path.isdir(tokdir):
        tokdir = os.path.join(REPO, "jaume_tokenizer")
    tokenizer = load_tokenizer(tokdir)
    cfg.vocab_size = len(tokenizer)
    enc = JaumeEmbeddingModel(cfg).to(device)
    enc.load_state_dict(
        load_state_dict(os.path.join(args.model, "pytorch_model.bin")), strict=True)
    enc.eval()
    for p_ in enc.parameters():
        p_.requires_grad_(False)
    print(f"[pg] encoder frozen; d={cfg.embedding_dim}", flush=True)

    results = []
    for nc in args.n_constraints:
        for seed in args.seeds:
            tag = f"nc={nc} seed={seed}"
            print(f"[pg] === {tag} ===", flush=True)
            seed_all(device, seed)
            rng = random.Random(seed)

            def make_tasks(n, base):
                out = []
                for t in range(n):
                    a, b, cons, sd = generate_task(args.n_items, args.n_slots,
                                                   nc, seed=base + t)
                    env = PlanningEnv(a, b, cons, sd)
                    opt = env.optimum()
                    if opt is not None:
                        out.append((a, b, cons, sd, opt))
                return out

            tr_tasks = make_tasks(args.train_tasks, 1000)
            te_tasks = make_tasks(args.test_tasks, 90000)
            print(f"[pg] train={len(tr_tasks)} test={len(te_tasks)}", flush=True)

            t0 = time.time()
            rows, eps = collect(enc, tokenizer, device, tr_tasks, nc,
                                args.budget, rng)
            print(f"[pg] collected {len(rows)} pairs in {time.time()-t0:.0f}s",
                  flush=True)

            t0 = time.time()
            s_txt = [r[0] for r in rows]
            a_txt = [r[1] for r in rows]
            y = torch.tensor([r[2] for r in rows], dtype=torch.float32)
            S = embed_pair(enc, tokenizer, s_txt, device, args.max_length)
            A = embed_pair(enc, tokenizer, a_txt, device, args.max_length)
            print(f"[pg] embedded {len(rows)} pairs in {time.time()-t0:.0f}s",
                  flush=True)

            # ---- the three heads -------------------------------------
            head = train_head(S, A, y, device, args.epochs, args.lr)

            srng = random.Random(seed + 997)
            y_shuf = y[torch.randperm(len(y), generator=torch.Generator().manual_seed(seed + 997))]
            head_shuf = train_head(S, A, y_shuf, device, args.epochs, args.lr)

            head_rand = ScoreModel(S.shape[1], A.shape[1]).to(device)
            head_rand.eval()

            conds = eval_all(enc, tokenizer, head, device, te_tasks,
                             args.budget, args.max_length,
                             random.Random(seed + 1))

            # same evaluation, but with the two control heads
            cache = {}
            for name, h in (("beam-shuffled", head_shuf),
                            ("beam-random", head_rand)):
                succ, prog = [], []
                for (n, m, cons, sd, opt) in te_tasks:
                    env = PlanningEnv(n, m, cons, sd)
                    ok, pg, _ = beam_search(enc, tokenizer, h, device, env,
                                            args.budget, max_length=args.max_length,
                                            action_cache=cache)
                    succ.append((ok, pg))
                conds[name] = {
                    "success": sum(1 for r in succ if r[0]) / max(len(succ), 1),
                    "progress": sum(r[1] for r in succ) / max(len(succ), 1)}

            for k, v in conds.items():
                print(f"[pg]   {k:14s} success={v['success']:.3f} "
                      f"progress={v['progress']:.3f}", flush=True)

            n_test = max(len(te_tasks), 1)
            noise = (conds["greedy"]["success"] *
                     (1 - conds["greedy"]["success"]) / n_test) ** 0.5
            g_learn = conds["beam-learned"]["success"] - conds["greedy"]["success"]
            g_shuf = conds["beam-learned"]["success"] - conds["beam-shuffled"]["success"]
            g_rand = conds["beam-learned"]["success"] - conds["beam-random"]["success"]
            passed = (g_learn > 2 * noise and g_shuf > 2 * noise
                      and g_rand > 2 * noise)
            print(f"[pg]   learned-greedy={g_learn:+.4f} "
                  f"learned-shuffled={g_shuf:+.4f} learned-random={g_rand:+.4f} "
                  f"noise={noise:.4f}", flush=True)
            print(f"[pg]   VERDICT: {'PASS' if passed else 'FAIL'}", flush=True)

            results.append({"n_constraints": nc, "seed": seed,
                            "conditions": conds, "noise": noise,
                            "gain_vs_greedy": g_learn,
                            "gain_vs_shuffled": g_shuf,
                            "gain_vs_random": g_rand,
                            "verdict": "PASS" if passed else "FAIL"})

    # ---- aggregate over seeds --------------------------------------
    summary = []
    for nc in args.n_constraints:
        sub = [r for r in results if r["n_constraints"] == nc]
        if not sub:
            continue
        keys = sub[0]["conditions"].keys()
        agg = {}
        for k in keys:
            vals = [r["conditions"][k]["success"] for r in sub]
            mean = sum(vals) / len(vals)
            sd = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
            agg[k] = {"mean": mean, "sd": sd, "values": vals}
        print(f"[pg] === nc={nc} over {len(sub)} seeds ===", flush=True)
        for k, v in agg.items():
            print(f"[pg]   {k:14s} mean={v['mean']:.4f} sd={v['sd']:.4f} "
                  f"{['%.4f' % x for x in v['values']]}", flush=True)
        summary.append({"n_constraints": nc, "n_seeds": len(sub),
                        "aggregate": agg})

    out = {"model": args.model, "args": vars(args), "runs": results,
           "summary": summary}
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"[pg] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
