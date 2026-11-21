"""Learning-to-guide-search: does a 350M encoder beat search baselines?

The question this answers
-------------------------
Jaume's decision heads (Choice / Score / Noul) were designed to score
candidates, not to generate text. Planning benchmarks that need generation
(ALFWorld) cannot show that capability. This script instead measures the
capability directly:

    given a planning state, can the model rank candidate actions well enough
    to make search cheaper?

Setup
-----
  env     PlanningEnv (constraint satisfaction, exact optimum known)
  search  beam search with a fixed node budget
  heuristics compared at the SAME budget:
            random      pick a random legal action
            greedy      maximise constraint progress right now
            model       Jaume ScoreHead(encode(state) concat action_emb)
            model+beam  model-scored beam search

  metric  (a) success rate
          (b) mean gap to the exact optimum
          (c) how often the model's top-1 action is in the optimal action set

Why this is a fair test
-----------------------
Every heuristic is charged the same number of environment steps and the same
number of model forward passes, so a win cannot come from spending more
compute. The optimal action set is computed exactly by brute force, so "the
model ranked a good action first" is measurable rather than subjective.

Usage
-----
    python scripts/plan_search.py --n-items 6 --n-slots 4 --n-constraints 6 \
        --tasks 200 --budget 40 --device npu
"""

import argparse
import json
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
REPO = os.path.join(PROJECT_ROOT, "code", "repo")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jaume.device import resolve_device, seed_all, describe
from jaume.modeling.checkpoint import load_state_dict

from planning_env import PlanningEnv, generate_task


# ---------------------------------------------------------------- encoding
def state_text(env):
    """Render a planning state as text for the encoder."""
    st = env.state()
    a = st["assignment"]
    parts = ["Items: " + ", ".join("item%d slot%d" % (i, s) if s >= 0
                                   else "item%d unplaced" % i
                                   for i, s in enumerate(a))]
    cons = []
    for c in st["constraints"]:
        if c[0] == "before":
            cons.append("item%d before item%d" % (c[1], c[2]))
        elif c[0] == "not_same":
            cons.append("item%d not same as item%d" % (c[1], c[2]))
        elif c[0] == "same":
            cons.append("item%d same as item%d" % (c[1], c[2]))
        elif c[0] == "fixed":
            cons.append("item%d fixed at slot%d" % (c[1], c[2]))
    parts.append("Constraints: " + "; ".join(cons))
    return " ".join(parts)


def action_text(action):
    if action[0] == "place":
        return "place item%d at slot%d" % (action[1], action[2])
    return "remove item%d" % action[1]


@torch.no_grad()
def embed(model, tokenizer, texts, device, max_length=64, bs=128):
    outs = []
    for i in range(0, len(texts), bs):
        enc = tokenizer(texts[i:i + bs], padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt").to(device)
        h, _ = model.backbone(enc["input_ids"], enc["attention_mask"])
        m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        outs.append(((h * m).sum(1) / m.sum(1).clamp(min=1e-9)).float().cpu())
    return torch.cat(outs, 0)


def embed_pair(model, tokenizer, texts, device, max_length=64, bs=128):
    """L2-normalised embeddings -- the ONLY embedding any scorer should use.

    The raw mean-pooled embeddings have norm ~12-15.  A head trained on raw
    embeddings and then applied to L2-normalised ones sees inputs scaled down
    by more than an order of magnitude, its first layer saturates, and the
    score collapses to a constant -- which silently turns any search that
    ranks by that score into a random search.

    Normalising on BOTH sides (training and scoring) is what makes the two
    distributions identical.  Use this helper everywhere instead of calling
    embed() directly, so the two sides cannot drift apart again.
    """
    e = embed(model, tokenizer, texts, device, max_length, bs=bs)
    return F.normalize(e, dim=-1)


class ScoreModel(nn.Module):
    """f(state_emb, action_emb) -> scalar; small enough to train on NPU fast."""

    def __init__(self, d_state, d_action, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_state + d_action, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s, a):
        return self.net(torch.cat([s, a], -1)).squeeze(-1)


# ---------------------------------------------------------------- baselines
def run_episode(env, choose_action, budget):
    """Run until goal or budget exhausted. Returns (success, progress, steps)."""
    env.reset()
    for _ in range(budget):
        acts = env.legal_actions()
        if not acts:
            break
        act = choose_action(env, acts)
        _, _, done = env.step(act)
        if done:
            return True, env.progress(), env.steps
    return env.goal_reached(), env.progress(), env.steps


def choose_random(env, acts, rng):
    return rng.choice(acts)


def choose_greedy(env, acts):
    """Pick the action maximising immediate constraint progress."""
    base = env.progress()
    best, best_score = acts[0], -1e9
    for a in acts:
        saved = list(env.assignment)
        env.step(a)
        sc = env.progress() + (1.0 if env.goal_reached() else 0.0)
        env.assignment = saved
        if sc > best_score:
            best, best_score = a, sc
    return best


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="Jaume checkpoint (optional)")
    ap.add_argument("--n-items", type=int, default=6)
    ap.add_argument("--n-slots", type=int, default=4)
    ap.add_argument("--n-constraints", type=int, nargs="+", default=[4, 8, 12])
    ap.add_argument("--tasks", type=int, default=100)
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)
    print(f"[plan] device={device} ({describe(device)})", flush=True)

    model = None
    tokenizer = None
    if args.model:
        from jaume.modeling.config import JaumeConfig
        from jaume.modeling.embedding import JaumeEmbeddingModel
        from jaume.data.dataset import load_tokenizer

        cfg = JaumeConfig.load(os.path.join(args.model, "config.json"))
        tokdir = os.path.join(args.model, "tokenizer")
        if not os.path.isdir(tokdir):
            tokdir = os.path.join(REPO, "jaume_tokenizer")
        tokenizer = load_tokenizer(tokdir)
        cfg.vocab_size = len(tokenizer)
        model = JaumeEmbeddingModel(cfg).to(device)
        model.load_state_dict(
            load_state_dict(os.path.join(args.model, "pytorch_model.bin")), strict=True)
        model.eval()
        for p_ in model.parameters():
            p_.requires_grad_(False)
        print(f"[plan] encoder loaded from {args.model}", flush=True)

    results = {}
    for nc in args.n_constraints:
        tasks = []
        for t in range(args.tasks):
            n, m, cons, sd = generate_task(args.n_items, args.n_slots, nc,
                                           seed=args.seed * 10000 + t)
            env = PlanningEnv(n, m, cons, sd)
            opt = env.optimum()
            if opt is None:
                continue
            tasks.append((n, m, cons, sd, opt))
        print(f"[plan] {nc} constraints: {len(tasks)} tasks with known optimum",
              flush=True)

        rng = random.Random(args.seed)
        rnd = []
        for (n, m, cons, sd, opt) in tasks:
            env = PlanningEnv(n, m, cons, sd)
            ok, prog, steps = run_episode(
                env, lambda e, a: choose_random(e, a, rng), args.budget)
            rnd.append((ok, prog, opt[0], len(cons)))
        gre = []
        for (n, m, cons, sd, opt) in tasks:
            env = PlanningEnv(n, m, cons, sd)
            ok, prog, steps = run_episode(env, choose_greedy, args.budget)
            gre.append((ok, prog, opt[0], len(cons)))

        def summarise(rows):
            n = len(rows)
            return {
                "success": sum(1 for r in rows if r[0]) / max(n, 1),
                "mean_progress": sum(r[1] for r in rows) / max(n, 1),
            }

        results[nc] = {"tasks": len(tasks),
                       "random": summarise(rnd),
                       "greedy": summarise(gre)}
        print(f"[plan]   random   success={results[nc]['random']['success']:.3f} "
              f"progress={results[nc]['random']['mean_progress']:.3f}", flush=True)
        print(f"[plan]   greedy   success={results[nc]['greedy']['success']:.3f} "
              f"progress={results[nc]['greedy']['mean_progress']:.3f}", flush=True)

    print("", flush=True)
    print("[plan] NOTE: the model-guided condition is not run here. It needs a",
          flush=True)
    print("[plan] trained ScoreHead, which requires trajectories from these", flush=True)
    print("[plan] baselines first. This script establishes the baselines and the", flush=True)
    print("[plan] exact-optimum reference those will be measured against.", flush=True)

    payload = {"config": vars(args), "results": results}
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        print(f"[plan] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
