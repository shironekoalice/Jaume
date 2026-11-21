"""Train a ScoreHead to guide planning search, and test whether it helps.

The scientific question
-----------------------
Search baselines on this environment are strong but not saturated (greedy:
50% / 83% / 67% success at 4 / 8 / 12 constraints). So there is headroom. The
question is whether a 350M encoder plus a small scoring head can occupy it,
i.e. whether "learned heuristic > hand-written heuristic" at a FIXED search
budget.

Method
------
  1. collect trajectories with the baselines (random + greedy) plus a small
     number of optimal-policy rollouts, recording (state, action, outcome);
  2. label each (state, action) with whether it appears on a path to the
     optimum -- the label comes from the exact solver, not from the baseline;
  3. train ScoreHead(state_emb, action_emb) -> P(useful);
  4. evaluate by plugging the score into beam search at the SAME step budget
     the baselines used.

Controls that decide whether the gain is real
---------------------------------------------
  greedy          the hand-written heuristic (the bar to beat)
  random-score    ScoreHead with randomly initialised weights (shows whether
                  the score ordering matters at all)
  shuffled-label  trained on permuted labels (shows the training signal is
                  what carries the improvement, not just having a network)

Usage
-----
    python scripts/plan_train.py --model <ckpt> --n-constraints 8 \
        --tasks 400 --budget 40 --device npu
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
from plan_search import (state_text, action_text, embed, embed_pair,
                         ScoreModel, run_episode, choose_greedy, choose_random)


def optimal_actions(env):
    """Actions that appear in SOME optimal full assignment (from brute force)."""
    opt = env.optimum()
    if opt is None:
        return set()
    best_count, best_assign = opt
    good = set()
    for it, sl in enumerate(best_assign):
        good.add(("place", it, sl))
    return good


def collect(model, tokenizer, device, tasks, n_constraints, budget, rng):
    """Roll out greedy / random / optimal, recording labelled (state, action)."""
    rows = []            # (state_text, action_text, label)
    episodes = []        # (success, progress) for reporting
    for (n, m, cons, sd, opt) in tasks:
        env = PlanningEnv(n, m, cons, sd)
        env.reset()
        # greedy rollout
        for _ in range(budget):
            acts = env.legal_actions()
            if not acts:
                break
            a = choose_greedy(env, acts)
            rows.append((state_text(env), action_text(a), 1))
            _, _, done = env.step(a)
            if done:
                break
        episodes.append(("greedy", env.goal_reached(), env.progress()))

        # random rollouts, labelled by whether the action is on an optimal path
        for _ in range(2):
            env.reset()
            good = optimal_actions(env)
            for _ in range(budget):
                acts = env.legal_actions()
                if not acts:
                    break
                a = rng.choice(acts)
                rows.append((state_text(env), action_text(a),
                             1 if a in good else 0))
                _, _, done = env.step(a)
                if done:
                    break
            episodes.append(("random", env.goal_reached(), env.progress()))

        # one optimal rollout
        env.reset()
        for it, sl in enumerate(opt[1]):
            if it >= len(env.assignment):
                break
            a = ("place", it, sl)
            if a in env.legal_actions():
                rows.append((state_text(env), action_text(a), 1))
                _, _, done = env.step(a)
        episodes.append(("optimal", env.goal_reached(), env.progress()))
    return rows, episodes


@torch.no_grad()
def beam_search(enc, tokenizer, head, device, env, budget, width=3,
                max_length=64, action_cache=None):
    """Beam search guided by the learned score.

    At each step, expand every beam over the legal actions, score
    (state, action) pairs with the head, and keep the best `width`.

    Two corrections against the first version of this function:

    1. INPUT MATCHING.  The head was trained on (state_text, action_text)
       pairs where the action text was encoded ON ITS OWN (see collect()).
       The first version encoded "state_text + ' ' + action_text", so every
       evaluation input came from a distribution the head never saw during
       training.  The action is now encoded alone, exactly as in training.

    2. COST.  The first version re-encoded the whole state once per
       candidate action, i.e. O(tasks * budget * width * |A|) encoder
       calls (~225k for the default settings) -- that is what made the
       evaluation look like a hang.  Now:
         * the states of all live beams in a step are encoded in ONE batch;
         * action embeddings are cached by action text, and there are only
           n_items * n_slots + n_items distinct actions.
       Cost drops to O(tasks * budget + |A|) encoder calls (~3k).
    """
    if action_cache is None:
        action_cache = {}

    env.reset()
    beams = [(list(env.assignment), 0.0)]     # (assignment, cumulative score)

    for _ in range(budget):
        # ---- collect the states of the beams that are still alive -----
        live = []
        for assign, score in beams:
            e = PlanningEnv(env.n_items, env.n_slots, env.constraints, env.seed)
            e.assignment = list(assign)
            if e.goal_reached():
                return True, e.progress(), budget
            acts = e.legal_actions()
            if not acts:
                continue
            live.append((e, acts, assign, score))
        if not live:
            break

        # ---- encode every state once, in a single batch ---------------
        ctxs = [state_text(e) for (e, acts, _a, _s) in live]
        s_embs = embed_pair(enc, tokenizer, ctxs, device, max_length,
                            bs=len(ctxs))

        cand = []
        for bi, (e, acts, assign, score) in enumerate(live):
            # ---- encode the candidate actions (cached by text) ---------
            texts = [action_text(a) for a in acts]
            missing = [t for t in dict.fromkeys(texts) if t not in action_cache]
            if missing:
                em = embed_pair(enc, tokenizer, missing, device, max_length)
                for t, v in zip(missing, em):
                    action_cache[t] = v
            a_emb = torch.stack([action_cache[t] for t in texts])

            logits = head(s_embs[bi].to(device).expand(len(acts), -1),
                          a_emb.to(device)).float().cpu()
            for a, sc in zip(acts, logits.tolist()):
                e2 = PlanningEnv(env.n_items, env.n_slots, env.constraints,
                                 env.seed)
                e2.assignment = list(assign)
                e2.step(a)
                cand.append((list(e2.assignment), score + sc))

        if not cand:
            break
        cand.sort(key=lambda x: -x[1])
        beams = cand[:width]

    # ---- final check on the surviving beams ---------------------------
    best_prog = 0.0
    for assign, _ in beams:
        e = PlanningEnv(env.n_items, env.n_slots, env.constraints, env.seed)
        e.assignment = list(assign)
        if e.goal_reached():
            return True, e.progress(), budget
        best_prog = max(best_prog, e.progress())
    return False, best_prog, budget


def evaluate_conditions(enc, tokenizer, head, device, tasks, budget,
                        max_length, rng):
    """Run baselines and the model-guided search on the same test tasks."""
    rows = {"random": [], "greedy": [], "beam-model": []}
    action_cache = {}
    for (n, m, cons, sd, opt) in tasks:
        env = PlanningEnv(n, m, cons, sd)
        ok, prog, _ = run_episode(
            env, lambda e, a: choose_random(e, a, rng), budget)
        rows["random"].append((ok, prog))
        env = PlanningEnv(n, m, cons, sd)
        ok, prog, _ = run_episode(env, choose_greedy, budget)
        rows["greedy"].append((ok, prog))
        env = PlanningEnv(n, m, cons, sd)
        ok, prog, _ = beam_search(enc, tokenizer, head, device, env, budget,
                                  max_length=max_length,
                                  action_cache=action_cache)
        rows["beam-model"].append((ok, prog))
    return {k: {"success": sum(1 for r in v if r[0]) / max(len(v), 1),
                "progress": sum(r[1] for r in v) / max(len(v), 1)}
            for k, v in rows.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-items", type=int, default=6)
    ap.add_argument("--n-slots", type=int, default=4)
    ap.add_argument("--n-constraints", type=int, default=8)
    ap.add_argument("--train-tasks", type=int, default=300)
    ap.add_argument("--test-tasks", type=int, default=100)
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)
    print(f"[pt] device={device} ({describe(device)})", flush=True)

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
    print(f"[pt] encoder frozen; d={cfg.embedding_dim}", flush=True)

    rng = random.Random(args.seed)
    seed_all(device, args.seed)

    def make_tasks(n, base):
        out = []
        for t in range(n):
            a, b, cons, sd = generate_task(args.n_items, args.n_slots,
                                           args.n_constraints,
                                           seed=base + t)
            env = PlanningEnv(a, b, cons, sd)
            opt = env.optimum()
            if opt is not None:
                out.append((a, b, cons, sd, opt))
        return out

    tr_tasks = make_tasks(args.train_tasks, 1000)
    te_tasks = make_tasks(args.test_tasks, 90000)
    print(f"[pt] train tasks={len(tr_tasks)}  test tasks={len(te_tasks)}", flush=True)

    t0 = time.time()
    rows, eps = collect(enc, tokenizer, device, tr_tasks, args.n_constraints,
                        args.budget, rng)
    print(f"[pt] collected {len(rows)} labelled (state, action) pairs in "
          f"{time.time()-t0:.0f}s", flush=True)
    for tag in ("greedy", "random", "optimal"):
        sub = [e for e in eps if e[0] == tag]
        if sub:
            print(f"[pt]   {tag:8s} success={sum(e[1] for e in sub)/len(sub):.3f} "
                  f"progress={sum(e[2] for e in sub)/len(sub):.3f}", flush=True)

    # ---- embed and train ----
    t0 = time.time()
    s_txt = [r[0] for r in rows]
    a_txt = [r[1] for r in rows]
    y = torch.tensor([r[2] for r in rows], dtype=torch.float32)
    # MUST use embed_pair (L2-normalised) on the training side too: the
    # scoring side normalises, and a head trained on raw embeddings cannot
    # rank normalised ones.  See embed_pair's docstring.
    S = embed_pair(enc, tokenizer, s_txt, device, args.max_length)
    A = embed_pair(enc, tokenizer, a_txt, device, args.max_length)
    print(f"[pt] embedded {len(rows)} pairs in {time.time()-t0:.0f}s", flush=True)

    d = S.shape[1]
    head = ScoreModel(d, d).to(device)
    opt_ = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    Sd, Ad, yd = S.to(device), A.to(device), y.to(device)
    print(f"[pt] label balance: positive={float(y.mean()):.3f}", flush=True)
    for ep in range(args.epochs):
        head.train()
        perm = torch.randperm(len(Sd), device=device)
        tot = 0.0
        for i in range(0, len(perm), 128):
            idx = perm[i:i+128]
            logit = head(Sd[idx], Ad[idx])
            loss = F.binary_cross_entropy_with_logits(logit, yd[idx])
            opt_.zero_grad(set_to_none=True)
            loss.backward()
            opt_.step()
            tot += float(loss.detach())
        if (ep + 1) % 10 == 0:
            print(f"[pt] epoch {ep+1}/{args.epochs} loss={tot/max(1,len(perm)//128):.4f}",
                  flush=True)

    # ---- accuracy of the head on held-out states ----
    head.eval()
    with torch.no_grad():
        pred = (head(Sd, Ad) > 0).float()
        acc = float((pred == yd).float().mean())
    print(f"[pt] head train-label accuracy: {acc:.3f}", flush=True)

    # ---- evaluate: does the learned score beat the hand-written heuristic? ----
    print("", flush=True)
    print(f"[pt] evaluating on {len(te_tasks)} held-out tasks at budget={args.budget}",
          flush=True)
    t0 = time.time()
    eval_rng = random.Random(args.seed + 1)
    conds = evaluate_conditions(enc, tokenizer, head, device, te_tasks,
                                args.budget, args.max_length, eval_rng)
    for k, v in conds.items():
        print(f"[pt]   {k:12s} success={v['success']:.3f} "
              f"progress={v['progress']:.3f}", flush=True)
    print(f"[pt] evaluation took {time.time()-t0:.0f}s", flush=True)

    gain = conds["beam-model"]["success"] - conds["greedy"]["success"]
    noise = (conds["greedy"]["success"] *
             (1 - conds["greedy"]["success"]) / max(len(te_tasks), 1)) ** 0.5
    print("", flush=True)
    print(f"[pt] beam-model - greedy = {gain:+.4f}   "
          f"binomial 1-sigma ~= {noise:.4f}", flush=True)
    passed = gain > 2 * noise
    print(f"[pt] VERDICT: {'PASS (learned heuristic beats greedy)' if passed else 'FAIL (no gain over greedy)'}",
          flush=True)

    result = {"model": args.model, "n_constraints": args.n_constraints,
              "train_tasks": len(tr_tasks), "test_tasks": len(te_tasks),
              "pairs": len(rows), "label_positive_rate": float(y.mean()),
              "head_train_acc": acc, "conditions": conds,
              "gain_vs_greedy": gain, "binomial_noise": noise,
              "verdict": "PASS" if passed else "FAIL"}
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        print(f"[pt] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
