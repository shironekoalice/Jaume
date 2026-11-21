"""Joint (state, action) encoding: does the encoder need to see them together?

Why this script exists
----------------------
The split encoding used everywhere else embeds the state and the action
separately and lets a shallow MLP combine two 512-d vectors:

    score = MLP(concat(embed(state), embed(action)))

That has a structural weakness: mean pooling a state string like

    "Items: item0 slot2, item1 unplaced, ... Constraints: item0 before item3, ..."

into one vector dilutes the correspondence "item3 must go in slot3".  The head
then has to reconstruct that correspondence from two pooled vectors, which
the 2026-11-07 diagnosis showed it does not manage (it learned a slot prior
instead: its top-3 actions were almost always "place itemX at slot3").

The joint encoding removes the bottleneck: the encoder sees

    "<state> || place item3 at slot3"

and the head only has to score one vector that already encodes the pairing.
It is also what the FIRST version of the evaluation accidentally did -- and
the reason that version was wrong was not the concatenation itself but that
the training side encoded the action ALONE while the scoring side encoded the
concatenation.  This script keeps the concatenation and makes both sides use
it.

Cost: one encoder call per (state, action) pair instead of one per state
plus one per distinct action.  On a 910B3 that is affordable; the split
encoding is still available through plan_head_study.py for comparison.

The three label sets are the same as in plan_head_study.py, so the two
scripts are directly comparable.
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
                         run_episode, choose_greedy, choose_random)
from plan_train import optimal_actions
from plan_head_study import collect_all

SEP = " || "


class JointHead(nn.Module):
    """Scores one already-joint-encoded vector."""

    def __init__(self, d, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def joint_texts(states, actions):
    return [s + SEP + a for s, a in zip(states, actions)]


@torch.no_grad()
def beam_joint(enc, tok, head, device, env, budget, width, max_length,
               bs=128):
    """Beam search where each (state, action) pair is encoded jointly."""
    env.reset()
    beams = [(list(env.assignment), 0.0)]
    for _ in range(budget):
        cand = []
        for assign, score in beams:
            e = PlanningEnv(env.n_items, env.n_slots, env.constraints, env.seed)
            e.assignment = list(assign)
            if e.goal_reached():
                return True, e.progress(), budget
            acts = e.legal_actions()
            if not acts:
                continue
            ctx = state_text(e)
            texts = [ctx + SEP + action_text(a) for a in acts]
            em = embed_pair(enc, tok, texts, device, max_length, bs=bs)
            sc = head(em.to(device)).float().cpu().tolist()
            for a, v in zip(acts, sc):
                e2 = PlanningEnv(env.n_items, env.n_slots, env.constraints,
                                 env.seed)
                e2.assignment = list(assign)
                e2.step(a)
                cand.append((list(e2.assignment), score + v))
        if not cand:
            break
        cand.sort(key=lambda x: -x[1])
        beams = cand[:width]
    for assign, _ in beams:
        e = PlanningEnv(env.n_items, env.n_slots, env.constraints, env.seed)
        e.assignment = list(assign)
        if e.goal_reached():
            return True, e.progress(), budget
    return False, 0.0, budget


def train_head(S, y, device, epochs, lr, pos_weight=None):
    head = JointHead(S.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    Sd, yd = S.to(device), y.to(device)
    pw = None if pos_weight is None else torch.tensor(
        float(pos_weight), device=device)
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(len(Sd), device=device)
        tot, n = 0.0, 0
        for i in range(0, len(perm), 128):
            idx = perm[i:i + 128]
            logit = head(Sd[idx])
            loss = F.binary_cross_entropy_with_logits(logit, yd[idx],
                                                      pos_weight=pw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            n += 1
        if (ep + 1) % 10 == 0:
            print(f"      epoch {ep+1}/{epochs} loss={tot/max(n,1):.4f}",
                  flush=True)
    head.eval()
    with torch.no_grad():
        acc = float(((head(Sd) > 0).float() == yd).float().mean())
    return head, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-items", type=int, default=6)
    ap.add_argument("--n-slots", type=int, default=4)
    ap.add_argument("--n-constraints", type=int, default=8)
    ap.add_argument("--train-tasks", type=int, default=300)
    ap.add_argument("--test-tasks", type=int, default=100)
    ap.add_argument("--budget", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--max-length", type=int, default=96)
    ap.add_argument("--width", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)
    print(f"[jt] device={device} ({describe(device)})", flush=True)

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
    print(f"[jt] encoder frozen; d={cfg.embedding_dim}", flush=True)

    seed_all(device, args.seed)
    rng = random.Random(args.seed)

    def make_tasks(n, base):
        out = []
        for t in range(n):
            a, b, cons, sd = generate_task(args.n_items, args.n_slots,
                                           args.n_constraints, seed=base + t)
            env = PlanningEnv(a, b, cons, sd)
            opt = env.optimum()
            if opt is not None:
                out.append((a, b, cons, sd, opt))
        return out

    tr_tasks = make_tasks(args.train_tasks, 1000)
    te_tasks = make_tasks(args.test_tasks, 90000)
    print(f"[jt] train={len(tr_tasks)} test={len(te_tasks)}", flush=True)

    states, actions, y_opt, y_greedy, y_orig = collect_all(
        tr_tasks, args.budget, rng)
    print(f"[jt] {len(states)} pairs", flush=True)

    t0 = time.time()
    texts = joint_texts(states, actions)
    X = embed_pair(enc, tokenizer, texts, device, args.max_length)
    print(f"[jt] embedded {len(texts)} joint texts in {time.time()-t0:.0f}s",
          flush=True)

    base = {"greedy": [], "random": []}
    for (n, m, cons, sd, opt) in te_tasks:
        env = PlanningEnv(n, m, cons, sd)
        ok, prog, _ = run_episode(env, choose_greedy, args.budget)
        base["greedy"].append((ok, prog))
        env = PlanningEnv(n, m, cons, sd)
        ok, prog, _ = run_episode(env, lambda e, a: choose_random(e, a, rng),
                                  args.budget)
        base["random"].append((ok, prog))
    for k, v in base.items():
        print(f"[jt] baseline {k:9s} success="
              f"{sum(1 for r in v if r[0])/max(len(v),1):.3f} "
              f"progress={sum(r[1] for r in v)/max(len(v),1):.3f}", flush=True)

    heads = {}
    for name, y in (("optimal-next", y_opt),
                    ("imitation", y_greedy),
                    ("in-optimal", y_orig)):
        print(f"[jt] --- head[{name}] ---", flush=True)
        head, acc = train_head(X, y, device, args.epochs, args.lr)
        print(f"[jt] head[{name}] train accuracy: {acc:.3f}", flush=True)
        succ, prog = [], []
        for (n, m, cons, sd, opt) in te_tasks:
            env = PlanningEnv(n, m, cons, sd)
            ok, pg, _ = beam_joint(enc, tokenizer, head, device, env,
                                   args.budget, args.width, args.max_length)
            succ.append((ok, pg))
        s_ = sum(1 for r in succ if r[0]) / max(len(succ), 1)
        p_ = sum(r[1] for r in succ) / max(len(succ), 1)
        print(f"[jt] beam[joint-{name:12s}] success={s_:.3f} progress={p_:.3f}",
              flush=True)
        heads[name] = {"train_acc": acc, "success": s_, "progress": p_}

    out = {"args": vars(args), "n_pairs": len(states), "heads": heads,
           "baselines": {k: {"success": sum(1 for r in v if r[0]) / max(len(v), 1),
                             "progress": sum(r[1] for r in v) / max(len(v), 1)}
                         for k, v in base.items()}}
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"[jt] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
