"""Which of the three training signals can the head actually learn?

The 2026-11-07 diagnosis showed the mechanism is fine and the head is not:

    beam[oracle]       success 1.000   (brute-force optimal action set)
    beam[greedy-score] success 0.900   (hand-written immediate progress)
    beam[head]         success 0.000   (the trained head)

So the question is no longer "does beam search work" -- it is "can the head
learn a usable state-conditioned score".  That splits into two very different
diagnoses:

  (a) the SIGNAL carries no state information, so no head can learn it;
  (b) the signal is learnable but this head/architecture cannot extract it.

Three label sets, one embedding pass, one architecture, three beams:

  optimal-next  1 for the action that continues SOME optimal solution from
                the current partial assignment.  This is a policy label:
                exactly one positive per state, on the optimal path only.
                The sharpest possible version of "what should happen next".

  imitation     1 for the action the hand-written greedy heuristic would take.
                The ceiling for "can the head see the state at all": if it
                cannot match greedy here, the encoder representation is not
                carrying the state, and no amount of label engineering helps.

  in-optimal    1 if the action appears in some optimal full assignment.
                The original label.  Weaker: several actions can be positive
                at once and the state barely matters.

Reading the result:
  * imitation ~ greedy and optimal-next ~ greedy  => signal is fine, the
    original label was the problem;
  * imitation far below greedy                     => the encoder does not
    expose the state to a shallow head, which is a capacity result about the
    350M base model, not about the labels;
  * all three ~ 0                                    => something else is
    still broken.

Usage
-----
    python scripts/plan_head_study.py --model <ckpt> --train-tasks 300 \
        --test-tasks 100 --budget 30 --width 3 --device npu \
        --out output/plan-head-study.json
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
from plan_train import beam_search, optimal_actions


def collect_all(tasks, budget, rng):
    """One pass over the tasks, three label sets, deduplicated pairs.

    Returns (states, actions, lab_opt, lab_greedy, lab_orig) where the label
    lists are aligned with the (state, action) lists.
    """
    states, actions = [], []
    index = {}
    lab_opt, lab_greedy, lab_orig = [], [], []

    def push(st, at):
        key = (st, at)
        i = index.get(key)
        if i is None:
            i = len(states)
            index[key] = i
            states.append(st)
            actions.append(at)
            lab_opt.append(0.0)
            lab_greedy.append(0.0)
            lab_orig.append(0.0)
        return i

    for (n, m, cons, sd, opt) in tasks:
        env = PlanningEnv(n, m, cons, sd)
        good = optimal_actions(env)
        best = opt[1]

        # ---- greedy rollout: every legal action, labelled by greedy ----
        env.reset()
        for _ in range(budget):
            acts = env.legal_actions()
            if not acts:
                break
            g = choose_greedy(env, acts)
            st = state_text(env)
            for a in acts:
                i = push(st, action_text(a))
                if a == g:
                    lab_greedy[i] = 1.0
                if a in good:
                    lab_orig[i] = 1.0
            env.step(g)

        # ---- optimal path: exactly one positive action per state -------
        env.reset()
        for it, sl in enumerate(best):
            acts = env.legal_actions()
            if not acts:
                break
            nxt = ("place", it, sl)
            if nxt not in acts:
                break
            st = state_text(env)
            for a in acts:
                i = push(st, action_text(a))
                if a == nxt:
                    lab_opt[i] = 1.0
                if a in good:
                    lab_orig[i] = 1.0
            env.step(nxt)

        # ---- random rollouts: only the taken action, as in the original
        env.reset()
        for _ in range(budget):
            acts = env.legal_actions()
            if not acts:
                break
            a = rng.choice(acts)
            i = push(state_text(env), action_text(a))
            if a in good:
                lab_orig[i] = 1.0
            env.step(a)

    return (states, actions,
            torch.tensor(lab_opt, dtype=torch.float32),
            torch.tensor(lab_greedy, dtype=torch.float32),
            torch.tensor(lab_orig, dtype=torch.float32))


def train_head(S, A, y, device, epochs, lr):
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
            print(f"      epoch {ep+1}/{epochs} loss={tot/max(n,1):.4f}",
                  flush=True)
    head.eval()
    with torch.no_grad():
        acc = float(((head(Sd, Ad) > 0).float() == yd).float().mean())
    return head, acc


def top1_quality(head, S, A, y, states_idx, actions_idx, good_of_state):
    """How often is the head's top-1 action the labelled positive?"""
    hits = 0
    by_state = {}
    for i, si in enumerate(states_idx):
        by_state.setdefault(si, []).append(i)
    for si, idxs in by_state.items():
        s = S[si].unsqueeze(0)
        a = A[idxs]
        sc = head(s.expand(len(idxs), -1), a).float().cpu()
        best = idxs[int(sc.argmax())]
        hits += 1 if y[best] > 0 else 0
    return hits / max(len(by_state), 1)


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
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--width", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)
    print(f"[hs] device={device} ({describe(device)})", flush=True)

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
    print(f"[hs] encoder frozen; d={cfg.embedding_dim}", flush=True)

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
    print(f"[hs] train={len(tr_tasks)} test={len(te_tasks)}", flush=True)

    t0 = time.time()
    states, actions, y_opt, y_greedy, y_orig = collect_all(
        tr_tasks, args.budget, rng)
    print(f"[hs] {len(states)} unique (state, action) pairs in "
          f"{time.time()-t0:.0f}s", flush=True)
    print(f"[hs] label balance: optimal-next={float(y_opt.mean()):.3f} "
          f"imitation={float(y_greedy.mean()):.3f} "
          f"in-optimal={float(y_orig.mean()):.3f}", flush=True)

    t0 = time.time()
    S = embed_pair(enc, tokenizer, states, device, args.max_length)
    A = embed_pair(enc, tokenizer, actions, device, args.max_length)
    print(f"[hs] embedded {len(states)} pairs in {time.time()-t0:.0f}s",
          flush=True)

    # ---- baselines on the same test tasks ----------------------------
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
        print(f"[hs] baseline {k:9s} success="
              f"{sum(1 for r in v if r[0])/max(len(v),1):.3f} "
              f"progress={sum(r[1] for r in v)/max(len(v),1):.3f}", flush=True)

    # ---- three heads, one architecture, one embedding ----------------
    heads = {}
    for name, y in (("optimal-next", y_opt),
                    ("imitation", y_greedy),
                    ("in-optimal", y_orig)):
        print(f"[hs] --- head[{name}] ---", flush=True)
        head, acc = train_head(S, A, y, device, args.epochs, args.lr)
        print(f"[hs] head[{name}] train accuracy: {acc:.3f}", flush=True)
        cache = {}
        succ, prog = [], []
        for (n, m, cons, sd, opt) in te_tasks:
            env = PlanningEnv(n, m, cons, sd)
            ok, pg, _ = beam_search(enc, tokenizer, head, device, env,
                                    args.budget, max_length=args.max_length,
                                    action_cache=cache)
            succ.append((ok, pg))
        s_ = sum(1 for r in succ if r[0]) / max(len(succ), 1)
        p_ = sum(r[1] for r in succ) / max(len(succ), 1)
        print(f"[hs] beam[head-{name:12s}] success={s_:.3f} progress={p_:.3f}",
              flush=True)
        heads[name] = {"train_acc": acc, "success": s_, "progress": p_}

    out = {"args": vars(args), "n_pairs": len(states),
           "label_balance": {"optimal-next": float(y_opt.mean()),
                             "imitation": float(y_greedy.mean()),
                             "in-optimal": float(y_orig.mean())},
           "baselines": {k: {"success": sum(1 for r in v if r[0]) / max(len(v), 1),
                             "progress": sum(r[1] for r in v) / max(len(v), 1)}
                         for k, v in base.items()},
           "heads": heads}
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"[hs] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
