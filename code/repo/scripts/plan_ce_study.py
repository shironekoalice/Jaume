"""Cross-entropy over actions instead of BCE over pairs.

The third design error, and probably the deepest one
-----------------------------------------------------
The head was trained with binary cross-entropy over individual
(state, action) pairs.  With BCE, "is this action good?" is scored
independently for every pair, so the model can reach a decent loss by
learning a prior over ACTIONS and ignoring the state entirely.  The
diagnosis showed exactly that: the head's top-3 actions were almost always
"place item2 at slot3" / "place item3 at slot3" regardless of the task, and
with a positive rate of 1.8% it collapsed further into "always negative"
(train accuracy 0.982 == the all-negative baseline).

What we actually want is NOT a per-pair score.  It is a choice: given this
state, which of the |A| legal actions should I take.  That is a ranking
problem WITHIN a state, so it should be trained as one -- softmax
cross-entropy over the legal actions of each state.  Under that loss the
state is not optional: with a fixed action prior the loss is flat at
log|A|, and the only way to reduce it is to use the state.

This script trains the same three label sets under that loss and compares.

  optimal-next  exactly one positive per state (the action that continues
                some optimal solution) -- softmax CE is well defined.
  imitation     exactly one positive per state (what greedy would do) --
                softmax CE is well defined.
  in-optimal    several positives per state -- only meaningful under BCE;
                reported for reference only.

Two encodings are supported:

  split  embed(state) and embed(action) separately, head sees concat.
         Cheap: actions are drawn from a fixed vocabulary of 30, so they
         are encoded once globally; only the states need encoding.
  joint  embed "<state> || <action>" as one string, so the encoder sees the
         pairing directly.  This removes the mean-pooling dilution that the
         split form suffers from.

Usage
-----
    python scripts/plan_ce_study.py --model <ckpt> --encoding split \
        --train-tasks 300 --test-tasks 100 --budget 30 --device npu \
        --out output/plan-ce-split.json
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

SEP = " || "


class PairHead(nn.Module):
    """score(state_emb, action_emb) -> scalar, for the split encoding."""

    def __init__(self, d_state, d_action, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_state + d_action, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s, a):
        return self.net(torch.cat([s, a], -1)).squeeze(-1)


class JointHead(nn.Module):
    """score(joint_emb) -> scalar, for the joint encoding."""

    def __init__(self, d, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def collect_groups(tasks, budget, rng, include_random=False):
    """Group (state, legal actions, labels) so the loss can rank within a state.

    Each group is a dict:
        state   the state text
        actions list of action texts (the legal actions)
        greedy  index of what the hand-written heuristic would pick
        optnext index of the action that continues some optimal solution
        inopt   set of indices that appear in some optimal full assignment

    The random path is only needed for the BCE reference labels (it is where
    "is this action in some optimal solution" gets its negatives from), so
    it is off by default -- under softmax CE every group carries exactly one
    positive and the random path adds nothing.
    """
    groups = []
    for (n, m, cons, sd, opt) in tasks:
        env = PlanningEnv(n, m, cons, sd)
        good = optimal_actions(env)
        best = opt[1]

        # ---- greedy path -------------------------------------------
        env.reset()
        for _ in range(budget):
            acts = env.legal_actions()
            if not acts:
                break
            g = choose_greedy(env, acts)
            st = state_text(env)
            groups.append({
                "state": st,
                "actions": [action_text(a) for a in acts],
                "greedy": acts.index(g),
                "optnext": None,
                "inopt": [i for i, a in enumerate(acts) if a in good],
            })
            env.step(g)

        # ---- optimal path ------------------------------------------
        env.reset()
        for it, sl in enumerate(best):
            acts = env.legal_actions()
            if not acts:
                break
            nxt = ("place", it, sl)
            if nxt not in acts:
                break
            st = state_text(env)
            groups.append({
                "state": st,
                "actions": [action_text(a) for a in acts],
                "greedy": None,
                "optnext": acts.index(nxt),
                "inopt": [i for i, a in enumerate(acts) if a in good],
            })
            env.step(nxt)

        if not include_random:
            continue

        # ---- random path (label coverage only) ----------------------
        env.reset()
        for _ in range(budget):
            acts = env.legal_actions()
            if not acts:
                break
            a = rng.choice(acts)
            groups.append({
                "state": state_text(env),
                "actions": [action_text(a)],
                "greedy": None,
                "optnext": None,
                "inopt": [0] if a in good else [],
            })
            env.step(a)
    return groups


def encode_groups(enc, tok, groups, device, encoding, max_length):
    """Encode every group into (state_emb, [action_emb], [joint_emb])."""
    if encoding == "split":
        # actions come from a small fixed vocabulary -> encode once
        vocab = sorted({a for g in groups for a in g["actions"]})
        av = embed_pair(enc, tok, vocab, device, max_length)
        aidx = {a: i for i, a in enumerate(vocab)}
        states = sorted({g["state"] for g in groups})
        sv = embed_pair(enc, tok, states, device, max_length)
        sidx = {s: i for i, s in enumerate(states)}
        for g in groups:
            g["s"] = sv[sidx[g["state"]]]
            g["a"] = [av[aidx[a]] for a in g["actions"]]
            g["j"] = None
    else:
        texts, spans = [], []
        for g in groups:
            start = len(texts)
            for a in g["actions"]:
                texts.append(g["state"] + SEP + a)
            spans.append((start, len(texts)))
        jv = embed_pair(enc, tok, texts, device, max_length)
        for g, (a, b) in zip(groups, spans):
            g["j"] = jv[a:b]
            g["s"] = None
            g["a"] = None
    return groups


def score_group(head, g, device, encoding):
    if encoding == "split":
        a = torch.stack(g["a"]).to(device)
        s = g["s"].to(device).expand(len(g["a"]), -1)
        return head(s, a)
    return head(g["j"].to(device))


def _flatten(groups, device, encoding):
    """Flatten the groups into one (pair, group_id) table.

    Every group has the same number of legal actions: with n_items items and
    n_slots slots the count is n_items * n_slots - placed + placed, i.e.
    constant.  The assert below turns that assumption into a checked fact.
    """
    lens = {len(g["actions"]) for g in groups}
    assert len(lens) == 1, f"ragged groups: {sorted(lens)[:5]}"
    Amax = lens.pop()
    G = len(groups)

    if encoding == "split":
        S = torch.stack([g["s"] for g in groups]).to(device)        # [G, d]
        A = torch.stack([torch.stack(g["a"]) for g in groups]).to(device)
        s = S.unsqueeze(1).expand(G, Amax, -1).reshape(G * Amax, -1)
        a = A.reshape(G * Amax, -1)
    else:
        A = torch.stack([torch.cat(g["j"]) for g in groups]).to(device)
        s = None
        a = A.reshape(G * Amax, -1)

    gid = torch.arange(G, device=device).repeat_interleave(Amax)
    tgt_pos = torch.tensor([g["_tgt"] for g in groups], device=device)
    tgt = torch.arange(G, device=device) * Amax + tgt_pos
    return s, a, gid, tgt, G


def train_ce(head, groups, device, epochs, lr, label_key):
    """Softmax cross-entropy over the legal actions of each state.

    One forward pass over all pairs per epoch, then a segmented log-sum-exp
    so every state's actions are normalised against each other.
    """
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    trainable = [g for g in groups if g.get(label_key) is not None
                 and len(g["actions"]) > 1]
    for g in trainable:
        g["_tgt"] = g[label_key]
    s, a, gid, tgt, G = _flatten(trainable, device, CUR_ENCODING)
    print(f"      trainable groups={G} pairs={len(gid)}", flush=True)

    for ep in range(epochs):
        head.train()
        if CUR_ENCODING == "split":
            logits = head(s, a)
        else:
            logits = head(a)
        seg_max = torch.full((G,), -float("inf"), device=logits.device)
        seg_max = seg_max.scatter_reduce(0, gid, logits, reduce="amax")
        ex = torch.exp(logits - seg_max[gid])
        seg_sum = torch.zeros(G, device=logits.device).scatter_add(0, gid, ex)
        logZ = torch.log(seg_sum) + seg_max
        loss = -(logits[tgt] - logZ[gid[tgt]]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (ep + 1) % 10 == 0:
            print(f"      epoch {ep+1}/{epochs} loss={float(loss.detach()):.4f}",
                  flush=True)

    head.eval()
    hit = 0
    with torch.no_grad():
        for g in trainable:
            logits = score_group(head, g, device, CUR_ENCODING)
            hit += 1 if int(logits.argmax()) == g[label_key] else 0
    return hit / max(len(trainable), 1)


def _score(raw, mode):
    """Turn a per-state vector of head outputs into path scores."""
    if mode == "logit":
        return raw.tolist()
    if mode == "logprob":
        return F.log_softmax(raw, dim=-1).tolist()
    raise ValueError(f"unknown score_mode: {mode}")


@torch.no_grad()
def beam(enc, tok, head, device, env, budget, width, max_length, encoding,
         cache=None, score_mode="logprob"):
    """Beam search driven by the head.

    score_mode matters.  A head trained with softmax cross-entropy is
    calibrated WITHIN a state: it says which of the state's actions is best,
    not how good the state is.  Summing raw logits across a path therefore
    mixes two different quantities and lets a constant per-action offset
    dominate the ranking -- which showed up as progress 0.000.  Summing
    log-softmax instead sums path log-probabilities, i.e. the probability of
    the path under the model's per-step policy, which is the quantity the
    loss actually trained.

    width=1 makes this a greedy execution of the learned policy.
    """
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
            if encoding == "split":
                ctx = state_text(e)
                if ctx not in cache:
                    cache[ctx] = embed_pair(enc, tok, [ctx], device, max_length)[0]
                s = cache[ctx].to(device)
                texts = [action_text(a) for a in acts]
                miss = [t for t in dict.fromkeys(texts) if t not in cache]
                if miss:
                    em = embed_pair(enc, tok, miss, device, max_length)
                    for t, v in zip(miss, em):
                        cache[t] = v
                a = torch.stack([cache[t] for t in texts]).to(device)
                raw = head(s.expand(len(acts), -1), a).float().cpu()
                sc = _score(raw, score_mode)
            else:
                ctx = state_text(e)
                texts = [ctx + SEP + action_text(a) for a in acts]
                em = embed_pair(enc, tok, texts, device, max_length)
                raw = head(em.to(device)).float().cpu()
                sc = _score(raw, score_mode)
            for a_, v in zip(acts, sc):
                e2 = PlanningEnv(env.n_items, env.n_slots, env.constraints,
                                 env.seed)
                e2.assignment = list(assign)
                e2.step(a_)
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


CUR_ENCODING = "split"


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
    ap.add_argument("--encoding", choices=["split", "joint"], default="split")
    ap.add_argument("--score-mode", choices=["logit", "logprob"],
                    default="logprob")
    ap.add_argument("--widths", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    global CUR_ENCODING
    CUR_ENCODING = args.encoding
    device = resolve_device(args.device)
    print(f"[ce] device={device} ({describe(device)}) "
          f"encoding={args.encoding}", flush=True)

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
    print(f"[ce] encoder frozen; d={cfg.embedding_dim}", flush=True)

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
    print(f"[ce] train={len(tr_tasks)} test={len(te_tasks)}", flush=True)

    t0 = time.time()
    groups = collect_groups(tr_tasks, args.budget, rng)
    print(f"[ce] {len(groups)} groups in {time.time()-t0:.0f}s", flush=True)

    t0 = time.time()
    groups = encode_groups(enc, tokenizer, groups, device, args.encoding,
                           args.max_length)
    print(f"[ce] encoded in {time.time()-t0:.0f}s", flush=True)

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
        print(f"[ce] baseline {k:9s} success="
              f"{sum(1 for r in v if r[0])/max(len(v),1):.3f} "
              f"progress={sum(r[1] for r in v)/max(len(v),1):.3f}", flush=True)

    d = cfg.embedding_dim
    heads = {}
    for label_key in ("greedy", "optnext"):
        name = "imitation" if label_key == "greedy" else "optimal-next"
        if args.encoding == "split":
            head = PairHead(d, d).to(device)
        else:
            head = JointHead(d).to(device)
        print(f"[ce] --- head[{name}] ({args.encoding}) ---", flush=True)
        acc = train_ce(head, groups, device, args.epochs, args.lr, label_key)
        print(f"[ce] head[{name}] train top-1 accuracy: {acc:.3f}", flush=True)
        for w in args.widths:
            for mode in (args.score_mode, "logit"):
                cache = {}
                succ, prog = [], []
                for (n, m, cons, sd, opt) in te_tasks:
                    env = PlanningEnv(n, m, cons, sd)
                    ok, pg, _ = beam(enc, tokenizer, head, device, env,
                                     args.budget, w, args.max_length,
                                     args.encoding, cache, score_mode=mode)
                    succ.append((ok, pg))
                s_ = sum(1 for r in succ if r[0]) / max(len(succ), 1)
                p_ = sum(r[1] for r in succ) / max(len(succ), 1)
                print(f"[ce] beam[{name:12s} w={w} {mode:7s}] "
                      f"success={s_:.3f} progress={p_:.3f}", flush=True)
                heads[f"{name}|w{w}|{mode}"] = {
                    "train_top1_acc": acc, "width": w, "score_mode": mode,
                    "success": s_, "progress": p_}

    out = {"args": vars(args), "n_groups": len(groups), "heads": heads,
           "baselines": {k: {"success": sum(1 for r in v if r[0]) / max(len(v), 1),
                             "progress": sum(r[1] for r in v) / max(len(v), 1)}
                         for k, v in base.items()}}
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"[ce] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
