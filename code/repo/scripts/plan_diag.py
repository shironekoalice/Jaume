"""Diagnose why the learned beam search scores 0.000.

The P0 gate came back as beam-model 0.000 vs greedy 0.810.  A gap that large
is either (a) the beam search mechanism is broken, or (b) the learned score
carries no usable signal.  These are very different claims, so they must be
separated before anything else is built.

The separation: drive the SAME beam search with three different score
functions and compare.

  oracle    the exact optimal action set, computed by brute force.  This is
            the mechanism's ceiling: if the beam cannot reach ~1.0 with a
            perfect score, the beam itself is broken.
  greedy    the hand-written immediate-progress score that choose_greedy
            uses.  This is the mechanism's floor: a sane beam should reach
            greedy's success rate with this score.
  head      the trained ScoreHead.  This is the actual method.

If oracle ~ 1.0 and greedy ~ 0.81 but head ~ 0.0, the mechanism is fine and
the learned score is the problem -- and the remaining question is whether the
training signal is learnable at all.

Additional checks:
  D1  score spread     -- std of the head's scores across actions at a state.
                           If this is ~0 the head cannot rank anything.
  D2  top-1 quality    -- P(head top-1 in optimal set) vs P(greedy in optimal set).
  D4  first-step trace -- what the beam picks first, what greedy picks.
  D5  label mix        -- where the training labels actually come from.

Usage
-----
    python scripts/plan_diag.py --model <ckpt> --test-tasks 50 \
        --n-constraints 8 --device npu
"""

import argparse
import json
import os
import random
import sys
import time
from collections import Counter

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
from plan_train import collect, optimal_actions


# ------------------------------------------------------------------ scores
def greedy_score(env, action):
    """The hand-written immediate-progress score used by choose_greedy."""
    saved = list(env.assignment)
    env.step(action)
    sc = env.progress() + (1.0 if env.goal_reached() else 0.0)
    env.assignment = saved
    return sc


def oracle_score(env, action, good):
    """1 if the action appears in some optimal full assignment."""
    return 1.0 if action in good else 0.0


@torch.no_grad()
def beam_by_score(env, budget, width, score_fn, good=None):
    """Beam search over an arbitrary (env, action) -> float score.

    Same shape as the model's beam search: expand every beam over the legal
    actions, sum the scores along the path, keep the best `width`.
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
            for a in acts:
                e2 = PlanningEnv(env.n_items, env.n_slots, env.constraints,
                                 env.seed)
                e2.assignment = list(assign)
                e2.step(a)
                if good is None:
                    sc = score_fn(e, a)
                else:
                    sc = oracle_score(e, a, good)
                cand.append((list(e2.assignment), score + sc))
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


@torch.no_grad()
def beam_by_head(enc, tok, head, device, env, budget, width, max_length,
                 cache):
    """The model's beam search, same as plan_train.beam_search."""
    env.reset()
    beams = [(list(env.assignment), 0.0)]
    for _ in range(budget):
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
        ctxs = [state_text(e) for (e, acts, _a, _s) in live]
        s_embs = embed_pair(enc, tok, ctxs, device, max_length,
                            bs=len(ctxs))
        cand = []
        for bi, (e, acts, assign, score) in enumerate(live):
            texts = [action_text(a) for a in acts]
            missing = [t for t in dict.fromkeys(texts) if t not in cache]
            if missing:
                em = embed_pair(enc, tok, missing, device, max_length)
                for t, v in zip(missing, em):
                    cache[t] = v
            a_emb = torch.stack([cache[t] for t in texts])
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
    for assign, _ in beams:
        e = PlanningEnv(env.n_items, env.n_slots, env.constraints, env.seed)
        e.assignment = list(assign)
        if e.goal_reached():
            return True, e.progress(), budget
    return False, 0.0, budget


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-items", type=int, default=6)
    ap.add_argument("--n-slots", type=int, default=4)
    ap.add_argument("--n-constraints", type=int, default=8)
    ap.add_argument("--train-tasks", type=int, default=300)
    ap.add_argument("--test-tasks", type=int, default=50)
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
    print(f"[diag] device={device} ({describe(device)})", flush=True)

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
    print(f"[diag] encoder frozen; d={cfg.embedding_dim}", flush=True)

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
    print(f"[diag] train={len(tr_tasks)} test={len(te_tasks)}", flush=True)

    # ---- D5: label composition -------------------------------------
    print("[diag] collecting labelled pairs ...", flush=True)
    t0 = time.time()
    rows, eps = collect(enc, tokenizer, device, tr_tasks, args.n_constraints,
                        args.budget, rng)
    print(f"[diag] {len(rows)} pairs in {time.time()-t0:.0f}s", flush=True)
    n_pos = sum(r[2] for r in rows)
    print(f"[diag] D5 label balance: positive={n_pos/len(rows):.3f} "
          f"n={len(rows)}", flush=True)

    # ---- train the head --------------------------------------------
    t0 = time.time()
    s_txt = [r[0] for r in rows]
    a_txt = [r[1] for r in rows]
    y = torch.tensor([r[2] for r in rows], dtype=torch.float32)
    S = embed_pair(enc, tokenizer, s_txt, device, args.max_length)
    A = embed_pair(enc, tokenizer, a_txt, device, args.max_length)
    print(f"[diag] embedded in {time.time()-t0:.0f}s", flush=True)

    d = S.shape[1]
    head = ScoreModel(d, d).to(device)
    opt_ = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    Sd, Ad, yd = S.to(device), A.to(device), y.to(device)
    for ep in range(args.epochs):
        head.train()
        perm = torch.randperm(len(Sd), device=device)
        tot, n = 0.0, 0
        for i in range(0, len(perm), 128):
            idx = perm[i:i + 128]
            logit = head(Sd[idx], Ad[idx])
            loss = F.binary_cross_entropy_with_logits(logit, yd[idx])
            opt_.zero_grad(set_to_none=True)
            loss.backward()
            opt_.step()
            tot += float(loss.detach())
            n += 1
        if (ep + 1) % 10 == 0:
            print(f"[diag] epoch {ep+1}/{args.epochs} loss={tot/max(n,1):.4f}",
                  flush=True)
    head.eval()
    with torch.no_grad():
        acc = float(((head(Sd, Ad) > 0).float() == yd).float().mean())
    print(f"[diag] head train accuracy: {acc:.3f}", flush=True)

    # ---- D1/D2/D4: score spread, top-1 quality, first step ----------
    spreads, hit_head, hit_greedy, base_pos, first_trace = [], [], [], [], []
    for (n, m, cons, sd, opt) in te_tasks[:20]:
        env = PlanningEnv(n, m, cons, sd)
        good = optimal_actions(env)
        acts = env.legal_actions()
        ctx = state_text(env)
        s = embed_pair(enc, tokenizer, [ctx], device, args.max_length)
        at = [action_text(a) for a in acts]
        ae = embed_pair(enc, tokenizer, at, device, args.max_length)
        sc = head(s.to(device).expand(len(acts), -1),
                  ae.to(device)).float().cpu().tolist()
        spreads.append(float(torch.tensor(sc).std()))
        base_pos.append(sum(1 for v in sc if v > 0) / len(sc))
        best = max(range(len(acts)), key=lambda i: sc[i])
        hit_head.append(1 if acts[best] in good else 0)
        g = choose_greedy(env, acts)
        hit_greedy.append(1 if g in good else 0)
        if len(first_trace) < 5:
            first_trace.append({
                "head_top1": at[best], "head_top1_in_optimal": acts[best] in good,
                "greedy": action_text(g), "greedy_in_optimal": g in good,
                "head_top3": [at[i] for i in
                              sorted(range(len(acts)), key=lambda i: -sc[i])[:3]]})
    print(f"[diag] D1 score spread: mean std={sum(spreads)/len(spreads):.4f} "
          f"(fraction of actions scored > 0: "
          f"{sum(base_pos)/len(base_pos):.3f})", flush=True)
    print(f"[diag] D2 top-1 in optimal set: head={sum(hit_head)/len(hit_head):.3f} "
          f"greedy={sum(hit_greedy)/len(hit_greedy):.3f}", flush=True)
    print("[diag] D4 first-step examples:", flush=True)
    for t in first_trace:
        print(f"[diag]   {t}", flush=True)

    # ---- D3: mechanism control --------------------------------------
    res = {"oracle": [], "greedy-score": [], "head": []}
    for (n, m, cons, sd, opt) in te_tasks:
        env = PlanningEnv(n, m, cons, sd)
        good = optimal_actions(env)
        ok, prog, _ = beam_by_score(env, args.budget, args.width,
                                     None, good=good)
        res["oracle"].append((ok, prog))

        env = PlanningEnv(n, m, cons, sd)
        ok, prog, _ = beam_by_score(env, args.budget, args.width, greedy_score)
        res["greedy-score"].append((ok, prog))

        env = PlanningEnv(n, m, cons, sd)
        ok, prog, _ = beam_by_head(enc, tokenizer, head, device, env,
                                   args.budget, args.width, args.max_length, {})
        res["head"].append((ok, prog))

    summary = {}
    for k, v in res.items():
        s_ = sum(1 for r in v if r[0]) / max(len(v), 1)
        p_ = sum(r[1] for r in v) / max(len(v), 1)
        summary[k] = {"success": s_, "progress": p_}
        print(f"[diag] D3 beam[{k:13s}] success={s_:.3f} progress={p_:.3f}",
              flush=True)

    out = {"args": vars(args), "label_balance": n_pos / len(rows),
           "head_train_acc": acc,
           "score_spread_mean_std": sum(spreads) / len(spreads),
           "frac_actions_positive": sum(base_pos) / len(base_pos),
           "top1_in_optimal": {"head": sum(hit_head) / len(hit_head),
                               "greedy": sum(hit_greedy) / len(hit_greedy)},
           "first_step_examples": first_trace,
           "beam_by_score": summary}
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"[diag] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
