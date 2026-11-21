"""Stage 2: the loop on a task that NEEDS different rounds.

Why the task has to change
--------------------------
The 2026-11-07 gate experiment closed the optimisation escape route: given
a free per-round reweighting, the model trains it to [1.04, 1.01, 1.00] and
the round-distance decay stays point-for-point the same.  On next-token
prediction over generic text, 'same update every round' IS the optimal
policy, because nothing in the task asks different rounds to do different
work.  The contraction is structural, not an optimisation failure.

So the first thing to change is the task.  This script trains the looped LM
on a task where round t and round t+1 have different, independently
verifiable semantics:

    PlanningEnv (from the planning direction, already in the repo):
      items  = {0..n-1}
      slots  = {0..m-1}
      state  = assignment[item] -> slot | -1
      goal   = before / not_same / same / fixed constraints, all satisfiable

The task has three properties that make it a good loop test:

  1. ROUNDS HAVE DIFFERENT SEMANTICS.  Round 1 has to READ the constraints
     and decide which item/slot pair is most constrained; round 2 has to
     COMMIT a placement; round 3 has to CHECK the remaining constraints.
     These are different operations on different state, not another copy of
     the same residual update.

  2. THE INTERMEDIATE RESULT IS VERIFIABLE.  Every constraint is either
     satisfied or not, so the probe question -- "does round t's output
     differ from round t+1's in the way the task predicts?" -- has a ground
     truth to compare against, unlike generic text.

  3. IT IS ALREADY HERE.  PlanningEnv, the generator, the exact optimum,
     and the greedy heuristic all exist in scripts/planning_env.py and
     plan_search.py.  No new environment code.

What is trained
---------------
The LM is trained on linearised planning trajectories:

    "task: <constraints> state: <assignment> action: place item3 at slot2"

with next-token prediction over the action tokens only (the state and
constraint prefix is conditioning, not target).  This is decision-style
training on an LM -- the model learns a policy, and the loop gets a task in
which intermediate rounds can carry distinct work.

The probe question afterwards is the same Q3 as before, but now with a
prediction attached: IF the loop is going to ever be useful, the
round-distance decay should be slower here than on generic text, and the
gate should learn a non-uniform assignment.
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

from planning_env import PlanningEnv, generate_task
from plan_search import state_text, action_text, choose_greedy


def render_task(n, m, cons):
    """Linearised constraint description."""
    parts = []
    for c in cons:
        kind = c[0]
        if kind == "before":
            parts.append(f"item{c[1]} before item{c[2]}")
        elif kind == "not_same":
            parts.append(f"item{c[1]} not same as item{c[2]}")
        elif kind == "same":
            parts.append(f"item{c[1]} same as item{c[2]}")
        elif kind == "fixed":
            parts.append(f"item{c[1]} fixed at slot{c[2]}")
    return "task: items=" + str(n) + " slots=" + str(m) + \
           " constraints: " + "; ".join(parts)


def render_state(env):
    a = env.assignment
    return "state: " + ",".join(
        f"item{i}=slot{s}" if s >= 0 else f"item{i}=none"
        for i, s in enumerate(a))


def _raw_constraints(rng, n_items, n_slots, n_constraints):
    """Generate constraints WITHOUT the expensive feasibility search.

    generate_task() retries up to 200 times and calls env.optimum() (a brute
    force over m^n assignments) on every attempt, which is needed when the
    task is going to be SCORED against an exact optimum.  Language-model
    training does not need that: the corpus is the greedy trajectory, and the
    greedy heuristic produces a legal trajectory for any constraint set.  So
    the feasibility search is skipped here -- 40000 tasks took 20+ minutes
    with it and well under a minute without.
    """
    cons = []
    for _ in range(n_constraints):
        kind = rng.choice(["before", "not_same", "same", "fixed"])
        x, y = rng.sample(range(n_items), 2) if n_items >= 2 else (0, 0)
        if kind == "fixed":
            cons.append(("fixed", x, rng.randrange(n_slots)))
        else:
            cons.append((kind, x, y))
    return cons


def build_trajectory_corpus(n_tasks, n_items, n_slots, n_constraints,
                            budget, seed, greedy_only=True):
    """One document per greedy step: prefix + action, target = action text."""
    rng = random.Random(seed)
    docs = []
    for t in range(n_tasks):
        sd = seed * 100003 + t
        cons = _raw_constraints(random.Random(sd), n_items, n_slots,
                                n_constraints)
        a, b = n_items, n_slots
        env = PlanningEnv(a, b, cons, sd)
        env.reset()
        task_str = render_task(a, b, cons)
        for step in range(budget):
            acts = env.legal_actions()
            if not acts:
                break
            act = choose_greedy(env, acts)
            act_str = action_text(act)
            # the document: everything before the action is conditioning;
            # the action itself is the target.  We store the prefix and the
            # action separately so the collator can mask the prefix.
            docs.append({
                "prefix": task_str + " " + render_state(env) + " action:",
                "action": " " + act_str,
                "task_seed": sd, "step": step,
            })
            env.step(act)
    return docs


def encode_docs(docs, tokenizer, max_length):
    """Tokenise prefix + action; return input_ids and the index where the
    action starts (targets are only the action tokens)."""
    out = []
    for d in docs:
        pre = tokenizer(d["prefix"], add_special_tokens=False)["input_ids"]
        act = tokenizer(d["action"], add_special_tokens=False)["input_ids"]
        # keep the LAST (max_length - len(act)) tokens of the prefix so the
        # action always fits inside the window
        keep = max_length - len(act)
        if keep <= 0:
            continue
        ids = pre[-keep:] + act
        out.append({"ids": ids, "act_start": len(ids) - len(act)})
    return out


def ntp_loss_actiononly(logits, ids, act_start, lengths):
    """Cross-entropy on the action tokens only, fully vectorised.

    Position i predicts token i+1, so the action tokens occupy target
    positions [act_start-1, length-2].  The mask is built with two
    comparisons on an arange, which keeps the whole thing on the accelerator:
    the previous per-row Python loop plus boolean gather triggered Ascend's
    aclnnNonzeroV2 fallback and stalled the run at step 0.
    """
    B, L, V = logits.shape
    tgt = ids[:, 1:]                       # [B, L-1]
    lg = logits[:, :-1, :]                 # [B, L-1, V]
    pos = torch.arange(L - 1, device=ids.device).unsqueeze(0)   # [1, L-1]
    lo = (act_start - 1).clamp(min=0).unsqueeze(1)
    hi = (lengths - 1).unsqueeze(1)
    mask = (pos >= lo) & (pos < hi)        # [B, L-1]
    tg = tgt.reshape(-1)
    maskf = mask.reshape(-1)
    # index_select with a mask is a simple gather on a flat index, which the
    # Ascend backend handles; F.cross_entropy then works on a dense [N, V].
    idx = maskf.nonzero(as_tuple=True)[0] if maskf.any() else None
    if idx is None or idx.numel() == 0:
        return logits.sum() * 0.0
    sel = lg.reshape(-1, V).index_select(0, idx)
    return F.cross_entropy(sel, tg.index_select(0, idx))


@torch.no_grad()
def evaluate(model, ids_all, len_all, act_all, val_idx, max_batches=4,
             batch_size=8):
    # no_grad is not optional: without it every eval batch builds a full
    # 369M-parameter graph that is never freed and the run eventually hangs
    # (observed as AICore ~0% with the step counter frozen).
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, min(len(val_idx), max_batches * batch_size), batch_size):
        sel = val_idx[i:i + batch_size].to(ids_all.device)
        ids = ids_all.index_select(0, sel)
        logits, aux = model(ids)
        loss = ntp_loss_actiononly(logits, ids,
                                   act_all.index_select(0, sel),
                                   len_all.index_select(0, sel))
        tot += float(loss) * len(sel)
        n += len(sel)
    model.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--init-from", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-tasks", type=int, default=20000)
    ap.add_argument("--n-items", type=int, default=6)
    ap.add_argument("--n-slots", type=int, default=4)
    ap.add_argument("--n-constraints", type=int, default=8)
    ap.add_argument("--budget", type=int, default=12)
    ap.add_argument("--num-loops", type=int, default=1)
    ap.add_argument("--loop-gate", action="store_true")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--val-docs", type=int, default=512)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)
    print(f"[pl] device={device} ({describe(device)})", flush=True)

    from jaume.modeling.config import JaumeConfig
    from jaume.modeling.loop_lm import LoopLM
    from jaume.modeling.checkpoint import load_state_dict
    from jaume.data.dataset import load_tokenizer

    cfg = JaumeConfig.load(args.config)
    tokdir = args.tokenizer or os.path.join(REPO, "jaume_tokenizer")
    tokenizer = load_tokenizer(tokdir)
    tokenizer.add_tokens(["[MASK]"])
    cfg.vocab_size = len(tokenizer)
    cfg.num_loops = args.num_loops
    cfg.loop_gate = args.loop_gate
    cfg.causal = True
    cfg.model_type = "jaume-loop"

    model = LoopLM(cfg).to(device)
    if args.init_from:
        sd = load_state_dict(os.path.join(args.init_from, "pytorch_model.bin"))
        tgt = model.backbone.state_dict()
        src = {k[len("backbone."):]: v for k, v in sd.items()
               if k.startswith("backbone.")}
        loaded = [k for k, v in src.items()
                  if k in tgt and tgt[k].shape == v.shape]
        for k in loaded:
            tgt[k] = src[k]
        model.backbone.load_state_dict(tgt)
        if "lm_head.weight" in sd:
            model.lm_head.weight.data = sd["lm_head.weight"].to(device)
        print(f"[pl] warm start {len(loaded)} tensors", flush=True)
    print(f"[pl] params={model.num_parameters()/1e6:.1f}M "
          f"depth={model.backbone.effective_depth}", flush=True)

    seed_all(device, args.seed)
    print(f"[pl] generating {args.n_tasks} tasks ...", flush=True)
    t0 = time.time()
    docs = build_trajectory_corpus(args.n_tasks, args.n_items, args.n_slots,
                                   args.n_constraints, args.budget,
                                   args.seed)
    print(f"[pl] {len(docs)} trajectory steps in {time.time()-t0:.0f}s",
          flush=True)
    encoded = encode_docs(docs, tokenizer, args.max_length)
    print(f"[pl] {len(encoded)} encoded docs", flush=True)

    # Pre-pad ONCE into dense tensors.  Building these per step from Python
    # lists is what made the first attempt stall at step 0 with AICore 0%:
    # the NPU sat idle while the host thread converted 8 x 480k nested lists
    # and the boolean-index gather below fell back to a slow path.
    L = max(len(d["ids"]) for d in encoded)
    n_enc = len(encoded)
    ids_all = torch.full((n_enc, L), cfg.pad_token_id, dtype=torch.long)
    len_all = torch.zeros(n_enc, dtype=torch.long)
    act_all = torch.zeros(n_enc, dtype=torch.long)
    for k, d in enumerate(encoded):
        n = len(d["ids"])
        ids_all[k, :n] = torch.tensor(d["ids"], dtype=torch.long)
        len_all[k] = n
        act_all[k] = d["act_start"]
    print(f"[pl] dense tensor {tuple(ids_all.shape)} "
          f"({ids_all.numel()*ids_all.element_size()/1e6:.0f} MB)", flush=True)
    n_val = args.val_docs
    ids_all_dev = ids_all.to(device)
    len_all_dev = len_all.to(device)
    act_all_dev = act_all.to(device)
    train_idx = torch.arange(n_val, n_enc, device=device)
    val_idx = torch.arange(0, n_val, device=device)
    print(f"[pl] train={len(train_idx)} val={len(val_idx)}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1,
                            betas=(0.9, 0.95))
    steps_per_epoch = max(1, len(train_idx) // (args.batch_size * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs

    def lr_at(step):
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        p = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))

    step = 0
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = train_idx[torch.randperm(len(train_idx), device=device)]
        opt.zero_grad(set_to_none=True)
        for bi, i in enumerate(range(0, len(train_idx) - args.batch_size + 1,
                                      args.batch_size)):
            sel = perm[i:i + args.batch_size].to(device)
            ids = ids_all_dev.index_select(0, sel)
            act_start = act_all_dev.index_select(0, sel)
            ln = len_all_dev.index_select(0, sel)
            logits, aux = model(ids)
            loss = ntp_loss_actiononly(logits, ids, act_start, ln)
            if aux is not None:
                loss = loss + cfg.load_balance_weight * aux
            (loss / args.grad_accum).backward()
            if (bi + 1) % args.grad_accum == 0:
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % args.log_every == 0:
                    print(f"[pl] epoch {ep} step {step}/{total_steps} "
                          f"loss {float(loss.detach()):.4f}", flush=True)
                if args.eval_every and step % args.eval_every == 0:
                    vl = evaluate(model, ids_all_dev, len_all_dev, act_all_dev, val_idx)
                    gate_str = ""
                    gp = getattr(model.backbone, "loop_gate_param", None)
                    if gp is not None:
                        gate_str = " gate[" + ",".join(
                            f"{float(v):.2f}" for v in gp.detach()) + "]"
                    print(f"[pl] epoch {ep} step {step} VAL loss {vl:.4f}"
                          f"{gate_str}", flush=True)

    vl = evaluate(model, ids_all_dev, len_all_dev, act_all_dev, val_idx)
    gate_str = ""
    gp = getattr(model.backbone, "loop_gate_param", None)
    if gp is not None:
        gate_str = " gate[" + ",".join(f"{float(v):.2f}" for v in gp.detach()) + "]"
    print(f"[pl] FINAL VAL loss {vl:.4f}{gate_str}", flush=True)

    os.makedirs(args.output, exist_ok=True)
    cfg.save(os.path.join(args.output, "config.json"))
    sd_out = {"backbone." + k: v for k, v in model.backbone.state_dict().items()}
    sd_out["lm_head.weight"] = model.lm_head.weight.detach().cpu()
    torch.save(sd_out, os.path.join(args.output, "pytorch_model.bin"))
    try:
        tokenizer.save_pretrained(os.path.join(args.output, "tokenizer"))
    except Exception:
        pass
    print(f"[pl] saved -> {args.output}", flush=True)
    print(f"[pl] done in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    import math
    sys.exit(main())
