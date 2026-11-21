"""Is the state-conditioned score INSIDE the 350M base model, or not?

The question
------------
With the encoder frozen, a shallow head trained with softmax cross-entropy
reaches only ~0.21 top-1 accuracy on "which action should I take here",
while the hand-written greedy heuristic is at 1.000 (greedy always picks an
action that lies in some optimal solution).  So the failure is not the label
and not the loss: a frozen, mean-pooled representation of the state does not
carry enough structure for a shallow head to rank the actions.

That leaves two very different worlds:

  (a) the capability IS in the base model but a frozen linear-ish read-out
      cannot extract it -> fine-tuning should fix it, and the whole
      "learn to guide search with a small encoder" programme is viable;
  (b) the capability is NOT in the base model -> no amount of training on
      this task will produce it, and the base model has to change.

This script runs (a) directly: it unfreezes the backbone and trains the
encoder and the scoring head end to end, with the same segment softmax
cross-entropy over each state's legal actions.

Design
------
* joint encoding ("<state> || <action>") -- the encoder has to see the
  pairing, otherwise mean pooling cannot represent it.
* --unfreeze N: 0 keeps everything frozen (the control), -1 unfreezes the
  whole encoder, N>0 unfreezes the top N transformer layers.
* gradient accumulation over micro-batches so the activation memory stays
  bounded.
* the label is "imitation" (what the greedy heuristic would do) by default,
  because it is the sharpest reference: a model that cannot learn to imitate
  the heuristic cannot be expected to beat it.
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
from plan_search import (state_text, action_text, embed_pair, run_episode,
                         choose_greedy, choose_random)
from plan_ce_study import collect_groups, SEP


class JointHead(nn.Module):
    def __init__(self, d, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def set_unfreeze(enc, n):
    """0 = fully frozen, -1 = fully trainable, N>0 = top N layers trainable."""
    for p in enc.parameters():
        p.requires_grad_(False)
    layers = getattr(enc.backbone, "layers", None)
    if n == 0:
        return "frozen"
    if n < 0:
        for p in enc.parameters():
            p.requires_grad_(True)
        return "full"
    if layers is None:
        for p in enc.parameters():
            p.requires_grad_(True)
        return "full(no-layers-attr)"
    for p in enc.backbone.embed_tokens.parameters():
        p.requires_grad_(True)
    for p in enc.backbone.norm.parameters() if hasattr(enc.backbone, "norm") \
            else []:
        p.requires_grad_(True)
    for layer in list(layers)[-n:]:
        for p in layer.parameters():
            p.requires_grad_(True)
    return f"top{n}"


def embed_train(enc, tok, texts, device, max_length, bs):
    """Differentiable joint embedding of a list of texts."""
    outs, auxes = [], []
    for i in range(0, len(texts), bs):
        batch = tok(texts[i:i + bs], padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt").to(device)
        h, aux = enc.backbone(batch["input_ids"], batch["attention_mask"])
        m = batch["attention_mask"].unsqueeze(-1).to(h.dtype)
        v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        outs.append(F.normalize(v, dim=-1))
        if aux is not None:
            auxes.append(aux)
    aux = torch.stack([a.reshape(()) for a in auxes]).mean() if auxes else None
    return torch.cat(outs, 0), aux


@torch.no_grad()
def beam_ft(enc, tok, head, device, env, budget, width, max_length, bs=96):
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
            em, _ = embed_train(enc, tok, texts, device, max_length, bs)
            raw = head(em.to(device)).float().cpu()
            sc = F.log_softmax(raw, dim=-1).tolist()
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-items", type=int, default=6)
    ap.add_argument("--n-slots", type=int, default=4)
    ap.add_argument("--n-constraints", type=int, default=8)
    ap.add_argument("--train-tasks", type=int, default=300)
    ap.add_argument("--test-tasks", type=int, default=100)
    ap.add_argument("--budget", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--max-length", type=int, default=96)
    ap.add_argument("--width", type=int, default=3)
    ap.add_argument("--micro-groups", type=int, default=32)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--label", choices=["imitation", "optimal-next"],
                    default="imitation")
    ap.add_argument("--unfreeze", type=int, default=-1,
                    help="0 frozen, -1 full, N>0 top N layers")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--save", default=None,
                    help="directory to dump the fine-tuned weights, so a "
                         "collapse can be inspected afterwards")
    args = ap.parse_args()

    device = resolve_device(args.device)
    print(f"[ft] device={device} ({describe(device)})", flush=True)

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
    mode = set_unfreeze(enc, args.unfreeze)
    n_train = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    print(f"[ft] backbone mode={mode} trainable={n_train/1e6:.1f}M", flush=True)

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
    groups = collect_groups(tr_tasks, args.budget, rng)
    key = "greedy" if args.label == "imitation" else "optnext"
    trainable = [g for g in groups if g.get(key) is not None
                 and len(g["actions"]) > 1]
    for g in trainable:
        g["_tgt"] = g[key]
    print(f"[ft] train={len(tr_tasks)} test={len(te_tasks)} "
          f"groups={len(trainable)} label={args.label}", flush=True)

    head = JointHead(cfg.embedding_dim).to(device)
    bb_params = [p for p in enc.parameters() if p.requires_grad]
    groups_bb = [{"params": bb_params, "lr": args.lr_backbone}] if bb_params else []
    opt = torch.optim.AdamW(
        [{"params": list(head.parameters()), "lr": args.lr_head}]
        + groups_bb, weight_decay=1e-4)

    Amax = len(trainable[0]["actions"])
    G = len(trainable)

    for ep in range(args.epochs):
        enc.train()
        head.train()
        perm = torch.randperm(G, device=device)
        tot, n = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for step, i in enumerate(range(0, G, args.micro_groups)):
            sel = perm[i:i + args.micro_groups]
            gsel = [trainable[int(j)] for j in sel.tolist()]
            texts, tgt_pos = [], []
            for g in gsel:
                for a in g["actions"]:
                    texts.append(g["state"] + SEP + a)
                tgt_pos.append(g["_tgt"])
            em, aux = embed_train(enc, tokenizer, texts, device,
                                  args.max_length, args.micro_groups * Amax)
            logits = head(em).float()
            lg = logits.view(len(gsel), Amax)
            loss = F.cross_entropy(lg, torch.tensor(tgt_pos, device=device))
            if aux is not None:
                loss = loss + 0.01 * aux
            (loss / args.accum).backward()
            if (step + 1) % args.accum == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
            tot += float(loss.detach())
            n += 1
        # Report top-1 every epoch, not only at the end: the whole question
        # this experiment answers is whether the encoder can be taught to
        # expose the state, and that shows up as top-1 accuracy long before
        # the training budget is spent.
        enc.eval()
        head.eval()
        hit = 0
        # Diagnostics, because "loss keeps falling while top-1 collapses"
        # has more than one possible cause and guessing is not allowed:
        #   gate_entropy  -> MoE router collapse (experts stop specialising)
        #   emb_cos       -> representation collapse (every state looks the
        #                    same after normalisation, so no head can rank)
        #   logit_std     -> the head stopped discriminating at all
        gate_ent, emb_cos, logit_std = [], [], []
        with torch.no_grad():
            for g in trainable[:200]:
                texts = [g["state"] + SEP + a for a in g["actions"]]
                em, _ = embed_train(enc, tokenizer, texts, device,
                                    args.max_length, 64)
                lg = head(em.to(device)).float().cpu().view(1, -1)
                hit += 1 if int(lg.argmax()) == g["_tgt"] else 0
                logit_std.append(float(lg.std()))
                # mean pairwise cosine between the action embeddings of one
                # state: 1.0 means every action looks identical
                e = F.normalize(em.to(device).float(), dim=-1)
                c = e @ e.T
                n = c.shape[0]
                emb_cos.append(float((c.sum() - n) / max(n * (n - 1), 1)))
            for layer in enc.backbone.layers:
                gate = getattr(layer.mlp, "gate", None)
                if gate is None:
                    continue
                # router entropy on the current weights is not directly
                # available, so report the weight norm spread of the router
                w = gate.weight.detach()
                gate_ent.append(float(w.norm(dim=-1).std() /
                                      w.norm(dim=-1).mean().clamp(min=1e-9)))
        enc.train()
        head.train()
        print(f"[ft] epoch {ep+1}/{args.epochs} loss={tot/max(n,1):.4f} "
              f"top1={hit/200:.3f} "
              f"logit_std={sum(logit_std)/len(logit_std):.4f} "
              f"emb_cos={sum(emb_cos)/len(emb_cos):.4f} "
              f"gate_wspread={sum(gate_ent)/max(len(gate_ent),1):.4f}",
              flush=True)

    enc.eval()
    head.eval()
    hit = 0
    with torch.no_grad():
        for g in trainable[:200]:
            texts = [g["state"] + SEP + a for a in g["actions"]]
            em, _ = embed_train(enc, tokenizer, texts, device, args.max_length, 64)
            lg = head(em.to(device)).float().cpu().view(1, -1)
            hit += 1 if int(lg.argmax()) == g["_tgt"] else 0
    print(f"[ft] final train top-1 accuracy (first 200 groups): {hit/200:.3f}",
          flush=True)

    if args.save:
        os.makedirs(args.save, exist_ok=True)
        torch.save({"backbone": enc.backbone.state_dict(),
                    "head": head.state_dict(),
                    "epoch": args.epochs},
                   os.path.join(args.save, "ft_state.pt"))
        print(f"[ft] saved state -> {args.save}", flush=True)

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
        print(f"[ft] baseline {k:9s} success="
              f"{sum(1 for r in v if r[0])/max(len(v),1):.3f} "
              f"progress={sum(r[1] for r in v)/max(len(v),1):.3f}", flush=True)

    res = {}
    for w in (1, args.width):
        succ, prog = [], []
        for (n, m, cons, sd, opt) in te_tasks:
            env = PlanningEnv(n, m, cons, sd)
            ok, pg, _ = beam_ft(enc, tokenizer, head, device, env, args.budget,
                                w, args.max_length)
            succ.append((ok, pg))
        s_ = sum(1 for r in succ if r[0]) / max(len(succ), 1)
        p_ = sum(r[1] for r in succ) / max(len(succ), 1)
        print(f"[ft] beam[ft w={w}] success={s_:.3f} progress={p_:.3f}",
              flush=True)
        res[f"w{w}"] = {"success": s_, "progress": p_}

    out = {"args": vars(args), "backbone_mode": mode,
           "trainable_params": n_train,
           "train_top1_acc": hit / 200, "beam": res,
           "baselines": {k: {"success": sum(1 for r in v if r[0]) / max(len(v), 1),
                             "progress": sum(r[1] for r in v) / max(len(v), 1)}
                         for k, v in base.items()}}
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"[ft] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
