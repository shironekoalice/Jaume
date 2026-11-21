import io
p = "code/repo/scripts/plan_ce_study.py"
src = io.open(p, encoding="utf-8").read()

# --- beam(): add score_mode -------------------------------------------
old = """@torch.no_grad()
def beam(enc, tok, head, device, env, budget, width, max_length, encoding,
         cache=None):"""
new = """@torch.no_grad()
def beam(enc, tok, head, device, env, budget, width, max_length, encoding,
         cache=None, score_mode="logprob"):
    \"\"\"Beam search driven by the head.

    score_mode matters.  A head trained with softmax cross-entropy is
    calibrated WITHIN a state: it says which of the state's actions is best,
    not how good the state is.  Summing raw logits across a path therefore
    mixes two different quantities and lets a constant per-action offset
    dominate the ranking -- which showed up as progress 0.000.  Summing
    log-softmax instead sums path log-probabilities, i.e. the probability of
    the path under the model's per-step policy, which is the quantity the
    loss actually trained.

    width=1 makes this a greedy execution of the learned policy.
    \"\"\""""
assert old in src
src = src.replace(old, new)

old = """                a = torch.stack([cache[t] for t in texts]).to(device)
                sc = head(s.expand(len(acts), -1), a).float().cpu().tolist()
            else:
                ctx = state_text(e)
                texts = [ctx + SEP + action_text(a) for a in acts]
                em = embed_pair(enc, tok, texts, device, max_length)
                sc = head(em.to(device)).float().cpu().tolist()"""
new = """                a = torch.stack([cache[t] for t in texts]).to(device)
                raw = head(s.expand(len(acts), -1), a).float().cpu()
                sc = _score(raw, score_mode)
            else:
                ctx = state_text(e)
                texts = [ctx + SEP + action_text(a) for a in acts]
                em = embed_pair(enc, tok, texts, device, max_length)
                raw = head(em.to(device)).float().cpu()
                sc = _score(raw, score_mode)"""
assert old in src
src = src.replace(old, new)

# --- helper ------------------------------------------------------------
old = """@torch.no_grad()
def beam(enc, tok, head"""
new = """def _score(raw, mode):
    \"\"\"Turn a per-state vector of head outputs into path scores.\"\"\"
    if mode == "logit":
        return raw.tolist()
    if mode == "logprob":
        return F.log_softmax(raw, dim=-1).tolist()
    raise ValueError(f"unknown score_mode: {mode}")


@torch.no_grad()
def beam(enc, tok, head"""
assert old in src
src = src.replace(old, new)

# --- CLI + call site ----------------------------------------------------
old = """    ap.add_argument("--encoding", choices=["split", "joint"], default="split")"""
new = """    ap.add_argument("--encoding", choices=["split", "joint"], default="split")
    ap.add_argument("--score-mode", choices=["logit", "logprob"],
                    default="logprob")
    ap.add_argument("--widths", type=int, nargs="+", default=[1, 3])"""
assert old in src
src = src.replace(old, new)

old = """        cache = {}
        succ, prog = [], []
        for (n, m, cons, sd, opt) in te_tasks:
            env = PlanningEnv(n, m, cons, sd)
            ok, pg, _ = beam(enc, tokenizer, head, device, env, args.budget,
                             args.width, args.max_length, args.encoding, cache)
            succ.append((ok, pg))
        s_ = sum(1 for r in succ if r[0]) / max(len(succ), 1)
        p_ = sum(r[1] for r in succ) / max(len(succ), 1)
        print(f"[ce] beam[ce-{name:12s}] success={s_:.3f} progress={p_:.3f}",
              flush=True)
        heads[name] = {"train_top1_acc": acc, "success": s_, "progress": p_}"""
new = """        for w in args.widths:
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
                    "success": s_, "progress": p_}"""
assert old in src
src = src.replace(old, new)
io.open(p, "w", encoding="utf-8", newline="\n").write(src)
print("patched score modes")
