import io
p = "code/repo/scripts/train_lm_planning.py"
src = io.open(p, encoding="utf-8").read()

# ---- 1) pre-pad everything ONCE into a dense tensor ---------------------
old = """    encoded = encode_docs(docs, tokenizer, args.max_length)
    print(f"[pl] {len(encoded)} encoded docs", flush=True)"""
new = """    encoded = encode_docs(docs, tokenizer, args.max_length)
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
          f"({ids_all.numel()*ids_all.element_size()/1e6:.0f} MB)", flush=True)"""
assert old in src
src = src.replace(old, new)

# ---- 2) vectorised action-only loss (no Python loop, no per-row mask) ---
old = """def ntp_loss_actiononly(logits, ids, act_start, lengths=None):
    \"\"\"Cross-entropy on the action tokens only (predicting one position
    ahead).  Position i predicts id[i+1], so the action target at position
    act_start-1 is predicted from logits position act_start-2.\"\"\"
    B, L, V = logits.shape
    tgt = ids[:, 1:]                       # shift: position i predicts i+1
    lg = logits[:, :-1, :]
    mask = torch.zeros_like(tgt, dtype=torch.bool)
    for b in range(B):
        s = act_start[b]
        end = lengths[b] - 1 if lengths is not None else L
        if s >= 1 and end > s - 1:
            mask[b, max(0, s - 1):end] = True
    n = int(mask.sum())
    if n == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(lg[mask], tgt[mask])"""
new = """def ntp_loss_actiononly(logits, ids, act_start, lengths):
    \"\"\"Cross-entropy on the action tokens only, fully vectorised.

    Position i predicts token i+1, so the action tokens occupy target
    positions [act_start-1, length-2].  The mask is built with two
    comparisons on an arange, which keeps the whole thing on the accelerator:
    the previous per-row Python loop plus boolean gather triggered Ascend's
    aclnnNonzeroV2 fallback and stalled the run at step 0.
    \"\"\"
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
    return F.cross_entropy(sel, tg.index_select(0, idx))"""
assert old in src
src = src.replace(old, new)

# ---- 3) training step: slice the dense tensors -------------------------
old = """            batch = [train[j] for j in perm[i:i + args.batch_size]]
            # variable length: pad to the longest, record the true length so
            # the loss can skip padding positions
            L = max(len(b["ids"]) for b in batch)
            pad = cfg.pad_token_id
            ids_list, act_list, len_list = [], [], []
            for b in batch:
                ids_list.append(b["ids"] + [pad] * (L - len(b["ids"])))
                act_list.append(b["act_start"])
                len_list.append(len(b["ids"]))
            ids = torch.tensor(ids_list, device=device)
            act_start = torch.tensor(act_list, device=device)
            logits, aux = model(ids)
            loss = ntp_loss_actiononly(logits, ids, act_start,
                                       torch.tensor(len_list, device=device))"""
new = """            sel = torch.tensor(perm[i:i + args.batch_size], device=device)
            ids = ids_all_dev.index_select(0, sel)
            act_start = act_all_dev.index_select(0, sel)
            ln = len_all_dev.index_select(0, sel)
            logits, aux = model(ids)
            loss = ntp_loss_actiononly(logits, ids, act_start, ln)"""
assert old in src
src = src.replace(old, new)

# ---- 4) move tensors to device once, and split train/val by index ------
old = """    val = encoded[:args.val_docs]
    train = encoded[args.val_docs:]
    print(f"[pl] train={len(train)} val={len(val)}", flush=True)"""
new = """    n_val = args.val_docs
    ids_all_dev = ids_all.to(device)
    len_all_dev = len_all.to(device)
    act_all_dev = act_all.to(device)
    train_idx = torch.arange(n_val, n_enc)
    val_idx = torch.arange(0, n_val)
    print(f"[pl] train={len(train_idx)} val={len(val_idx)}", flush=True)"""
assert old in src
src = src.replace(old, new)

# ---- 5) evaluate on the dense tensors ---------------------------------
old = """@torch.no_grad()
def evaluate(model, val_blocks, device, pad_id, loss_chunk=0, max_batches=4,
             batch_size=8):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, min(len(val_blocks), max_batches * batch_size),
                   batch_size):
        batch = val_blocks[i:i + batch_size]
        L = max(len(b["ids"]) for b in batch)
        pad = pad_id
        ids_list, act_list, len_list = [], [], []
        for b in batch:
            ids_list.append(b["ids"] + [pad] * (L - len(b["ids"])))
            act_list.append(b["act_start"])
            len_list.append(len(b["ids"]))
        ids = torch.tensor(ids_list, device=device)
        act_start = torch.tensor(act_list, device=device)
        logits, aux = model(ids)
        loss = ntp_loss_actiononly(logits, ids, act_start,
                                   torch.tensor(len_list, device=device))
        tot += float(loss) * len(batch)
        n += len(batch)
    model.train()
    return tot / max(n, 1)"""
new = """@torch.no_grad()
def evaluate(model, ids_all, len_all, act_all, val_idx, max_batches=4,
             batch_size=8):
    # no_grad is not optional: without it every eval batch builds a full
    # 369M-parameter graph that is never freed and the run eventually hangs
    # (observed as AICore ~0% with the step counter frozen).
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, min(len(val_idx), max_batches * batch_size), batch_size):
        sel = val_idx[i:i + batch_size]
        ids = ids_all.index_select(0, sel)
        logits, aux = model(ids)
        loss = ntp_loss_actiononly(logits, ids,
                                   act_all.index_select(0, sel),
                                   len_all.index_select(0, sel))
        tot += float(loss) * len(sel)
        n += len(sel)
    model.train()
    return tot / max(n, 1)"""
assert old in src
src = src.replace(old, new)

# ---- 6) fix the call sites --------------------------------------------
src = src.replace(
    "vl = evaluate(model, val, device, cfg.pad_token_id)",
    "vl = evaluate(model, ids_all_dev, len_all_dev, act_all_dev, val_idx)")
src = src.replace(
    "                    vl = evaluate(model, val, device, cfg.pad_token_id)",
    "                    vl = evaluate(model, ids_all_dev, len_all_dev,\n"
    "                                  act_all_dev, val_idx)")

# ---- 7) the epoch loop indexes into the dense tensors -----------------
old = """        perm = list(range(len(train)))
        random.Random(args.seed).shuffle(perm)
        opt.zero_grad(set_to_none=True)
        for bi, i in enumerate(range(0, len(train) - args.batch_size + 1,
                                      args.batch_size)):"""
new = """        perm = train_idx[torch.randperm(len(train_idx))]
        opt.zero_grad(set_to_none=True)
        for bi, i in enumerate(range(0, len(train_idx) - args.batch_size + 1,
                                      args.batch_size)):"""
assert old in src
src = src.replace(old, new)

old = """            sel = torch.tensor(perm[i:i + args.batch_size], device=device)
            ids = ids_all_dev.index_select(0, sel)"""
new = """            sel = perm[i:i + args.batch_size]
            ids = ids_all_dev.index_select(0, sel)"""
assert old in src
src = src.replace(old, new)
io.open(p, "w", encoding="utf-8", newline="\n").write(src)
print("rewritten: dense tensors + vectorised loss")
