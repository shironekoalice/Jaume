import io
p = "code/repo/scripts/train_lm_planning.py"
src = io.open(p, encoding="utf-8").read()
old = """            batch = [train[j] for j in perm[i:i + args.batch_size]]
            ids = torch.tensor([b["ids"] for b in batch], device=device)
            act_start = torch.tensor([b["act_start"] for b in batch],
                                     device=device)"""
new = """            batch = [train[j] for j in perm[i:i + args.batch_size]]
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
            act_start = torch.tensor(act_list, device=device)"""
assert old in src
src = src.replace(old, new)

old = """def ntp_loss_actiononly(logits, ids, act_start):"""
new = """def ntp_loss_actiononly(logits, ids, act_start, lengths=None):"""
assert old in src
src = src.replace(old, new)
old = """    mask = torch.zeros_like(tgt, dtype=torch.bool)
    for b in range(B):
        s = act_start[b]
        if s >= 1 and s < L:
            mask[b, max(0, s - 1):] = True"""
new = """    mask = torch.zeros_like(tgt, dtype=torch.bool)
    for b in range(B):
        s = act_start[b]
        end = lengths[b] - 1 if lengths is not None else L
        if s >= 1 and end > s - 1:
            mask[b, max(0, s - 1):end] = True"""
assert old in src
src = src.replace(old, new)
old = """            logits, aux = model(ids)
            loss = ntp_loss_actiononly(logits, ids, act_start)"""
new = """            logits, aux = model(ids)
            loss = ntp_loss_actiononly(logits, ids, act_start,
                                       torch.tensor(len_list, device=device))"""
assert old in src
src = src.replace(old, new)
old = """        ids = torch.tensor([b["ids"] for b in batch], device=device)
        act_start = torch.tensor([b["act_start"] for b in batch],
                                 device=device)
        logits, aux = model(ids)
        loss = ntp_loss_actiononly(logits, ids, act_start)"""
new = """        L = max(len(b["ids"]) for b in batch)
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
assert old in src
src = src.replace(old, new)
io.open(p, "w", encoding="utf-8", newline="\n").write(src)
print("patched: padding + length-aware loss")
