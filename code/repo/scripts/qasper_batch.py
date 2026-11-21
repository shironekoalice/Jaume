"""Batched utilities for the Qasper experiment.

Why this exists: the first implementation encoded one block at a time (batch 1)
for each of ~12-16 blocks per paper. Measured GPU utilisation was 4% -- the
encoder spent nearly all its time waiting on tiny forward passes.

Here papers that are in the same training batch advance their block pointers in
LOCK-STEP: at position i, every paper contributes its i-th block, so each
forward carries B blocks (B = papers per step) instead of one. Papers with fewer
blocks drop out of later positions, which is handled by keeping a mapping from
row -> paper.

The recurrent state is [B, slots, dim] and `encode` already accepts it in that
form, so passing the state across positions is a plain tensor.
"""
import torch
import torch.nn.functional as F


def build_stage(batch_rows, max_blocks):
    """Flatten a batch of papers into per-position block lists.

    Returns:
        texts   : list[str], grouped by position
        owners  : list[list[int]] the paper index for each text at that position
        n_pos   : number of positions (max blocks among the batch)
    """
    n_pos = max(len(r["blocks"][:max_blocks]) for r in batch_rows)
    texts, owners = [], []
    for pos in range(n_pos):
        t, o = [], []
        for pi, r in enumerate(batch_rows):
            bs = r["blocks"][:max_blocks]
            if pos < len(bs):
                t.append(bs[pos])
                o.append(pi)
        texts.append(t)
        owners.append(o)
    return texts, owners, n_pos


def run_batch_states(rm, batch_rows, pass_state, max_length=256, max_blocks=16):
    """Encode a batch of papers through the (optional) recurrent path.

    Returns a list of [n_blocks, dim] tensors, one per paper, where row i is the
    representation AFTER reading blocks 0..i (state path) or the block's own
    pooled vector (no state).
    """
    texts, owners, n_pos = build_stage(batch_rows, max_blocks)
    B = len(batch_rows)
    dim = rm.model.config.hidden_size
    slots = rm.model.pooling.num_slots if hasattr(rm.model.pooling, "num_slots") else 1

    state = None                      # [B, slots, dim] or None
    collected = [[] for _ in range(B)]

    for pos in range(n_pos):
        part = texts[pos]
        own = owners[pos]
        if not part:
            continue
        e = rm.tokenizer(part, padding=True, truncation=True,
                         max_length=max_length, return_tensors="pt").to(rm.device)
        st_in = None
        if pass_state and state is not None:
            st_in = state.index_select(0, torch.tensor(own, device=rm.device))
        pooled, _, new_state = rm.model.encode(e["input_ids"], e["attention_mask"], st_in)

        if new_state is not None:
            # place the freshly produced states back at their paper indices
            if state is None:
                state = torch.zeros(B, new_state.size(1), new_state.size(2),
                                    device=new_state.device, dtype=new_state.dtype)
            idx = torch.tensor(own, device=new_state.device)
            state = state.index_copy(0, idx, new_state)
            vec = new_state.mean(dim=1)
        else:
            vec = pooled
        vec = F.normalize(rm.model.projection(vec), dim=-1)
        for k, pi in enumerate(own):
            collected[pi].append(vec[k])

    return [torch.stack(c) if c else torch.zeros(0, rm.model.config.embedding_dim)
            for c in collected]


def question_embs(rm, questions, max_length=128):
    outs = []
    with torch.no_grad():
        for i in range(0, len(questions), 32):
            part = questions[i : i + 32]
            e = rm.tokenizer(part, padding=True, truncation=True,
                             max_length=max_length, return_tensors="pt").to(rm.device)
            emb, _ = rm.model(e["input_ids"], e["attention_mask"])
            outs.append(emb)
    return torch.cat(outs, dim=0)


def rank_all(reps, qemb, golds):
    """accuracy@1 and MRR for a batch, plus per-item ranks."""
    a1 = mrr = 0.0
    ranks = []
    for rep, q, g in zip(reps, qemb, golds):
        sc = F.normalize(rep, dim=-1) @ q
        order = torch.argsort(sc, descending=True).tolist()
        r = order.index(g) + 1
        ranks.append(r)
        a1 += 1.0 if r == 1 else 0.0
        mrr += 1.0 / r
    n = max(1, len(ranks))
    return a1 / n, mrr / n, ranks
