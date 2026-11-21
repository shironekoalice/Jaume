"""GradCache: large-batch contrastive training with bounded activation memory.

Gao et al. 2021. Two passes over the batch:
1. Encode every chunk without grad, compute the contrastive loss on detached
   embeddings, and take per-example embedding gradients.
2. Re-encode each chunk WITH grad and backprop the surrogate
   sum(embedding * per_example_grad), which reproduces the exact full-batch
   gradient without ever holding activations for the whole batch.

Gradient equivalence holds exactly (same graph, chunked); the two passes
differ only by float nondeterminism when autocast is on.
"""
import torch


@torch.no_grad()
def _encode(model, ids, mask, device_type, amp_dtype, use_amp):
    with torch.autocast(device_type, dtype=amp_dtype, enabled=use_amp):
        emb, aux = model(ids, mask)
    return emb, aux


def gradcache_step(
    model,
    loss_fn,
    q_ids,
    q_mask,
    p_ids,
    p_mask,
    n_ids=None,
    n_mask=None,
    num_hard=0,
    chunk_size=32,
    load_balance_weight=0.0,
    device_type="cuda",
    amp_dtype=torch.bfloat16,
):
    """One optimizer step's worth of gradients, GradCache style.

    loss_fn(query_emb, passage_emb, hard_negative_emb_or_None) -> scalar.
    Queries/passages/hard negatives are encoded in chunks of `chunk_size`
    sequences... note chunk_size counts PAIRS here for q/p (a chunk with H
    hard negatives per pair processes chunk_size*(2+H) sequences).

    Adds `load_balance_weight * aux` from the query forwards, matching the
    plain training path. Gradients accumulate into model parameters; the
    caller is responsible for optimizer.step()/zero_grad(). Returns the
    true loss value (detached tensor) for logging.
    """
    use_amp = device_type in ("cuda", "xpu")
    B = q_ids.size(0)
    idx = torch.arange(B)
    chunks = torch.split(idx, chunk_size)

    # pass 1: no-grad embeddings for the whole (effective) batch.
    # Chunks index pairs; each pair owns H hard-negative rows.
    q_parts, p_parts, n_parts = [], [], []
    for chunk in chunks:
        q_emb, aux = _encode(model, q_ids[chunk], q_mask[chunk], device_type, amp_dtype, use_amp)
        p_emb, _ = _encode(model, p_ids[chunk], p_mask[chunk], device_type, amp_dtype, use_amp)
        q_parts.append(q_emb.float())
        p_parts.append(p_emb.float())
        if n_ids is not None:
            n_slice = slice(int(chunk[0]) * num_hard, (int(chunk[-1]) + 1) * num_hard)
            n_emb, _ = _encode(model, n_ids[n_slice], n_mask[n_slice], device_type, amp_dtype, use_amp)
            n_parts.append(n_emb.float())
    q_all = torch.cat(q_parts)
    p_all = torch.cat(p_parts)
    n_all = torch.cat(n_parts) if n_parts else None

    # per-example gradients of the loss w.r.t. the embeddings
    q_leaf = q_all.detach().requires_grad_(True)
    p_leaf = p_all.detach().requires_grad_(True)
    n_leaf = n_all.detach().requires_grad_(True) if n_all is not None else None
    n_view = n_leaf.view(B, num_hard, -1) if n_leaf is not None else None
    loss = loss_fn(q_leaf, p_leaf, n_view)
    loss.backward()
    q_grad, p_grad = q_leaf.grad, p_leaf.grad
    n_grad = n_leaf.grad if n_leaf is not None else None

    # pass 2: replay with grad; sum(emb * g) has the same parameter gradient.
    # The MoE aux loss is a token MEAN per forward, so each chunk's aux term
    # is weighted by its token share to reproduce the full-batch mean.
    total_q_tokens = int(q_mask.sum())
    for i, chunk in enumerate(chunks):
        with torch.autocast(device_type, dtype=amp_dtype, enabled=use_amp):
            q_emb, aux = model(q_ids[chunk], q_mask[chunk])
            p_emb, _ = model(p_ids[chunk], p_mask[chunk])
            token_share = int(q_mask[chunk].sum()) / max(1, total_q_tokens)
            surrogate = (
                q_emb.float() * q_grad[chunk]
                + p_emb.float() * p_grad[chunk]
            ).sum() + load_balance_weight * aux.float() * token_share
            if n_ids is not None:
                n_slice = slice(int(chunk[0]) * num_hard, (int(chunk[-1]) + 1) * num_hard)
                n_emb, _ = model(n_ids[n_slice], n_mask[n_slice])
                surrogate = surrogate + (n_emb.float() * n_grad[n_slice]).sum()
        surrogate.backward()
    return loss.detach()
