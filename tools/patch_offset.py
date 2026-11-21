import io
p = "code/repo/scripts/train_lm.py"
src = io.open(p, encoding="utf-8").read()

old = """def loop_align_loss(loop_states, emb_weight, input_ids):"""
new = """def loop_align_loss(loop_states, emb_weight, input_ids, offset=1):"""
assert old in src
src = src.replace(old, new)

old = """    The alignment here is the cosine between the L2-normalised loop-t hidden
    state at position i and the L2-normalised token embedding at position
    i+t.  Only positions where i+t is still inside the sequence contribute.
    The last round is excluded: it carries the next-token prediction, and
    aligning it with the future as well would double-count the objective.
    \"\"\""""
new = """    The alignment here is the cosine between the L2-normalised loop-t hidden
    state at position i and the L2-normalised token embedding at position
    i+t+offset.  Only positions where the target is still inside the
    sequence contribute.  The last round is excluded: it carries the
    next-token prediction, and aligning it with the future as well would
    double-count the objective.

    offset: a causal model's hidden state at position i encodes the token at
    position i+1, so "the token t steps ahead" of what that state predicts
    is at position i+1+t -- that is offset=1, the default.  offset=0 aligns
    against the token already visible at position i, which is one step
    closer than LoopMTP's wording implies; it is kept switchable so the
    choice can be checked rather than assumed.
    \"\"\""""
assert old in src
src = src.replace(old, new)

old = """    for t in range(1, T):
        h = F.normalize(loop_states[t][:, : L - t].float(), dim=-1)
        e = F.normalize(emb_weight[input_ids[:, t:]].float(), dim=-1)
        loss = 1.0 - (h * e).sum(-1).mean()"""
new = """    for t in range(1, T):
        tgt_slice = input_ids[:, t + offset:]
        h = F.normalize(loop_states[t][:, : L - t - offset].float(), dim=-1)
        e = F.normalize(emb_weight[tgt_slice].float(), dim=-1)
        loss = 1.0 - (h * e).sum(-1).mean()"""
assert old in src
src = src.replace(old, new)

old = """                loss = loss + args.loop_align_weight * loop_align_loss(
                    loop_states, model.backbone.embed_tokens.weight, inp)"""
new = """                loss = loss + args.loop_align_weight * loop_align_loss(
                    loop_states, model.backbone.embed_tokens.weight, inp,
                    args.loop_align_offset)"""
assert old in src
src = src.replace(old, new)

old = """                         "positions ahead. 0 disables it.")"""
new = """                         "positions ahead. 0 disables it.")
    ap.add_argument("--loop-align-offset", type=int, default=1,
                    help="0 aligns against the token already visible at the "
                         "same position; 1 (default) against the token one "
                         "step further, which matches 't steps ahead' for a "
                         "causal model whose hidden state predicts i+1.")"""
assert old in src
src = src.replace(old, new)
io.open(p, "w", encoding="utf-8", newline="\n").write(src)
print("patched: align offset")
