import io
p = "code/repo/scripts/train_lm.py"
src = io.open(p, encoding="utf-8").read()

old = """    ap.add_argument("--loop-position-mode", choices=["repeat", "increment"],
                    default="repeat")"""
new = """    ap.add_argument("--loop-position-mode", choices=["repeat", "increment"],
                    default="repeat")
    ap.add_argument("--loop-align-weight", type=float, default=0.0,
                    help="LoopMTP-style latent soft alignment: softly align the "
                         "hidden state of loop t with the token embedding t "
                         "positions ahead. 0 disables it.")"""
assert old in src
src = src.replace(old, new)

old = """            logits, aux = model(inp)
            loss = ntp_loss(logits, tgt, args.loss_chunk)
            if aux is not None:
                loss = loss + cfg.load_balance_weight * aux"""
new = """            if args.loop_align_weight > 0:
                logits, aux, loop_states = model(inp, return_loop_states=True)
            else:
                logits, aux = model(inp)
                loop_states = None
            loss = ntp_loss(logits, tgt, args.loss_chunk)
            if aux is not None:
                loss = loss + cfg.load_balance_weight * aux
            if loop_states is not None:
                loss = loss + args.loop_align_weight * loop_align_loss(
                    loop_states, model.backbone.embed_tokens.weight, inp)"""
assert old in src
src = src.replace(old, new)

old = """def main():
    ap = argparse.ArgumentParser()"""
new = '''def loop_align_loss(loop_states, emb_weight, input_ids):
    """LoopMTP-style latent soft alignment.

    LoopMTP (arXiv 2608.03624) observes that a model that loops T times can
    anticipate T future tokens, and supervises the loop by softly aligning
    the hidden state of loop t with the embedding of the token t positions
    ahead.  That gives the loop a dense, forward-looking signal, which plain
    next-token prediction does not provide to the intermediate rounds.

    The alignment here is the cosine between the L2-normalised loop-t hidden
    state at position i and the L2-normalised token embedding at position
    i+t.  Only positions where i+t is still inside the sequence contribute.
    The last round is excluded: it carries the next-token prediction, and
    aligning it with the future as well would double-count the objective.
    """
    total = None
    n = 0
    B, L = input_ids.shape
    T = len(loop_states)
    for t in range(1, T):
        h = F.normalize(loop_states[t][:, : L - t].float(), dim=-1)
        e = F.normalize(emb_weight[input_ids[:, t:]].float(), dim=-1)
        loss = 1.0 - (h * e).sum(-1).mean()
        total = loss if total is None else total + loss
        n += 1
    if total is None:
        return torch.zeros((), device=input_ids.device)
    return total / n


def main():
    ap = argparse.ArgumentParser()'''
assert old in src
src = src.replace(old, new)
io.open(p, "w", encoding="utf-8", newline="\n").write(src)
print("patched: loop alignment loss")
