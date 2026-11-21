import io
p = "code/repo/scripts/train_lm.py"
src = io.open(p, encoding="utf-8").read()

# ---- split train / val blocks ----------------------------------------
old = """    t0 = time.time()
    blocks = build_blocks(texts, tokenizer, args.max_length, args.max_blocks,
                          seed=args.seed)
    print(f"[lm] {blocks.shape[0]} blocks of {args.max_length} tokens "
          f"({time.time()-t0:.0f}s)", flush=True)"""
new = """    t0 = time.time()
    blocks = build_blocks(texts, tokenizer, args.max_length, args.max_blocks,
                          seed=args.seed)
    n_val = max(args.val_blocks, 8)
    val_blocks = blocks[:n_val]
    blocks = blocks[n_val:]
    print(f"[lm] {blocks.shape[0]} train blocks of {args.max_length} tokens, "
          f"{val_blocks.shape[0]} held out ({time.time()-t0:.0f}s)", flush=True)"""
assert old in src
src = src.replace(old, new)

# ---- CLI ---------------------------------------------------------------
old = """    ap.add_argument("--max-blocks", type=int, default=None)"""
new = """    ap.add_argument("--max-blocks", type=int, default=None)
    ap.add_argument("--val-blocks", type=int, default=64)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-batches", type=int, default=8)"""
assert old in src
src = src.replace(old, new)

# ---- eval function -----------------------------------------------------
old = """def loop_align_loss(loop_states, emb_weight, input_ids):"""
new = '''@torch.no_grad()
def evaluate(model, val_blocks, device, loss_chunk=0, max_batches=8,
             batch_size=8):
    """Held-out loss. This is the number that decides whether a loop helps."""
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, min(val_blocks.shape[0], max_batches * batch_size),
                   batch_size):
        b = val_blocks[i:i + batch_size].to(device)
        inp, tgt = b[:, :-1], b[:, 1:]
        logits, aux = model(inp)
        loss = ntp_loss(logits, tgt, loss_chunk)
        tot += float(loss) * b.shape[0]
        n += b.shape[0]
    model.train()
    return tot / max(n, 1)


def loop_align_loss(loop_states, emb_weight, input_ids):'''
assert old in src
src = src.replace(old, new)

# ---- call it during training ------------------------------------------
old = """                if step % args.log_every == 0:
                    tok_s = (step * args.batch_size * args.grad_accum
                             * args.max_length) / max(time.time() - t0, 1e-9)
                    print(f"[lm] epoch {ep} step {step}/{total_steps} "
                          f"loss {float(loss):.4f} ppl {math.exp(min(float(loss),20)):.2f} "
                          f"lr {lr_at(step):.2e} tokens/s {tok_s:.0f}", flush=True)"""
new = """                if step % args.log_every == 0:
                    tok_s = (step * args.batch_size * args.grad_accum
                             * args.max_length) / max(time.time() - t0, 1e-9)
                    print(f"[lm] epoch {ep} step {step}/{total_steps} "
                          f"loss {float(loss.detach()):.4f} "
                          f"lr {lr_at(step):.2e} tokens/s {tok_s:.0f}", flush=True)
                if args.eval_every and step % args.eval_every == 0:
                    vl = evaluate(model, val_blocks, device, args.loss_chunk,
                                  args.eval_batches, args.batch_size)
                    print(f"[lm] epoch {ep} step {step} VAL loss {vl:.4f} "
                          f"ppl {math.exp(min(vl, 20)):.2f}", flush=True)"""
assert old in src
src = src.replace(old, new)

# ---- final val ---------------------------------------------------------
old = """    if args.output:
        save(model, cfg, tokenizer, args.output, step)
    print(f"[lm] done in {time.time()-t0:.0f}s", flush=True)"""
new = """    vl = evaluate(model, val_blocks, device, args.loss_chunk,
                  args.eval_batches, args.batch_size)
    print(f"[lm] FINAL VAL loss {vl:.4f} ppl {math.exp(min(vl, 20)):.2f}",
          flush=True)
    if args.output:
        save(model, cfg, tokenizer, args.output, step)
    print(f"[lm] done in {time.time()-t0:.0f}s", flush=True)"""
assert old in src
src = src.replace(old, new)
io.open(p, "w", encoding="utf-8", newline="\n").write(src)
print("patched: validation")
