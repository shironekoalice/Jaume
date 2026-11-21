import io
p = "code/repo/scripts/plan_finetune.py"
src = io.open(p, encoding="utf-8").read()
old = """        print(f"[ft] epoch {ep+1}/{args.epochs} loss={tot/max(n,1):.4f}",
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
    print(f"[ft] train top-1 accuracy (first 200 groups): {hit/200:.3f}",
          flush=True)"""
new = """        # Report top-1 every epoch, not only at the end: the whole question
        # this experiment answers is whether the encoder can be taught to
        # expose the state, and that shows up as top-1 accuracy long before
        # the training budget is spent.
        enc.eval()
        head.eval()
        hit = 0
        with torch.no_grad():
            for g in trainable[:200]:
                texts = [g["state"] + SEP + a for a in g["actions"]]
                em, _ = embed_train(enc, tokenizer, texts, device,
                                    args.max_length, 64)
                lg = head(em.to(device)).float().cpu().view(1, -1)
                hit += 1 if int(lg.argmax()) == g["_tgt"] else 0
        enc.train()
        head.train()
        print(f"[ft] epoch {ep+1}/{args.epochs} loss={tot/max(n,1):.4f} "
              f"top1={hit/200:.3f}", flush=True)

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
          flush=True)"""
assert old in src
src = src.replace(old, new)
io.open(p, "w", encoding="utf-8", newline="\n").write(src)
print("patched: per-epoch top-1")
