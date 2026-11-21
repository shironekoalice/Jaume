import io
p = "code/repo/scripts/plan_finetune.py"
src = io.open(p, encoding="utf-8").read()

# ---- diagnostics inside the per-epoch evaluation block ------------------
old = """        enc.eval()
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
              f"top1={hit/200:.3f}", flush=True)"""
new = """        enc.eval()
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
              flush=True)"""
assert old in src
src = src.replace(old, new)

# ---- save a checkpoint so a collapse can be inspected afterwards --------
old = """    enc.eval()
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
new = """    enc.eval()
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
        print(f"[ft] saved state -> {args.save}", flush=True)"""
assert old in src
src = src.replace(old, new)

old = """    ap.add_argument("--out", default=None)
    args = ap.parse_args()"""
new = """    ap.add_argument("--out", default=None)
    ap.add_argument("--save", default=None,
                    help="directory to dump the fine-tuned weights, so a "
                         "collapse can be inspected afterwards")
    args = ap.parse_args()"""
assert old in src
src = src.replace(old, new)
io.open(p, "w", encoding="utf-8", newline="\n").write(src)
print("patched: diagnostics + checkpoint saving")
