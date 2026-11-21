import io
p = "code/repo/scripts/train_vlm_probe.py"
s = io.open(p, encoding="utf-8").read()

# add --unfreeze-layers: unfreeze the top N transformer layers
s = s.replace(
'''    ap.add_argument("--device", default=None)''',
'''    ap.add_argument("--unfreeze-layers", type=int, default=0,
                    help="unfreeze the top N Jaume layers so the encoder can "
                         "adapt to the visual prefix (0 = fully frozen)")
    ap.add_argument("--device", default=None)''')

s = s.replace(
'''    for p in enc.parameters():
        p.requires_grad_(False)
    d = cfg.hidden_size''',
'''    for p in enc.parameters():
        p.requires_grad_(False)
    if args.unfreeze_layers > 0:
        # The frozen encoder cannot re-interpret a visual prefix: its layers
        # were trained on token embeddings only.  Unfreezing the top N layers
        # lets it learn to consume the projector's output, which is the same
        # design choice modern VLMs make (train projector + top blocks).
        for layer in list(enc.backbone.layers)[-args.unfreeze_layers:]:
            for p in layer.parameters():
                p.requires_grad_(True)
        for p in enc.backbone.norm.parameters():
            p.requires_grad_(True)
    d = cfg.hidden_size''')

# include the unfrozen encoder params in the optimizer
s = s.replace(
'''    opt = torch.optim.AdamW(
        list(vis.parameters()) + list(proj.parameters()) + list(head.parameters()),
        lr=args.lr, weight_decay=0.01)''',
'''    trainable = (list(vis.parameters()) + list(proj.parameters())
                 + list(head.parameters())
                 + [p for p in enc.parameters() if p.requires_grad])
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    print("[vlm] unfreezing top %d encoder layers -> %d trainable tensors" % (
        args.unfreeze_layers, len(trainable)), flush=True)''')
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("patched")
