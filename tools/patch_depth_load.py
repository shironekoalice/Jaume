import io
p = r"code\repo\scripts\train_vlm_probe.py"
s = io.open(p, encoding="utf-8").read()
old = """    enc = JaumeEmbeddingModel(cfg).to(device)
    enc.load_state_dict(load_state_dict(
        os.path.join(args.model, "pytorch_model.bin")), strict=True)
    enc.eval()"""
new = """    enc = JaumeEmbeddingModel(cfg).to(device)
    pretrained = load_state_dict(os.path.join(args.model, "pytorch_model.bin"))
    if cfg.num_layers == PRETRAINED_LAYERS:
        enc.load_state_dict(pretrained, strict=True)
    else:
        # Depth scan: keep every pretrained weight and reuse the last trained
        # block for extra layers.  Otherwise 8L fails on unexpected keys and
        # 32L fails on missing keys.
        own = enc.state_dict()
        merged, reused = {}, 0
        for k in own:
            if k in pretrained:
                merged[k] = pretrained[k]
            elif k.startswith("backbone.layers."):
                rest = k.split(".", 3)[3]
                src = "backbone.layers.%d.%s" % (PRETRAINED_LAYERS - 1, rest)
                if src in pretrained:
                    merged[k] = pretrained[src]
                    reused += 1
                else:
                    merged[k] = own[k]
            else:
                merged[k] = own[k]
        enc.load_state_dict(merged, strict=True)
        print("[vlm] depth=%d depth-upcycled %d tensors from layer %d"
              % (cfg.num_layers, reused, PRETRAINED_LAYERS - 1), flush=True)
    enc.eval()"""
assert old in s, "anchor not found"
s = s.replace(old, new)
if "PRETRAINED_LAYERS = 16" not in s:
    idx = s.index("def main():")
    s = s[:idx] + "PRETRAINED_LAYERS = 16\n\n\n" + s[idx:]
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("patched")
