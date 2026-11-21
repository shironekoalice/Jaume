import io
p = "code/repo/scripts/train_lm.py"
src = io.open(p, encoding="utf-8").read()
old = """    ap.add_argument("--loop-position-mode", choices=["repeat", "increment"],
                    default="repeat")"""
new = """    ap.add_argument("--loop-position-mode", choices=["repeat", "increment"],
                    default="repeat")
    ap.add_argument("--loop-gate", action="store_true",
                    help="per-round learned gate on each round's residual "
                         "update. Round 0 stays pinned to 1.0 so at init the "
                         "model equals the plain loop.")
    ap.add_argument("--loop-gate-min", type=float, default=0.0)"""
assert old in src
src = src.replace(old, new)
old = """    cfg.loop_position_mode = args.loop_position_mode"""
new = """    cfg.loop_position_mode = args.loop_position_mode
    cfg.loop_gate = args.loop_gate
    cfg.loop_gate_min = args.loop_gate_min"""
assert old in src
src = src.replace(old, new)
# report the gate values with the val eval so the decay can be watched
old = """                if args.eval_every and step % args.eval_every == 0:
                    vl = evaluate(model, val_blocks, device, args.loss_chunk,
                                  args.eval_batches, args.batch_size)
                    print(f"[lm] epoch {ep} step {step} VAL loss {vl:.4f} "
                          f"ppl {math.exp(min(vl, 20)):.2f}", flush=True)"""
new = """                if args.eval_every and step % args.eval_every == 0:
                    vl = evaluate(model, val_blocks, device, args.loss_chunk,
                                  args.eval_batches, args.batch_size)
                    gate_str = ""
                    gp = getattr(model.backbone, "loop_gate_param", None)
                    if gp is not None:
                        gate_str = " gate[" + ",".join(
                            f"{float(v):.2f}" for v in gp.detach()) + "]"
                    print(f"[lm] epoch {ep} step {step} VAL loss {vl:.4f} "
                          f"ppl {math.exp(min(vl, 20)):.2f}{gate_str}",
                          flush=True)"""
assert old in src
src = src.replace(old, new)
io.open(p, "w", encoding="utf-8", newline="\n").write(src)
print("patched train_lm: gate")
