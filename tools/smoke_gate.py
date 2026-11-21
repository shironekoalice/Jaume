"""Verify the gate is a no-op at init and can diverge when trained."""
import os, sys, torch
sys.path.insert(0, "code/repo")
from jaume.modeling.config import JaumeConfig
from jaume.modeling.loop_lm import LoopLM

cfg = JaumeConfig.load("code/repo/configs/jaume-350m-loop.json")
cfg.vocab_size = 500; cfg.hidden_size = 64; cfg.intermediate_size = 128
cfg.num_layers = 4; cfg.num_attention_heads = 4; cfg.num_key_value_heads = 2
cfg.expert_num = 4; cfg.topk = 2; cfg.embedding_dim = 64
cfg.matryoshka_dims = (16, 32, 48, 64)
cfg.num_loops = 4
cfg.loop_gate = True

m = LoopLM(cfg)
ids = torch.randint(0, 500, (2, 16))

# 1) gated model at init == plain loop exactly (round 0 pinned, others start at 1)
cfg2 = cfg; cfg2.loop_gate = False
m2 = LoopLM(cfg2)
m2.load_state_dict(m.state_dict(), strict=False)
m.eval(); m2.eval()
with torch.no_grad():
    a, _ = m(ids); b, _ = m2(ids)
diff = (a-b).abs().max().item()
print(f"gated-vs-plain at init: max|diff| = {diff:.2e}  -> {'NO-OP OK' if diff < 1e-5 else 'BROKEN'}")

# 2) after moving the gate, the outputs differ
with torch.no_grad():
    m.backbone.loop_gate_param.data[1:] = 0.5
    c, _ = m(ids)
diff2 = (c-a).abs().max().item()
print(f"after setting gate to 0.5: max|diff| = {diff2:.2e} -> {'EFFECT OK' if diff2 > 1e-3 else 'NO EFFECT?'}")

# 3) gradient flows into the gate
m.train()
out, _ = m(ids)
out.sum().backward()
g = m.backbone.loop_gate_param.grad
print(f"gate grad (rounds 1..T-1): {[round(float(x),4) for x in g]}")
print(f"param count: {m.backbone.loop_gate_param.shape[0]} (expect T-1 = 3)")
