"""Smoke test for the looped LM: shapes, loop accounting, and warm start."""
import os, sys, torch
sys.path.insert(0, "code/repo")
from jaume.modeling.config import JaumeConfig
from jaume.modeling.loop_lm import LoopLM
from jaume.modeling.checkpoint import load_state_dict

cfg = JaumeConfig.load("code/repo/configs/jaume-350m-loop.json")
cfg.vocab_size = 1000  # tiny for the smoke test
cfg.hidden_size = 64
cfg.intermediate_size = 128
cfg.num_layers = 4
cfg.num_attention_heads = 4
cfg.num_key_value_heads = 2
cfg.expert_num = 4
cfg.topk = 2
cfg.embedding_dim = 64
cfg.matryoshka_dims = (16, 32, 48, 64)

for form, nl, nlayers in (("full", 1, 0), ("full", 3, 0), ("tail", 3, 2)):
    cfg.num_loops = nl
    cfg.loop_form = form
    cfg.loop_layers = nlayers
    m = LoopLM(cfg)
    ids = torch.randint(0, 1000, (2, 16))
    logits, aux, ls = m(ids, return_loop_states=True)
    print(f"form={form} loops={nl}: logits={tuple(logits.shape)} "
          f"aux={None if aux is None else float(aux):.4f} "
          f"loop_states={len(ls)} x {tuple(ls[0].shape)} "
          f"eff_depth={m.backbone.effective_depth} "
          f"params={m.num_parameters()/1e6:.3f}M")
    assert logits.shape == (2, 16, 1000)
    assert len(ls) == nl

# warm start from the real causal checkpoint
cfg2 = JaumeConfig.load("code/repo/configs/jaume-350m-loop.json")
cfg2.num_loops = 1
cfg2.vocab_size = 151647  # the checkpoint was trained with [MASK]
m2 = LoopLM(cfg2)
sd = load_state_dict(".mscache/LightNeko/Jaume/output/350m-causal-mlm/final/pytorch_model.bin")
src = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
tgt = m2.backbone.state_dict()
loaded = [k for k, v in src.items() if k in tgt and tgt[k].shape == v.shape]
skipped = [k for k in src if k not in loaded]
for k in loaded:
    tgt[k] = src[k]
m2.backbone.load_state_dict(tgt)
print(f"warm start: loaded {len(loaded)} tensors, skipped {len(skipped)}")
print("  skipped sample:", skipped[:6])
print("  lm_head shape:", tuple(m2.lm_head.weight.shape))
