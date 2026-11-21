"""Step 0 of the state probe: record the REAL tensor shapes and state flow.

Before writing any diagnostic classifier we must know exactly what comes out of
`encode(...)` for the recurrent arms: is the state [B, slots, dim] or [B, dim],
what does `slot_pass` return, and does `carry` actually differ from the freshly
read slots in B5?

Nothing here changes the model. It only reads weights and prints shapes.
"""
import os as _os
PROJECT_ROOT = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import json
import os
import sys

import torch

REPO = fr"{PROJECT_ROOT}\code\repo"
sys.path.insert(0, REPO)
from jaume.inference.encode import JaumeEmbedding  # noqa: E402

ARMS = {
    "B5": fr"{PROJECT_ROOT}\output\mem\B5-recurrent-s1\final",
    "B6": fr"{PROJECT_ROOT}\output\mem\B6-recurrent-gate-s1\final",
    "B3": fr"{PROJECT_ROOT}\output\mem\B3-latent-s1\final",
}

TEXT_A = ("Protocol DORULEIORUM retains audit logs for 43 days. "
          "Protocol LOMEFIEOMIR retains audit logs for 79 days.")
TEXT_B = "Adopt the protocol that carries the smaller value."

for name, path in ARMS.items():
    print(f"\n===== {name}  {path} =====")
    if not os.path.exists(path):
        print("  missing")
        continue
    m = JaumeEmbedding.from_pretrained(path)
    cfg = m.model.config
    pooler = m.model.pooling
    print(f"  pooling={cfg.pooling} num_slots={getattr(cfg,'num_slots',None)} "
          f"slot_iterations={getattr(cfg,'slot_iterations',None)} "
          f"slot_gru={getattr(cfg,'slot_gru',None)} "
          f"slot_competition={getattr(cfg,'slot_competition',None)} "
          f"slot_pool={getattr(cfg,'slot_pool',None)}")
    print(f"  pooler class = {type(pooler).__name__}")

    # does the pooler expose use_gate?
    print(f"  pooler.use_gate = {getattr(pooler,'use_gate','n/a')}")
    print(f"  pooler.num_slots = {getattr(pooler,'num_slots','n/a')}")

    with torch.no_grad():
        e = m.tokenizer([TEXT_A], padding=True, truncation=True,
                        max_length=256, return_tensors="pt").to(m.device)
        pooled, aux, state = m.model.encode(e["input_ids"],
                                            e["attention_mask"], None)
        print(f"  encode(chunkA, state=None) -> ")
        print(f"      pooled.shape = {tuple(pooled.shape)}")
        print(f"      state.shape  = {None if state is None else tuple(state.shape)}")
        print(f"      state.dtype  = {None if state is None else state.dtype}")

        e2 = m.tokenizer([TEXT_B], padding=True, truncation=True,
                         max_length=256, return_tensors="pt").to(m.device)
        pooled2, _, state2 = m.model.encode(e2["input_ids"],
                                            e2["attention_mask"], state)
        print(f"  encode(chunkB, state=s_A) ->")
        print(f"      pooled.shape = {tuple(pooled2.shape)}")
        print(f"      state.shape  = {None if state2 is None else tuple(state2.shape)}")

        if state is not None and state2 is not None:
            cos = torch.nn.functional.cosine_similarity(
                state.reshape(-1, state.shape[-1]),
                state2.reshape(-1, state2.shape[-1]), dim=-1).mean()
            print(f"  mean cosine(s_A, s_AB) = {float(cos):.4f}")
            print(f"  L2 norm s_A={float(state.norm()):.2f} "
                  f"s_AB={float(state2.norm()):.2f}")

        # per-slot statistics: are the 8 slots distinct?
        if state is not None and state.dim() == 3 and state.size(1) > 1:
            s = state[0]                                  # [slots, dim]
            sn = torch.nn.functional.normalize(s, dim=-1)
            gram = sn @ sn.T
            off = gram - torch.eye(s.size(0), device=s.device)
            print(f"  slots={s.size(0)} dim={s.size(1)}")
            print(f"  slot pairwise mean cosine (off-diag) = {float(off.sum()/(s.size(0)*(s.size(0)-1))):.4f}")
            print(f"  slot norm min/max = {float(s.norm(dim=-1).min()):.2f}/"
                  f"{float(s.norm(dim=-1).max()):.2f}")
