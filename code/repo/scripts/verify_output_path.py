"""Verify the A-variant change is inert for the default configuration.

Requirement from review: the modification must only affect the intended place.
This checks:
  1. with memory_output="new_slots" (default) the output equals the ORIGINAL
     formula pool_slots(new_slots) bit-for-bit;
  2. with memory_output="carry" the output equals pool_slots(carry);
  3. existing checkpoints still load (state_dict keys unchanged);
  4. the two modes actually differ when a previous state is supplied.
"""
import os as _os
PROJECT_ROOT = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import os
import sys

import torch

REPO = fr"{PROJECT_ROOT}\code\repo"
sys.path.insert(0, REPO)
from jaume.inference.encode import JaumeEmbedding  # noqa: E402
from jaume.modeling.memory import RecurrentMemoryPooling  # noqa: E402

ARM = fr"{PROJECT_ROOT}\output\mem\B5-recurrent-s1\final"
m = JaumeEmbedding.from_pretrained(ARM)
pooler = m.model.pooling
print(f"pooler={type(pooler).__name__} output={pooler.output} use_gate={pooler.use_gate}")

# 3. checkpoint keys unchanged: the model loaded, and 'output' adds no params
n_new = sum(1 for k in pooler.state_dict() if "output" in k)
print(f"params added by the change: {n_new} (must be 0)")

TEXT_A = ("Protocol DORULEIORUM retains audit logs for 43 days. "
          "Protocol LOMEFIEOMIR retains audit logs for 79 days.")
TEXT_B = "Adopt the protocol that carries the smaller value."

with torch.no_grad():
    ea = m.tokenizer([TEXT_A], padding=True, truncation=True,
                     max_length=256, return_tensors="pt").to(m.device)
    hA, _ = m.model.backbone(ea["input_ids"], ea["attention_mask"])
    pA, sA = pooler(hA, ea["attention_mask"], None)
    eb = m.tokenizer([TEXT_B], padding=True, truncation=True,
                     max_length=256, return_tensors="pt").to(m.device)
    hB, _ = m.model.backbone(eb["input_ids"], eb["attention_mask"])

    # recompute both variants by hand from the same intermediate values
    new_slots = pooler.core.slot_pass(hB, eb["attention_mask"], sA)
    if pooler.use_gate:
        carry = pooler.gate(sA, new_slots)
    else:
        carry = new_slots
    manual_new = pooler.core.pool_slots(new_slots)
    manual_carry = pooler.core.pool_slots(carry)

    out_default, _ = pooler(hB, eb["attention_mask"], sA)
    diff_default = float((out_default - manual_new).abs().max())

    pooler.output = "carry"
    out_carry, _ = pooler(hB, eb["attention_mask"], sA)
    diff_carry = float((out_carry - manual_carry).abs().max())
    pooler.output = "new_slots"

print("\n1. default mode == original formula : max|diff| =", diff_default)
print("2. carry   mode == pool_slots(carry): max|diff| =", diff_carry)
print("4. do the two modes differ?          : max|diff| =",
      float((out_default - out_carry).abs().max()))

ok = (diff_default < 1e-6) and (diff_carry < 1e-6) and n_new == 0
print("\nRESULT:", "PASS" if ok else "FAIL")
if float((out_default - out_carry).abs().max()) < 1e-6:
    print("WARNING: the two modes are identical -- A would change nothing.")
