"""P0 audit: do the five strategies actually execute different code paths?

The concern (raised in review): on non-recurrent arms, `reset` and `carried`
may fall into the SAME branch, in which case their equality across 25 cells is
a code identity rather than evidence about the model. This script traces which
branch each (arm, strategy) pair takes by instrumenting _doc_vector.
"""
import os as _os
PROJECT_ROOT = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import json
import os
import sys

sys.path.insert(0, fr"{PROJECT_ROOT}\code\repo")

import torch  # noqa: E402

sys.path.insert(0, fr"{PROJECT_ROOT}\code\repo\scripts")
import eval_memory_crossinput as emc  # noqa: E402
from jaume.inference.encode import JaumeEmbedding  # noqa: E402

ARMS = [fr"{PROJECT_ROOT}\output\mem\B0-mean-s1\final",
        fr"{PROJECT_ROOT}\output\mem\B3-latent-s1\final",
        fr"{PROJECT_ROOT}\output\mem\B6-recurrent-gate-s1\final"]

item = {"chunks": ["alpha fact one.", "filler text here.", "beta fact two."],
        "query": "which one?", "candidates": ["beta fact two.", "alpha fact one."],
        "gold": 0}

orig_encode = emc._encode

for ARM in ARMS:
    model = JaumeEmbedding.from_pretrained(ARM)
    cfg = model.model.config
    rec = cfg.pooling in emc.RECURRENT_ARMS
    print(f"=== {os.path.basename(os.path.dirname(ARM))}  "
          f"pooling={cfg.pooling} recurrent={rec} ===")

    for st in emc.STRATEGIES:
        hits = {"encode": 0, "texts": 0, "stateful": 0}

        def traced_encode(m, texts):
            hits["encode"] += 1
            hits["texts"] = len(texts)
            return torch.zeros(len(texts), cfg.embedding_dim)

        emc._encode = traced_encode
        try:
            v = emc._doc_vector(model, item, st)
            print(f"  {st:14s} batch_encode_calls={hits['encode']} "
                  f"texts={hits['texts']}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {st:14s} raised {type(exc).__name__}: {str(exc)[:70]}")
        finally:
            emc._encode = orig_encode
    print()
