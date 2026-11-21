"""Verify the batched block runner matches the per-block reference.

The batched version advances all papers in a batch in lock-step. If the state
bookkeeping (index_select / index_copy) is wrong, representations silently mix
across papers and every downstream number is meaningless. This compares against
a straightforward one-block-at-a-time implementation.
"""
import os as _os
PROJECT_ROOT = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import json
import os
import sys

import torch
import torch.nn.functional as F

REPO = fr"{PROJECT_ROOT}\code\repo"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from jaume.inference.encode import JaumeEmbedding  # noqa: E402
import qasper_batch as QB  # noqa: E402


def reference(rm, rows, pass_state, max_length=256, max_blocks=8):
    out = []
    for r in rows:
        reps, state = [], None
        for b in r["blocks"][:max_blocks]:
            e = rm.tokenizer([b], padding=True, truncation=True,
                             max_length=max_length, return_tensors="pt").to(rm.device)
            with torch.no_grad():
                pooled, _, new_state = rm.model.encode(
                    e["input_ids"], e["attention_mask"], state if pass_state else None)
            v = new_state.mean(dim=1) if pass_state else pooled
            reps.append(F.normalize(rm.model.projection(v), dim=-1))
            if pass_state:
                state = new_state
        out.append(torch.cat(reps, dim=0))
    return out


def main():
    rm = JaumeEmbedding.from_pretrained(fr"{PROJECT_ROOT}\output\mem\P1b-s0")
    rm.model.eval()
    rows = [json.loads(l) for l in
            open(fr"{PROJECT_ROOT}\data\qasper_blocks\test.jsonl", encoding="utf-8")
            if l.strip()][:4]

    for pass_state in (True, False):
        with torch.no_grad():
            batched = QB.run_batch_states(rm, rows, pass_state, max_blocks=8)
            ref = reference(rm, rows, pass_state, max_blocks=8)
        print(f"pass_state={pass_state}")
        for i, (b, r) in enumerate(zip(batched, ref)):
            d = (b - r).abs().max()
            print(f"  paper {i}: shape {tuple(b.shape)} vs {tuple(r.shape)} "
                  f"max|diff| = {float(d):.6f}")
    print("\nA correct implementation shows max|diff| ~ 0 (<1e-4).")


if __name__ == "__main__":
    main()
