"""D4-D8: is the gate able to decide how much history to keep?

Read-only diagnosis. Does not modify the model, does not train, does not
overwrite any checkpoint.

IMPORTANT correction made here
------------------------------
There are TWO independent gates in the real forward path:

  G1  LatentMemoryPooling.slot_pass(state_update="bounded")
        input  [slots ; read]  (2*hidden)
        output slots = norm((1-g)*slots + g*read)
      -> only exists in the R1b experiment

  G2  RecurrentMemoryPooling.gate = GatedSlotUpdate
        input  [prev_state ; new_slots]  (2*hidden)
        output carry = g*new_slots + (1-g)*prev_state
      -> this is the gate that produces the ACTUAL returned state for B6

The earlier D1/D3 diagnosis measured the output of slot_pass, which is NOT the
returned state: `RecurrentMemoryPooling.forward` computes
`new_slots = slot_pass(...)` then `carry = gate(memory_state, new_slots)` and
returns `carry`. So the earlier numbers described the pre-gate intermediate,
not the state that propagates. This script measures the correct object.

Checks
  D4  read distinguishability (evidence vs non-evidence blocks)
  D5  gate response: distribution of g, correlation with read novelty
  D6  update contribution: relative norm of history term vs new term
  D7  counterfactual: hold history, swap the current block; and vice versa
  D8  consistency: the measured path equals the real forward return value

Usage:
    python scripts/diagnose_gate.py --mode additive
"""
import os as _os
PROJECT_ROOT = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import argparse
import json
import os
import statistics
import sys

import torch
import torch.nn.functional as F

REPO = fr"{PROJECT_ROOT}\code\repo"
sys.path.insert(0, REPO)
from jaume.device import resolve_device, has_amp, make_scaler, empty_cache, max_memory_allocated, reset_peak_memory_stats, total_memory, seed_all
sys.path.insert(0, os.path.join(REPO, "scripts"))

import train_r1 as R  # noqa: E402


def encode_block(rm, text):
    e = rm.tokenizer([text], padding=True, truncation=True,
                     max_length=128, return_tensors="pt").to(rm.device)
    return e


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = sum((a - mx) ** 2 for a in xs) ** 0.5
    dy = sum((b - my) ** 2 for b in ys) ** 0.5
    return num / (dx * dy) if dx * dy > 1e-12 else float("nan")


def trace_paper(rm, row, max_blocks):
    """Walk one paper and record, per block: read, pre-gate slots, G2 gate
    values, and the returned state. Also verifies consistency with forward()."""
    base = rm.model
    pool = base.pooling
    core = pool.core
    blocks = row["blocks"][:max_blocks]

    recs = []
    state = None            # this is the RETURNED state (carry)
    with torch.no_grad():
        for i, b in enumerate(blocks):
            e = encode_block(rm, b)
            h, _ = base.backbone(e["input_ids"], e["attention_mask"])

            init = state if state is not None else None
            slots0 = core.slots.unsqueeze(0).expand(h.size(0), -1, -1).contiguous() \
                if init is None else init

            # replicate slot_pass but keep intermediates
            kv = core._kv(h)
            read, attn = core._cross_attend(slots0, h, e["attention_mask"], kv)
            new_slots = core._refine(slots0, read) if getattr(core, "state_update",
                                                              "additive") == "additive" \
                else slots0 + read

            # G2 gate (the one that produces the returned state)
            if state is not None and getattr(pool, "use_gate", False):
                g2 = torch.sigmoid(pool.gate.gate(
                    torch.cat([state, new_slots], dim=-1)))
                carry = g2 * new_slots + (1.0 - g2) * state
            else:
                g2 = None
                carry = new_slots

            # D8 consistency: compare against the real forward's return value
            pooled, _, state_fwd = base.encode(e["input_ids"], e["attention_mask"], state)
            d8 = float((carry - state_fwd).abs().max())

            recs.append({
                "i": i,
                "is_evidence": i == row["evidence_idx"],
                "read": read.detach().float().cpu(),
                "new_slots": new_slots.detach().float().cpu(),
                "state_prev": None if state is None else state.detach().float().cpu(),
                "state": carry.detach().float().cpu(),
                "g2_mean": None if g2 is None else float(g2.mean()),
                "g2_std": None if g2 is None else float(g2.std()),
                "true_g2": g2,
                "true_carry": carry,
                "d8": d8,
            })
            state = state_fwd
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="additive")
    ap.add_argument("--data", default=fr"{PROJECT_ROOT}\data\qasper_blocks")
    ap.add_argument("--blocks", type=int, default=12)
    ap.add_argument("--papers", type=int, default=4)
    ap.add_argument("--ckpt", default=fr"{PROJECT_ROOT}\output\mem\R1-smoke-additive")
    args = ap.parse_args()

    rows = R.trim(R.load(os.path.join(args.data, "val.jsonl")), args.blocks)
    rows = [r for r in rows if r["blocks"]][: args.papers]
    rm = R.make_model(args.ckpt, resolve_device(), args.mode)
    rm.model.eval()
    print(f"checkpoint={args.ckpt} mode={args.mode} papers={len(rows)} "
          f"blocks={args.blocks}")
    pool = rm.model.pooling
    print(f"pooling={type(pool).__name__} use_gate={getattr(pool,'use_gate',None)} "
          f"state_update={getattr(pool.core,'state_update',None)}")

    all_d4, all_g2, all_nov, all_d6, d8s = [], [], [], [], []
    ev_read_cos, non_read_cos = [], []

    for row in rows:
        recs = trace_paper(rm, row, args.blocks)
        d8s += [r["d8"] for r in recs]
        ev_idx = row["evidence_idx"]

        read_vecs = torch.cat([r["read"].mean(dim=1) for r in recs], dim=0)
        read_n = F.normalize(read_vecs, dim=-1)
        sim = read_n @ read_n.T
        k = sim.shape[0]

        for i in range(1, k):
            nov = 1.0 - float(sim[i, i - 1])          # novelty of block i
            all_nov.append(nov)
            if recs[i]["g2_mean"] is not None:
                all_g2.append(recs[i]["g2_mean"])
            # D4: is the read of the evidence block more/less similar to others?
            if i == ev_idx:
                ev_read_cos.append(float(
                    (sim[i].sum() - sim[i, i]) / max(1, k - 1)))
            else:
                non_read_cos.append(float(
                    (sim[i].sum() - sim[i, i]) / max(1, k - 1)))

        # D6: relative norm of the new-block contribution vs the carried state
        for i in range(1, k):
            sp = recs[i]["state_prev"]
            st = recs[i]["state"]
            delta = (st - sp).norm()
            all_d6.append(float(delta / (sp.norm() + 1e-8)))

        all_d4 += [1.0 - float(sim[i, i - 1]) for i in range(1, k)]

    print("\n=== D8 consistency (measured path vs real forward return) ===")
    print(f"  max |diff| over all blocks/papers = {max(d8s):.3e}  "
          f"{'PASS' if max(d8s) < 1e-4 else 'FAIL'}")

    print("\n=== D4 read distinguishability ===")
    print(f"  adjacent-read novelty (1-cos): mean={statistics.mean(all_d4):.4f} "
          f"min={min(all_d4):.4f} max={max(all_d4):.4f}")
    if ev_read_cos and non_read_cos:
        print(f"  evidence blocks   mean read cos to others = "
              f"{statistics.mean(ev_read_cos):.4f} (n={len(ev_read_cos)})")
        print(f"  non-evidence      mean read cos to others = "
              f"{statistics.mean(non_read_cos):.4f} (n={len(non_read_cos)})")

    print("\n=== D5 gate response (G2, the gate that produces the state) ===")
    if all_g2:
        print(f"  g2 mean over blocks = {statistics.mean(all_g2):.4f} "
              f"sd={statistics.stdev(all_g2) if len(all_g2)>1 else 0:.4f} "
              f"min={min(all_g2):.4f} max={max(all_g2):.4f}")
        print(f"  distance from 0.5  = {abs(statistics.mean(all_g2)-0.5):.4f}")
        m = min(len(all_g2), len(all_nov))
        r = pearson(all_g2[:m], all_nov[:m])
        print(f"  corr(g2, read novelty) = {r:+.4f}  (n={m})")
    else:
        print("  no gate (use_gate=False or no previous state)")

    print("\n=== D6 update contribution ===")
    print(f"  |state_i - state_{{i-1}}| / |state_{{i-1}}| : "
          f"mean={statistics.mean(all_d6):.4f} min={min(all_d6):.4f} "
          f"max={max(all_d6):.4f}")

    print("\n=== D7 counterfactual sensitivity ===")
    row = rows[0]
    recs = trace_paper(rm, row, args.blocks)
    i = min(6, len(recs) - 1)
    base_rec = recs[i]

    # (a) fix history, swap the CURRENT block with a different block
    rm2 = R.make_model(args.ckpt, resolve_device(), args.mode)
    rm2.model.eval()
    row_swapped = dict(row)
    bl = list(row["blocks"][: args.blocks])
    j = 1 if i != 1 else 2
    bl[i], bl[j] = bl[j], bl[i]
    row_swapped["blocks"] = bl
    recs_sw = trace_paper(rm2, row_swapped, args.blocks)
    d_state = float((recs_sw[i]["state"] - base_rec["state"]).abs().mean())
    d_read = float((recs_sw[i]["read"] - base_rec["read"]).abs().mean())
    print(f"  swap current block #{i}<->#{j}:")
    print(f"     |Δ read| = {d_read:.5f}   |Δ state| = {d_state:.5f}")
    if base_rec["g2_mean"] is not None:
        print(f"     g2 {base_rec['g2_mean']:.4f} -> {recs_sw[i]['g2_mean']:.4f}")

    print("\nINTERPRETATION (pre-registered decision rules):")
    print("  read distinguishable + gate unresponsive -> inspect gate INPUT design")
    print("  gate responsive but new-block share tiny  -> inspect update formula")
    print("  read NOT distinguishable                  -> go back to slot_pass read")
    print("  all normal but task still fails           -> the update rule is not")
    print("                                               the root cause")


if __name__ == "__main__":
    main()
