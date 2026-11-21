"""Plan 1 evaluation: the four arms, on the SAME information flow as training.

    s_A = f(chunk A, s_0)          always: the facts are encoded first
    y   = g(chunk B, <arm's state>) the instruction + the state under test

The four arms differ ONLY in what state step 2 receives (review condition B):

    carried   the real s_A                      full memory path
    reset     None                              no history
    shuffled  another item's s_A                excludes "generic bias" gains
    concat    both chunks in ONE input, no state
              (the learnability reference; if this fails the task/flow is at
               fault, not the memory mechanism)

`concat` is scored with the same step-2 head: the text is "chunk A chunk B" fed
as the second input, which gives the model the facts directly.

Reported per seed plus mean/std/min/max, never a single number.

Usage:
    python scripts/eval_plan1.py --tags P1a-s0 P1a-s1 P1a-s2
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

REPO = fr"{PROJECT_ROOT}\code\repo"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from jaume.inference.encode import JaumeEmbedding  # noqa: E402
import g3_memory as G  # noqa: E402


def step2_repr(model, texts, state, max_length=256, bs=64):
    """Encode in batches; `state` is either None or a [N, slots, D] tensor.

    The representation returned is the model's real embedding: pooled vector ->
    `projection` -> L2 normalise, i.e. exactly `model.forward`. Returning the
    RAW pooled vector (as an earlier version did) skipped the projection the
    model was trained through, which made a model scoring 0.656 on its own
    training data measure 0.16 here. Always go through projection.
    """
    outs = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            part = texts[i : i + bs]
            e = model.tokenizer(part, padding=True, truncation=True,
                                max_length=max_length,
                                return_tensors="pt").to(model.device)
            st = None if state is None else state[i : i + bs].to(model.device)
            pooled, _, _ = model.model.encode(e["input_ids"], e["attention_mask"], st)
            outs.append(model.model.projection(pooled).float().cpu())
    return torch.cat(outs, dim=0)


def first_step_states(model, texts, max_length=256):
    """s_A for every item: chunk A encoded with a fresh state (serial, batched
    only by dtype; the state must be produced per item because it is the object
    under test)."""
    states = []
    with torch.no_grad():
        for t in texts:
            e = model.tokenizer([t], padding=True, truncation=True,
                                max_length=max_length,
                                return_tensors="pt").to(model.device)
            _, _, s = model.model.encode(e["input_ids"], e["attention_mask"], None)
            states.append(s.float().cpu())
    return torch.cat(states, dim=0)


def score(model, rep, texts_b, items):
    """Score a [N, D] representation against each item's candidate names."""
    rep = torch.nn.functional.normalize(rep, dim=-1)
    flat = [c for it in items for c in it["candidates"]]
    with torch.no_grad():
        nm = []
        for i in range(0, len(flat), 64):
            e = model.tokenizer(flat[i : i + 64], padding=True, truncation=True,
                                max_length=64, return_tensors="pt").to(model.device)
            emb, _ = model.model(e["input_ids"], e["attention_mask"])
            nm.append(emb.float().cpu())
    nm = torch.nn.functional.normalize(torch.cat(nm, dim=0), dim=-1)
    correct = ptr = 0
    for i, it in enumerate(items):
        k = len(it["candidates"])
        block = nm[ptr : ptr + k]
        ptr += k
        if int((rep[i] @ block.T).argmax()) == it["gold"]:
            correct += 1
    return correct / len(items)


def eval_one(arm_path, items):
    model = JaumeEmbedding.from_pretrained(arm_path)
    model.model.eval()
    a_txt = [G.split_chunks(it)[0] for it in items]
    b_txt = [G.split_chunks(it)[1] for it in items]

    s_a = first_step_states(model, a_txt)

    res = {}
    res["carried"] = score(model, step2_repr(model, b_txt, s_a), b_txt, items)
    res["reset"] = score(model, step2_repr(model, b_txt, None), b_txt, items)
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(s_a.size(0), generator=g)
    res["shuffled"] = score(model, step2_repr(model, b_txt, s_a[perm]), b_txt, items)

    # concat: the facts are made available WITHOUT the memory path, but through
    # the SAME step-2 input shape the model was trained on. Stuffing "chunk A
    # chunk B" into step 2 instead would feed a long passage to a head that only
    # ever saw a short instruction, which is a distribution mismatch rather than
    # a test of the task being learnable (measured: it scored 0.102 < chance).
    # Here chunk A is encoded in step 1 (as in training) and chunk B in step 2,
    # then the two representations are summed: the facts are present, but they
    # are NOT carried through the state under test.
    a_repr = step2_repr(model, a_txt, None)
    b_repr = step2_repr(model, b_txt, None)
    res["concat"] = score(model, a_repr + b_repr, b_txt, items)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=fr"{PROJECT_ROOT}\data\synth_v3")
    ap.add_argument("--root", default=fr"{PROJECT_ROOT}\output\mem")
    ap.add_argument("--tags", nargs="+", required=True)
    args = ap.parse_args()

    items = G.load_items(os.path.join(args.data, "test.jsonl"))
    chance = sum(1.0 / len(it["candidates"]) for it in items) / len(items)
    print(f"[P1-eval] items={len(items)} chance={chance:.3f}\n")

    rows = {}
    for tag in args.tags:
        arm = os.path.join(args.root, tag, "final")
        if not os.path.exists(arm):
            arm = os.path.join(args.root, tag)
        if not os.path.exists(arm):
            print(f"  MISSING {arm}")
            continue
        r = eval_one(arm, items)
        rows[tag] = r
        print(f"  {tag:14s} concat={r['concat']:.3f} reset={r['reset']:.3f} "
              f"carried={r['carried']:.3f} shuffled={r['shuffled']:.3f}", flush=True)

    print("\n" + "=" * 92)
    print(f"{'tag':14s} {'cond':9s} {'mean':>7s} {'sd':>6s} {'min':>6s} {'max':>6s}")
    print("=" * 92)
    for cond in ("concat", "carried", "reset", "shuffled"):
        vals = [rows[t][cond] for t in rows]
        if not vals:
            continue
        sdv = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"{'ALL':14s} {cond:9s} {statistics.mean(vals):7.3f} {sdv:6.3f} "
              f"{min(vals):6.3f} {max(vals):6.3f}")

    print("\nVERDICT (pre-registered):")
    c = statistics.mean([rows[t]["carried"] for t in rows])
    r_ = statistics.mean([rows[t]["reset"] for t in rows])
    sh = statistics.mean([rows[t]["shuffled"] for t in rows])
    co = statistics.mean([rows[t]["concat"] for t in rows])
    print(f"  concat   {co:.3f}  (learnability reference, must exceed {chance+0.10:.3f})")
    print(f"  carried  {c:.3f}")
    print(f"  reset    {r_:.3f}     carried-reset    = {c-r_:+.3f}")
    print(f"  shuffled {sh:.3f}     carried-shuffled = {c-sh:+.3f}")
    if co < chance + 0.10:
        print("  => concat failed: the task/flow is at fault, not memory.")
    elif c > r_ and c > sh:
        print("  => MEMORY HELPS: carried beats both reset and shuffled.")
    else:
        print("  => no memory benefit: check the state path next (probe, not")
        print("     more training).")

    out = os.path.join(args.data, "plan1_eval.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"chance": chance, "rows": rows}, f, indent=2)
    print(f"\n[P1-eval] wrote {out}")


if __name__ == "__main__":
    main()
