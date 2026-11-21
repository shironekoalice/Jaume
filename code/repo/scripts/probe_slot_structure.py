"""Distinguish "the state has no information" from "the information lives in the
slot STRUCTURE and mean-pooling discards it".

The earlier probe reported  mean(state_A) == pooled_A  (max|diff| = 0). That
only proves the two tensors are the same representation -- it says nothing about
whether the 8 individual slots carry information that the mean destroys. The
review correctly flagged this as an over-reach, so it is measured directly here.

Four readouts of the SAME frozen state, increasing in what they are allowed to use:

  1. mean        : mean over slots (= what the model currently outputs)
  2. slot_select : one slot, argmax by a probe's own choice           (1 x 512)
  3. flat        : all 8 slots concatenated                          (8 x 512)
  4. slot_wise   : a probe per slot, take the best                   (8 x 512)

If (1) is low but (3)/(4) are clearly higher, the information IS in the slots
and mean-pooling is the bottleneck. If all are equally low, the state itself
does not hold the fact.

Targets: chunk A's two numeric values (regression, R^2), plus a categorical
target (which protocol holds the smaller value) as a second view.

Usage:
    python scripts/probe_slot_structure.py --arm B5 --n 1500
"""
import os as _os
PROJECT_ROOT = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import argparse
import json
import os
import re
import sys

import torch
import torch.nn as nn

REPO = fr"{PROJECT_ROOT}\code\repo"
sys.path.insert(0, REPO)
from jaume.inference.encode import JaumeEmbedding  # noqa: E402

ARMS = {
    "B5": fr"{PROJECT_ROOT}\output\mem\B5-recurrent-s1\final",
    "B6": fr"{PROJECT_ROOT}\output\mem\B6-recurrent-gate-s1\final",
}


def split_chunks(it):
    def has_num(s):
        return any(ch.isdigit() for ch in s)
    a, b = it["chunks"][0], it["chunks"][1]
    if has_num(a) and not has_num(b):
        return a, b
    if has_num(b) and not has_num(a):
        return b, a
    return (a, b) if len(a) >= len(b) else (b, a)


def load_items(path, n):
    out = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        out.append(json.loads(line))
        if len(out) >= n:
            break
    return out


def extract_states(model, items):
    """Return stacked slot states [N, slots, dim] after reading chunk A."""
    states = []
    for it in items:
        a, _ = split_chunks(it)
        with torch.no_grad():
            e = model.tokenizer([a], padding=True, truncation=True,
                                max_length=256, return_tensors="pt").to(model.device)
            _, _, s = model.model.encode(e["input_ids"], e["attention_mask"], None)
        states.append(s.float().cpu())
    return torch.cat(states, dim=0)


def targets(items):
    y1, y2, ycat = [], [], []
    for it in items:
        a, _ = split_chunks(it)
        nums = [int(x) for x in re.findall(r"\d+", a)]
        nums = (nums + [0, 0])[:2]
        y1.append(nums[0] / 240.0)
        y2.append(nums[1] / 240.0)
        ycat.append(1.0 if nums[0] < nums[1] else 0.0)
    return (torch.tensor(y1).unsqueeze(-1),
            torch.tensor(y2).unsqueeze(-1),
            torch.tensor(ycat).unsqueeze(-1))


def r2_probe(X, y, nonlinear=False, seed=0, epochs=300, lr=1e-2):
    torch.manual_seed(seed)
    n = X.size(0)
    perm = torch.randperm(n)
    cut = int(n * 0.8)
    tr, te = perm[:cut], perm[cut:]
    Xtr, ytr = X[tr], y[tr]
    Xte, yte = X[te], y[te]
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp(min=1e-6)
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    net = (nn.Sequential(nn.Linear(X.size(1), 64), nn.GELU(), nn.Linear(64, 1))
           if nonlinear else nn.Linear(X.size(1), 1))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        lossf(net(Xtr), ytr).backward()
        opt.step()
    with torch.no_grad():
        pred = net(Xte)
        ss_res = ((yte - pred) ** 2).sum()
        ss_tot = ((yte - yte.mean()) ** 2).sum().clamp(min=1e-8)
        return 1.0 - float(ss_res / ss_tot)


def slot_wise_best(S, y, seed=0):
    """Probe each slot separately and return the best R^2.

    This answers: does ANY single slot carry the fact? Mean-pooling would bury
    a fact that lives in one slot, so a high value here is evidence that the
    structure holds information the mean destroys.
    """
    best = float("-inf")
    per = []
    for j in range(S.size(1)):
        r = r2_probe(S[:, j, :], y, seed=seed)
        per.append(round(r, 4))
        best = max(best, r)
    return best, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=fr"{PROJECT_ROOT}\data\synth_v3")
    ap.add_argument("--arm", default="B5")
    ap.add_argument("--n", type=int, default=1500)
    args = ap.parse_args()

    items = load_items(os.path.join(args.data, "test.jsonl"), args.n)
    model = JaumeEmbedding.from_pretrained(ARMS[args.arm])
    S = extract_states(model, items)          # [N, slots, dim]
    y1, y2, ycat = targets(items)
    N, K, D = S.shape
    print(f"[slot] arm={args.arm} N={N} slots={K} dim={D}")

    mean_rep = S.mean(dim=1)
    flat_rep = S.reshape(N, -1)

    print(f"\n{'readout':16s} {'dim':>6s} {'R2 v1':>8s} {'R2 v2':>8s} {'R2 cat':>8s}")
    res = {}

    def row(name, X):
        a = r2_probe(X, y1)
        b = r2_probe(X, y2)
        c = r2_probe(X, ycat)
        res[name] = {"dim": X.size(1), "v1": a, "v2": b, "cat": c}
        print(f"{name:16s} {X.size(1):6d} {a:8.4f} {b:8.4f} {c:8.4f}")

    row("mean", mean_rep)
    row("flat(8 slots)", flat_rep)

    best1, per1 = slot_wise_best(S, y1)
    best2, per2 = slot_wise_best(S, y2)
    res["slot_wise_best"] = {"v1": best1, "v2": best2, "per_slot_v1": per1,
                             "per_slot_v2": per2}
    print(f"{'slot_wise_best':16s} {D:6d} {best1:8.4f} {best2:8.4f} {'-':>8s}")
    print(f"  per-slot R2 (v1): {per1}")

    # how different are the slots from each other?
    sn = torch.nn.functional.normalize(S, dim=-1)
    gram = sn @ sn.transpose(1, 2)
    off = gram - torch.eye(K).unsqueeze(0)
    pw = off.sum(dim=(1, 2)) / (K * (K - 1))
    print(f"\n  slot pairwise cosine: mean={float(pw.mean()):.4f} "
          f"min={float(pw.min()):.4f} max={float(pw.max()):.4f}")
    # variance of the mean vs variance of individual slots
    print(f"  std(mean_rep)={float(mean_rep.std()):.4f}  "
          f"std(slots)={float(S.std()):.4f}  "
          f"ratio={float(mean_rep.std()/S.std()):.4f}")

    out = os.path.join(args.data, f"slot_structure_{args.arm}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"\n[slot] wrote {out}")
    print("\nREAD THIS AS:")
    print("  mean low, flat/slot_wise clearly higher -> the fact IS in the slots")
    print("       and mean-pooling discards it (B candidate justified)")
    print("  all equally low                         -> the state itself lacks it")


if __name__ == "__main__":
    main()
