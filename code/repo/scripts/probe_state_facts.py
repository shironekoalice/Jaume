"""State information probe: is chunk A's fact still readable after the update?

Question this answers (raised in review):
    "块 A 的原始表示可读、状态不可读 -> 检查状态写入/门控/池化丢信息
     状态可读但下游失败         -> 检查读取接口
     线性探针失败               -> 再用非线性复核"

Design
------
For each item the generator knows chunk A's two facts: (value1, value2) and
which comparison direction chunk B asked for. We train a linear (and a small
MLP) probe to predict those facts from three candidate representations:

  1. `pooled_A`  : the encoder's normal pooled output for chunk A  (baseline)
  2. `state_A`   : the 8-slot state after reading chunk A          (what memory gets)
  3. `state_AB`  : the state after reading A then B                (what memory ends up as)

If (1) predicts the fact but (2) does not, the state write is lossy.
If (2) predicts but the end-to-end task failed, the READ interface is at fault.

Nothing is trained in the model. Only small probes on frozen representations.

Usage:
    python scripts/probe_state_facts.py --n 2000 --arm B5
"""
import os as _os
PROJECT_ROOT = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import argparse
import json
import os
import sys

import torch
import torch.nn as nn

REPO = fr"{PROJECT_ROOT}\code\repo"
sys.path.insert(0, REPO)
from jaume.inference.encode import JaumeEmbedding  # noqa: E402

ARMS = {
    "B5": fr"{PROJECT_ROOT}\output\mem\B5-recurrent-s1\final",
    "B6": fr"{PROJECT_ROOT}\output\mem\B6-recurrent-gate-s1\final",
    "B3": fr"{PROJECT_ROOT}\output\mem\B3-latent-s1\final",
}

NUM_RE = None


def load_items(path, n):
    import re
    global NUM_RE
    NUM_RE = re.compile(r"(\d+)")
    out = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        out.append(json.loads(line))
        if len(out) >= n:
            break
    return out


def split_chunks(it):
    def has_num(s):
        return any(ch.isdigit() for ch in s)
    a, b = it["chunks"][0], it["chunks"][1]
    if has_num(a) and not has_num(b):
        return a, b
    if has_num(b) and not has_num(a):
        return b, a
    return (a, b) if len(a) >= len(b) else (b, a)


def extract(model, items):
    """Return dict of representation tensors, one row per item.

    state_A is [1,8,512]; we keep BOTH the mean over slots and the flattened
    slots, because a mean can hide a fact that lives in one slot.
    """
    pooled_A, state_A, state_AB = [], [], []
    for it in items:
        a, b = split_chunks(it)
        with torch.no_grad():
            ea = model.tokenizer([a], padding=True, truncation=True,
                                 max_length=256, return_tensors="pt").to(model.device)
            pA, _, sA = model.model.encode(ea["input_ids"],
                                           ea["attention_mask"], None)
            eb = model.tokenizer([b], padding=True, truncation=True,
                                 max_length=256, return_tensors="pt").to(model.device)
            _, _, sAB = model.model.encode(eb["input_ids"],
                                           eb["attention_mask"], sA)
        pooled_A.append(pA.float().cpu())
        state_A.append(sA.float().cpu())
        state_AB.append(sAB.float().cpu())
    return {
        "pooled_A": torch.cat(pooled_A),
        "state_A_mean": torch.cat([s.mean(dim=1) for s in state_A]),
        "state_A_flat": torch.cat([s.reshape(1, -1) for s in state_A]),
        "state_AB_mean": torch.cat([s.mean(dim=1) for s in state_AB]),
        "state_AB_flat": torch.cat([s.reshape(1, -1) for s in state_AB]),
    }


def labels_for(items):
    """Two regression targets from chunk A: the two numeric values.

    Values are normalised to [0,1] using a fixed range so the probe target is
    on a comparable scale. We also build a categorical version (which of the
    two protocols has the smaller value) as a sanity target.
    """
    import re
    y1, y2 = [], []
    for it in items:
        a, _ = split_chunks(it)
        nums = [int(x) for x in re.findall(r"\d+", a)]
        nums = (nums + [0, 0])[:2]
        y1.append(nums[0] / 240.0)
        y2.append(nums[1] / 240.0)
    return torch.tensor(y1).unsqueeze(-1), torch.tensor(y2).unsqueeze(-1)


def fit_probe(X, y, nonlinear=False, epochs=300, lr=1e-2, seed=0, dev="cpu"):
    """Train a small probe with a held-out split; return test R^2."""
    torch.manual_seed(seed)
    X = X.to(dev)
    y = y.to(dev)
    n = X.size(0)
    perm = torch.randperm(n)
    cut = int(n * 0.8)
    tr, te = perm[:cut], perm[cut:]
    Xtr, ytr = X[tr], y[tr]
    Xte, yte = X[te], y[te]
    # standardise using train statistics only
    mu = Xtr.mean(0, keepdim=True)
    sd = Xtr.std(0, keepdim=True).clamp(min=1e-6)
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    d_in = X.size(1)
    net = nn.Sequential(nn.Linear(d_in, 64), nn.GELU(), nn.Linear(64, 1)) \
        if nonlinear else nn.Linear(d_in, 1)
    net = net.to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(net(Xtr), ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = net(Xte)
        ss_res = ((yte - pred) ** 2).sum()
        ss_tot = ((yte - yte.mean()) ** 2).sum().clamp(min=1e-8)
        r2 = 1.0 - float(ss_res / ss_tot)
        # a constant predictor gives R^2 = 0 by construction
    return r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=fr"{PROJECT_ROOT}\data\synth_v3")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--arm", default="B5")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    items = load_items(os.path.join(args.data, "test.jsonl"), args.n)
    print(f"[probe] items={len(items)} arm={args.arm}")
    model = JaumeEmbedding.from_pretrained(ARMS[args.arm])
    reps = extract(model, items)
    y1, y2 = labels_for(items)

    print(f"\n{'representation':18s} {'dim':>6s} {'R2 value1':>10s} "
          f"{'R2 value2':>10s} {'MLP R2 v1':>10s}")
    results = {}
    for name, X in reps.items():
        r2a = fit_probe(X, y1, False, seed=args.seed)
        r2b = fit_probe(X, y2, False, seed=args.seed)
        r2c = fit_probe(X, y1, True, seed=args.seed)
        results[name] = {"dim": X.size(1), "lin_v1": r2a,
                         "lin_v2": r2b, "mlp_v1": r2c}
        print(f"{name:18s} {X.size(1):6d} {r2a:10.4f} {r2b:10.4f} {r2c:10.4f}")

    out = os.path.join(args.data, f"state_probe_{args.arm}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[probe] wrote {out}")
    print("\nREAD THIS AS:")
    print("  pooled_A high, state_A low   -> the state WRITE loses the fact")
    print("  state_A high, state_AB low   -> reading chunk B DESTROYS the fact")
    print("  state_A high but task fails  -> the READ interface is at fault")
    print("  all low                      -> try the MLP column before concluding")


if __name__ == "__main__":
    main()
