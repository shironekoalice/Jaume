"""T0 screen: can a frozen backbone + a trained head beat the floor on SciFact?

This is the gate that must pass before any of the TTC work is meaningful. If a
frozen backbone cannot tell NEI / SUPPORT / CONTRADICT apart on held-out rows,
then no amount of layer-path search on top of it can either -- there is no
signal to route.

Protocol (identical for every backbone, so the comparison is controlled):
  * the backbone is FROZEN; only a linear head is trained;
  * input = stage1 + stage2 + claim in one sequence, truncated at --max-length;
  * metric = macro-F1 on the held-out test split;
  * the floor is the constant-predictor macro-F1 (0.196 for this split,
    printed by check_scifact_data.py);
  * report mean and range across seeds, never a single number.

Usage:
    python scripts/t0_screen.py --model <checkpoint dir> --seed 0 1 2
    python scripts/t0_screen.py --model <dir> --out results/t0-mymodel.json
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
REPO = os.path.join(PROJECT_ROOT, "code", "repo")
sys.path.insert(0, REPO)

from jaume.device import resolve_device, seed_all, describe


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def macro_f1(y_true, y_pred, n_cls=3):
    f1s = []
    for c in range(n_cls):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s)


@torch.no_grad()
def encode(model, tokenizer, texts, device, max_length, bs=32):
    """Mean-pool the frozen backbone's last hidden states."""
    out = []
    for i in range(0, len(texts), bs):
        chunk = texts[i:i + bs]
        enc = tokenizer(chunk, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt").to(device)
        hidden, _ = model.backbone(enc["input_ids"], enc["attention_mask"])
        mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        out.append(pooled.float().cpu())
    return torch.cat(out, 0)


def text_of(r):
    return (r["stage1"] + " " + r["stage2"] + " " + r["claim"]).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="checkpoint dir (config.json + pytorch_model.bin)")
    ap.add_argument("--data", default=os.path.join(PROJECT_ROOT, "data", "scifact_blocks"))
    ap.add_argument("--seed", type=int, nargs="+", default=[0])
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)
    print(f"[t0] device={device} ({describe(device)})", flush=True)
    print(f"[t0] model={args.model}", flush=True)

    tr = load_jsonl(os.path.join(args.data, "train.jsonl"))
    va = load_jsonl(os.path.join(args.data, "val.jsonl"))
    te = load_jsonl(os.path.join(args.data, "test.jsonl"))
    print(f"[t0] train={len(tr)} val={len(va)} test={len(te)}", flush=True)

    from jaume.modeling.config import JaumeConfig
    from jaume.modeling.embedding import JaumeEmbeddingModel
    from jaume.data.dataset import load_tokenizer

    tok_path = os.path.join(args.model, "tokenizer")
    if not os.path.isdir(tok_path):
        tok_path = os.path.join(REPO, "jaume_tokenizer")
    # load_tokenizer repairs checkpoints whose tokenizer_config.json stores
    # extra_special_tokens as a list, which crashes transformers >= 4.5x
    tokenizer = load_tokenizer(tok_path)

    config = JaumeConfig.load(os.path.join(args.model, "config.json"))
    config.vocab_size = len(tokenizer)
    model = JaumeEmbeddingModel(config).to(device)
    # weights_only=False: the checkpoints this project writes go through
    # save_pretrained() and, under torch >= 2.6, fail the stricter
    # weights_only unpickler with "Unsupported global: GLOBAL getattr".
    # These are our own training outputs, not third-party downloads, so the
    # full unpickler is the right tool here.
    state = torch.load(os.path.join(args.model, "pytorch_model.bin"),
                       map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    print(f"[t0] backbone frozen; causal={getattr(config,'causal',False)} "
          f"model_type={config.model_type}", flush=True)

    def featurize(rows, tag):
        t0 = time.time()
        x = encode(model, tokenizer, [text_of(r) for r in rows], device, args.max_length)
        y = torch.tensor([r["label"] for r in rows])
        print(f"[t0] {tag}: {tuple(x.shape)} in {time.time()-t0:.0f}s", flush=True)
        return x, y

    Xtr, ytr = featurize(tr, "train")
    Xva, yva = featurize(va, "val")
    Xte, yte = featurize(te, "test")

    results = []
    for seed in args.seed:
        seed_all(device, seed)
        torch.manual_seed(seed)
        head = nn.Linear(Xtr.shape[1], 3).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
        Xtr_d, ytr_d = Xtr.to(device), ytr.to(device)
        Xva_d, yva_d = Xva.to(device), yva.to(device)
        Xte_d, yte_d = Xte.to(device), yte.to(device)

        best_val, best_state, best_ep = -1, None, -1
        for ep in range(args.epochs):
            head.train()
            perm = torch.randperm(len(Xtr_d), device=device)
            for i in range(0, len(perm), args.batch_size):
                idx = perm[i:i + args.batch_size]
                loss = nn.functional.cross_entropy(head(Xtr_d[idx]), ytr_d[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            head.eval()
            with torch.no_grad():
                pv = head(Xva_d).argmax(-1)
            vf1 = macro_f1(yva.tolist(), pv.cpu().tolist())
            if vf1 > best_val:
                best_val, best_ep = vf1, ep
                best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}

        head.load_state_dict(best_state)
        head.eval()
        with torch.no_grad():
            pt = head(Xte_d).argmax(-1)
        tf1 = macro_f1(yte.tolist(), pt.cpu().tolist())
        # constant baseline: always predict the train majority class
        maj = max(range(3), key=lambda c: (ytr == c).sum().item())
        cf1 = macro_f1(yte.tolist(), [maj] * len(yte))
        results.append({"seed": seed, "val_f1": best_val, "test_f1": tf1,
                        "best_epoch": best_ep, "const_f1": cf1})
        print(f"[t0] seed {seed}: best_val={best_val:.4f} (ep {best_ep})  "
              f"test_macroF1={tf1:.4f}  const={cf1:.4f}", flush=True)

    f1s = [r["test_f1"] for r in results]
    mean = sum(f1s) / len(f1s)
    lo, hi = min(f1s), max(f1s)
    const = results[0]["const_f1"]
    verdict = "PASS" if lo > const else ("WEAK" if mean > const else "FAIL")
    print("", flush=True)
    print(f"[t0] test macro-F1  mean={mean:.4f}  range=[{lo:.4f}, {hi:.4f}]  "
          f"constant_floor={const:.4f}", flush=True)
    print(f"[t0] VERDICT: {verdict}  "
          f"({'beats the floor' if verdict=='PASS' else 'does NOT reliably beat the floor'})",
          flush=True)

    payload = {"model": args.model, "device": device, "max_length": args.max_length,
               "epochs": args.epochs, "lr": args.lr, "results": results,
               "mean_test_f1": mean, "min_test_f1": lo, "max_test_f1": hi,
               "constant_floor": const, "verdict": verdict}
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[t0] wrote {args.out}", flush=True)
    return 0 if verdict != "FAIL" else 1


if __name__ == "__main__":
    sys.exit(main())
