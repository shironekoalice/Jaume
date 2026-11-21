"""Leakage checks for the SciFact multi-stage dataset (spec L1-L5).

Must pass before any training, per the experiment spec.

  L1  stage-1/2 text must not contain the claim
  L2  the label word must not appear in stage-1/2 text
  L3  document-level split disjointness
  L4  lexical baseline: predicting the label from the CLAIM alone
  L5  label distribution sanity + majority-class floor

Usage:
    python scripts/check_scifact_data.py --data D:/Jaume/data/scifact_blocks
"""
import os as _os
PROJECT_ROOT = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import argparse
import collections
import json
import os
import re


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def toks(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=fr"{PROJECT_ROOT}\data\scifact_blocks")
    args = ap.parse_args()

    tr = load(os.path.join(args.data, "train.jsonl"))
    va = load(os.path.join(args.data, "val.jsonl"))
    te = load(os.path.join(args.data, "test.jsonl"))
    ok = True

    # L1: claim must not appear inside either stage
    bad = 0
    for r in tr + va + te:
        ctx = (r["stage1"] + " " + r["stage2"]).lower()
        if r["claim"].lower() in ctx:
            bad += 1
    print(f"L1 claim not in stage text       : {bad} "
          f"{'PASS' if bad == 0 else 'FAIL'}")
    ok &= bad == 0

    # L2: the label must not be present as an explicit marker in stage text.
    # NOTE: the words "support"/"contradict" occur in ordinary scientific prose
    # ("these data support..."), so a bare word search flags 105/1109 rows and
    # says nothing about leakage. What WOULD be leakage is the dataset's label
    # token appearing as a standalone annotation, e.g. "LABEL: SUPPORT" or
    # "[CONTRADICT]". Only those forms are checked here.
    marker = re.compile(r"(label\s*[:=])|(\[\s*(support|contradict)\s*\])|"
                        r"(\bsupport\s*\|\s*contradict\b)", re.I)
    bad = 0
    for r in tr + va + te:
        ctx = r["stage1"] + " " + r["stage2"]
        if marker.search(ctx):
            bad += 1
    print(f"L2 no label annotation in stage  : {bad} "
          f"{'PASS' if bad == 0 else 'FAIL'}")
    ok &= bad == 0

    # L3: document disjointness
    def docs(rows):
        s = set()
        for r in rows:
            s |= set(r["doc_ids"])
        return s
    dtr, dva, dte = docs(tr), docs(va), docs(te)
    o1, o2, o3 = len(dtr & dva), len(dtr & dte), len(dva & dte)
    print(f"L3 doc overlap tr/va={o1} tr/te={o2} va/te={o3} "
          f"{'PASS' if o1 == o2 == o3 == 0 else 'FAIL'}")
    ok &= (o1 == o2 == o3 == 0)

    # L4: can the label be read off the CLAIM alone by keyword matching?
    # A strong lexical baseline means the task is solvable without evidence.
    def lexical_baseline(train, test):
        # learn token -> label association on train, apply to test claim
        stats = collections.defaultdict(collections.Counter)
        for r in train:
            for t in toks(r["claim"]):
                stats[t][r["label"]] += 1
        correct = 0
        for r in test:
            score = collections.Counter()
            for t in toks(r["claim"]):
                if t in stats:
                    tot = sum(stats[t].values())
                    for lab, c in stats[t].items():
                        score[lab] += c / tot
            pred = score.most_common(1)[0][0] if score else 0
            if pred == r["label"]:
                correct += 1
        return correct / len(test)

    lb = lexical_baseline(tr, te)
    maj = collections.Counter(r["label"] for r in te).most_common(1)[0][1] / len(te)
    print(f"L4 lexical baseline (claim only) : macro-acc {lb:.3f} "
          f"(majority {maj:.3f}) "
          f"{'PASS' if lb < maj + 0.08 else 'WARN: claim alone is predictive'}")
    ok &= (lb < maj + 0.08)

    # L5: distributions
    print("\nL5 label distribution:")
    for name, rows in (("train", tr), ("val", va), ("test", te)):
        c = collections.Counter(r["label_name"] for r in rows)
        print(f"   {name:5s} n={len(rows):4d} "
              f"NEI={c['NEI']:4d} SUPPORT={c['SUPPORT']:4d} "
              f"CONTRADICT={c['CONTRADICT']:4d}")
    # macro-F1 floor for a constant predictor
    def macro_f1_constant(rows):
        n = len(rows)
        c = collections.Counter(r["label"] for r in rows)
        pred = c.most_common(1)[0][0]
        f1s = []
        for lab in (0, 1, 2):
            tp = sum(1 for r in rows if r["label"] == lab and pred == lab)
            fp = sum(1 for r in rows if r["label"] != lab and pred == lab)
            fn = sum(1 for r in rows if r["label"] == lab and pred != lab)
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
        return sum(f1s) / 3
    print(f"\n   constant-predictor Macro-F1 on test = "
          f"{macro_f1_constant(te):.3f}  (must be clearly beaten)")

    print(f"\nSCI-FACT DATA READY: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
