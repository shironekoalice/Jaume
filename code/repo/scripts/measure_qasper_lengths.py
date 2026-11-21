"""Measure Qasper's real text lengths to size the concat-truncation problem.

Success criterion S3 needs `concat` to see the SAME content as `carried`.
Qasper full papers are far longer than the encoder's window (512 tokens), so
concat will truncate. This script quantifies that BEFORE any modelling decision,
because a baseline that sees less text is not a fair reference (review rule 4).

Outputs: token-length distribution of full texts, and the fraction of papers
that would be truncated at several candidate windows.

Run with the bundled interpreter (needs the tokenizer).
"""
import os as _os
PROJECT_ROOT = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import json
import os
import sys

REPO = fr"{PROJECT_ROOT}\code\repo"
sys.path.insert(0, REPO)

from transformers import AutoTokenizer  # noqa: E402

TOK = fr"{PROJECT_ROOT}\output\mem\SF-s0"


def paper_text(p):
    parts = [p.get("title", ""), p.get("abstract", "")]
    for sec in p.get("full_text", []):
        parts.append(sec.get("section_name", ""))
        parts.extend(sec.get("paragraphs", []))
    return "\n".join(x for x in parts if x)


def main():
    path = fr"{PROJECT_ROOT}\data\qasper_official\qasper-train-v0.3.json"
    d = json.load(open(path, encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(TOK)

    lens = []
    for pid in list(d.keys()):
        t = paper_text(d[pid])
        lens.append(len(tok(t, truncation=False)["input_ids"]))
    lens.sort()
    n = len(lens)

    def pct(q):
        return lens[min(n - 1, int(q * n))]

    print(f"papers={n}")
    print(f"token length  min={lens[0]}  p25={pct(.25)}  median={pct(.5)}  "
          f"p75={pct(.75)}  p90={pct(.90)}  max={lens[-1]}")
    print()
    print("truncation rate if concat uses a single window of size W:")
    for W in (256, 512, 768, 1024, 2048):
        over = sum(1 for L in lens if L > W)
        print(f"  W={W:5d}: {over:4d}/{n} papers exceed it "
              f"({100*over/n:5.1f}%)")

    print()
    print("chunks needed at 256 tokens per block:")
    import collections
    c = collections.Counter(max(1, -(-L // 256)) for L in lens)
    med_chunks = sorted(c.elements())[n // 2]
    print(f"  median blocks per paper = {med_chunks}")
    print(f"  blocks distribution (top): {sorted(c.items())[:8]}")


if __name__ == "__main__":
    main()

