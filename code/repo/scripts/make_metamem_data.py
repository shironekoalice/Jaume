"""Build the meta-memory write-selection dataset from qasper_blocks.

Groups the per-(paper, question) records of data/qasper_blocks/{train,val,test}.jsonl
into one record per paper: all blocks plus the UNION of evidence indices across
that paper's questions (the preregistered proxy label for "worth remembering",
docs/memory-metamem-spec.md §1).

Usage:
    python scripts/make_metamem_data.py --out data/metamem/papers.jsonl
"""
import argparse
import json
import os
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.abspath(os.path.join(REPO, "..", ".."))
DATA = os.path.join(PROJECT_ROOT, "data", "qasper_blocks")


def evidence_list(rec):
    ev = rec["evidence_idx"]
    if isinstance(ev, int):
        ev = [ev]
    n = rec["n_blocks"]
    return sorted({i for i in ev if isinstance(i, int) and 0 <= i < n})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "data", "metamem", "papers.jsonl"))
    args = ap.parse_args()

    papers = {}
    stats = {"papers": 0, "blocks": 0, "questions": 0, "inconsistent_blocks": 0}
    for split in ("train", "val", "test"):
        path = os.path.join(DATA, f"{split}.jsonl")
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                pid = rec["paper_id"]
                p = papers.setdefault(
                    pid,
                    {"paper_id": pid, "split": split, "blocks": rec["blocks"],
                     "sections": rec["sections"], "questions": [], "evidence_union": set()},
                )
                if len(p["blocks"]) != rec["n_blocks"]:
                    stats["inconsistent_blocks"] += 1
                p["questions"].append(
                    {"question": rec["question"], "answer": rec["answer"],
                     "evidence": evidence_list(rec)}
                )
                p["evidence_union"].update(evidence_list(rec))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in papers.values():
            p["evidence_union"] = sorted(p["evidence_union"])
            stats["papers"] += 1
            stats["blocks"] += len(p["blocks"])
            stats["questions"] += len(p["questions"])
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    by_split = defaultdict(int)
    for p in papers.values():
        by_split[p["split"]] += 1
    print(f"[metamem] papers: {stats['papers']} ({dict(by_split)})  "
          f"blocks: {stats['blocks']}  questions: {stats['questions']}  "
          f"inconsistent-block records: {stats['inconsistent_blocks']}")

    # class balance sanity (spec §2 quotes these numbers)
    pos = sum(len(p["evidence_union"]) for p in papers.values())
    print(f"[metamem] positive blocks: {pos} "
          f"(ratio {pos / max(1, stats['blocks']):.3f})")


if __name__ == "__main__":
    main()
