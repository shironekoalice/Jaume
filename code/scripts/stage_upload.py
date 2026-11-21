"""Stages the ModelScope upload tree.

The full output/ tree is ~18.8GB, almost all of it duplicate weights
(17 runs x 1.1GB pytorch_model.bin + a 24MB tokenizer copy per run). The
released 350m checkpoints already live in the repo, so this stages the
EXPERIMENT RECORD instead: configs, seeds, throughput, TensorBoard events and
every evaluation result, with the per-run weight files and tokenizer copies
left out. A manifest records exactly what was skipped, so the upload is not
silently lossy.

Usage:
    python scripts/stage_upload.py --src D:/Jaume --dst D:/Jaume/_upload
"""
import argparse
import json
import os
import shutil

# never worth uploading: per-run weights and the per-run tokenizer copy
SKIP_NAMES = {"pytorch_model.bin", "optimizer.pt", "scheduler.pt",
              "training_args.bin", "rng_state.pth"}
SKIP_DIRS = {"tokenizer", "__pycache__", ".git"}


def stage(src, dst, dry=False):
    copied, skipped = [], []
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and d != "_upload"]
        for fn in files:
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, src)
            size = os.path.getsize(full)
            if fn in SKIP_NAMES or rel.startswith("data" + os.sep) and size > 200e6:
                skipped.append({"path": rel, "bytes": size, "reason": "weights/huge"})
                continue
            target = os.path.join(dst, rel)
            copied.append({"path": rel, "bytes": size})
            if not dry:
                os.makedirs(os.path.dirname(target), exist_ok=True)
                shutil.copy2(full, target)
    return copied, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="D:/Jaume")
    ap.add_argument("--dst", default="D:/Jaume/_upload")
    args = ap.parse_args()

    copied, skipped = stage(args.src, args.dst)
    total = sum(c["bytes"] for c in copied)
    skipped_bytes = sum(s["bytes"] for s in skipped)
    print(f"[stage] files staged : {len(copied):4d}  ({total/1e6:.1f} MB)")
    print(f"[stage] files skipped: {len(skipped):4d}  ({skipped_bytes/1e9:.2f} GB)")
    for s in skipped[:20]:
        print(f"          skip {s['path']} ({s['bytes']/1e6:.0f} MB)")

    manifest = {
        "staged_files": len(copied), "staged_mb": round(total / 1e6, 2),
        "skipped_files": len(skipped), "skipped_gb": round(skipped_bytes / 1e9, 3),
        "skipped": skipped,
    }
    os.makedirs(args.dst, exist_ok=True)
    with open(os.path.join(args.dst, "UPLOAD_MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[stage] wrote {os.path.join(args.dst, 'UPLOAD_MANIFEST.json')}")


if __name__ == "__main__":
    main()
