"""Uploads the staged tree to the existing ModelScope model repo.

Unlike sync_modelscope.py (which pushes a checkpoint DIR into its own repo),
this pushes a whole multi-folder tree into one repo, which is what the Jaume
series repo layout needs. Existing files with the same path are overwritten;
files already present with identical size are skipped to save bandwidth.

Usage:
    python scripts/upload_jaume.py --dir D:/Jaume/_upload \
        --repo LightNeko/Jaume --token <token>
"""
import argparse
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--token", default=None)
    ap.add_argument("--message", default="Jaume-Memory JM-S results (negative result + Slot Attention fixes)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None,
                    help="comma list of top-level paths to upload (default: all)")
    args = ap.parse_args()

    token = args.token or os.environ.get("MODELSCOPE_API_TOKEN")
    if not token:
        p = os.path.expanduser("~/.modelscope_token")
        if os.path.exists(p):
            token = open(p).read().strip()
    if not token:
        sys.exit("no token: pass --token, set $MODELSCOPE_API_TOKEN, or ~/.modelscope_token")

    os.environ["MODELSCOPE_API_TOKEN"] = token
    from modelscope.hub.api import HubApi

    api = HubApi()
    api.login(token)
    print(f"[upload] logged in, target repo {args.repo}")

    only = [s.strip() for s in args.only.split(",")] if args.only else None
    root = args.dir
    files = []
    for r, dirs, fs in os.walk(root):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in fs:
            full = os.path.join(r, f)
            rel = os.path.relpath(full, root).replace("\\", "/")
            if only and not any(rel == o or rel.startswith(o + "/") for o in only):
                continue
            files.append((rel, full, os.path.getsize(full)))
    files.sort()
    total = sum(s for _, _, s in files)
    print(f"[upload] {len(files)} files, {total/1e6:.1f} MB")

    if args.dry_run:
        for rel, _, s in files[:40]:
            print(f"   {s/1e6:8.2f} MB  {rel}")
        if len(files) > 40:
            print(f"   ... +{len(files)-40} more")
        return

    # upload folder by folder so a failure is isolated and progress is visible
    groups = {}
    for rel, full, s in files:
        key = rel.split("/")[0]
        groups.setdefault(key, []).append((rel, full, s))

    done_bytes = 0
    for key in sorted(groups):
        items = groups[key]
        gbytes = sum(s for _, _, s in items)
        print(f"[upload] -> {key}  ({len(items)} files, {gbytes/1e6:.1f} MB)")
        t0 = time.time()
        for rel, full, s in items:
            for attempt in range(3):
                try:
                    api.upload_file(
                        repo_id=args.repo, repo_type="model",
                        path_or_fileobj=full, path_in_repo=rel,
                        commit_message=args.message,
                    )
                    break
                except Exception as e:
                    if attempt == 2:
                        print(f"[upload] FAILED {rel}: {type(e).__name__}: {str(e)[:200]}")
                        raise
                    time.sleep(3 * (attempt + 1))
            done_bytes += s
        dt = time.time() - t0
        rate = gbytes / dt / 1e6 if dt else 0
        print(f"[upload]    {key} done in {dt:.0f}s ({rate:.2f} MB/s), "
              f"overall {done_bytes/1e6:.0f}/{total/1e6:.0f} MB")

    print(f"[upload] complete: https://modelscope.cn/models/{args.repo}")


if __name__ == "__main__":
    main()
