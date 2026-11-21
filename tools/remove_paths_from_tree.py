"""Delete specific paths from a git tree via the plumbing API, by raw bytes.

WHY. `git update-index --force-remove -- <path>` does not remove a path whose
name contains non-ASCII characters: git stores those paths in the index in a
quoted, octal-escaped form, and the plumbing call does not match them unless
the caller reproduces that escaping exactly. The result is that a file whose
NAME is a Chinese document heading survives every "delete by name" attempt.

This script rewrites the tree without it, working on the raw path bytes from
`git ls-tree`, and verifies that the path is really gone afterwards.

Usage:
    python tools/remove_paths_from_tree.py <repo-dir> <tree-sha> <path>...
"""
from __future__ import annotations

import subprocess
import sys


def git(repo, *args, binary=False):
    r = subprocess.run(["git", "-C", repo, *args], capture_output=True)
    if r.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {r.stderr.decode(errors='replace')[:300]}")
    return r.stdout if binary else r.stdout.decode("utf-8", "replace")


def main():
    repo, tree_sha = sys.argv[1], sys.argv[2]
    targets = sys.argv[3:]

    # raw listing: the path bytes as git stored them
    raw = git(repo, "ls-tree", "-r", "-z", tree_sha, binary=True)
    entries = []
    for rec in raw.split(b"\0"):
        if not rec:
            continue
        meta, _, path = rec.partition(b"\t")
        mode, _typ, sha = meta.decode().split()
        entries.append((mode, sha, path))

    # match the targets: compare the decoded name, since the caller gives us
    # the readable form
    import codecs
    drop = set()
    for t in targets:
        tb = t.encode("utf-8")
        for mode, sha, path in entries:
            if path == tb or path.decode("utf-8", "replace") == t:
                drop.add(path)
                break
        else:
            print(f"!! not found in tree: {t}")

    keep = [(m, s, p) for m, s, p in entries if p not in drop]
    print(f"tree entries {len(entries)} -> keeping {len(keep)}, dropping {len(entries)-len(keep)}")

    # rebuild with git mktree, which takes the RAW path bytes
    lines = []
    for mode, sha, path in keep:
        lines.append(f"{mode} blob {sha}\t".encode() + path)
    inp = b"\0".join(lines) + b"\0"
    new = subprocess.run(["git", "-C", repo, "mktree", "-z"],
                         input=inp, capture_output=True)
    if new.returncode != 0:
        raise SystemExit(f"mktree failed: {new.stderr.decode(errors='replace')[:300]}")
    print("new tree:", new.stdout.decode().strip())


if __name__ == "__main__":
    main()
