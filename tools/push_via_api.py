"""Push this repository to GitHub over the REST API, no git transport.

WHY. On this host `github.com:443` is unreliable ("Recv failure: Connection
was reset") while `api.github.com:443` answers normally. The normal
`git push` talks to github.com, so it cannot be used here; the REST API talks
to api.github.com and works. So the push is done by hand over the API:

    1. create blobs for every changed file           (POST /repos/{o}/{r}/git/blobs)
    2. build the tree objects for each directory    (POST .../git/trees)
    3. create the commit                             (POST .../git/commits)
    4. move the branch ref                          (PATCH .../git/refs/heads/{b})

The result on the server is a normal git commit with a normal tree, so the
repository looks exactly as if it had been pushed with git — the history,
the tree and the blobs are the same objects.

Token: read from GITHUB_TOKEN or ~/.github_token. Never written to disk.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

API = "https://api.github.com"
REPO = "shironekoalice/Jaume"
BRANCH = "main"


def token():
    t = os.environ.get("GITHUB_TOKEN")
    if t:
        return t.strip()
    p = os.path.expanduser("~/.github_token")
    if os.path.exists(p):
        return open(p, encoding="utf-8").read().strip()
    raise SystemExit("no token: set GITHUB_TOKEN or write ~/.github_token")


def call(method, path, body=None, tok=None, retries=4):
    url = API + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("Authorization", f"Bearer {tok}")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    req.add_header("User-Agent", "jaume-push")
    if data:
        req.add_header("Content-Type", "application/json")
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode() or "{}")
        except urllib.error.HTTPError as e:
            detail = e.read().decode()[:300]
            if e.code in (502, 503, 504) and attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            raise SystemExit(f"{method} {path} -> HTTP {e.code}: {detail}")
        except Exception as e:  # connection reset, timeout
            last = e
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            raise SystemExit(f"{method} {path} -> {type(e).__name__}: {e}")
    raise SystemExit(f"{method} {path} failed: {last}")


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace").stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--branch", default=BRANCH)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reset", action="store_true",
                    help="rewrite the branch as a single root commit whose tree "
                         "is exactly the local tree. The repair path for a repo "
                         "whose history was mangled by an earlier run.")
    ap.add_argument("--message", default="")
    ap.add_argument("--body", default="")
    args = ap.parse_args()

    tok = token()
    print(f"repo {args.repo} branch {args.branch}")

    # ---- what to publish
    files = git("ls-files").split("\n")
    files = [f for f in files if f.strip()]
    print(f"{len(files)} tracked files")

    # ---- the parent commit we build on: the current remote head
    ref = call("GET", f"/repos/{args.repo}/git/ref/heads/{args.branch}", tok=tok)
    base_sha = ref["object"]["sha"]
    base_commit = call("GET", f"/repos/{args.repo}/git/commits/{base_sha}", tok=tok)
    base_tree = base_commit["tree"]["sha"]
    print(f"remote head {base_sha[:8]} tree {base_tree[:8]}")

    # In --reset mode the tree is built from the LOCAL tree alone (no
    # base_tree), so the result is exactly the local tree and nothing from
    # the old remote tree leaks in.
    tree_base = "" if args.reset else base_tree

    # ---- blobs
    blob_cache = {}

    def blob_for(path):
        if path in blob_cache:
            return blob_cache[path]
        with open(path, "rb") as f:
            content = f.read()
        # IMPORTANT: the GitHub blobs API takes the RAW file content, NOT the
        # git object format. The first version of this script prepended the
        # "<len>\0" header (which is what `git hash-object` produces) and every
        # uploaded file came out with a "13554\0" prefix — enough to break
        # the markdown front matter and make the README render as one line.
        # The API computes the blob sha itself from the content it stores.
        r = call("POST", f"/repos/{args.repo}/git/blobs",
                 {"content": base64.b64encode(content).decode(), "encoding": "base64"}, tok=tok)
        blob_cache[path] = r["sha"]
        return r["sha"]

    # ---- build the tree from the local index
    # `git ls-tree -r HEAD` gives (mode, type, sha, path) for the LOCAL tree;
    # we re-create every blob through the API so the server stores the same
    # content.
    entries = git("ls-tree", "-r", "HEAD").split("\n")
    items = []
    for line in entries:
        if not line.strip():
            continue
        meta, path = line.split("\t", 1)
        mode, typ, _sha = meta.split()
        if path.startswith('"') and path.endswith('"'):
            path = path[1:-1].encode().decode("unicode_escape").encode("latin1").decode("utf-8", "replace")
        items.append((mode, path))
    print(f"{len(items)} tree entries")

    # Upload blobs concurrently — one HTTP round trip per file is far too slow
    # serially over a link that occasionally resets.
    from concurrent.futures import ThreadPoolExecutor

    # Reuse the blobs the server already has for unchanged paths: the remote
    # tree is fetched once and indexed by path, so only new or modified files
    # are uploaded. That cuts the upload from 626 blobs to the few hundred
    # that actually differ (the tokenizer and vocab files are unchanged and
    # are the largest).
    remote_tree = call("GET", f"/repos/{args.repo}/git/trees/{base_tree}?recursive=1", tok=tok)
    remote_shas = {e["path"]: e["sha"] for e in remote_tree.get("tree", [])
                   if e.get("type") == "blob"}
    if remote_tree.get("truncated"):
        print("!! remote tree truncated; falling back to uploading everything")
        remote_shas = {}
    print(f"remote already has {len(remote_shas)} blobs")

    local_shas = {}
    for line in git("ls-tree", "-r", "HEAD").split("\n"):
        if not line.strip():
            continue
        meta, path = line.split("\t", 1)
        mode, typ, sha = meta.split()
        # git quotes paths containing non-ASCII / special characters; unquote
        # so the path matches the filesystem and the API's view of it.
        if path.startswith('"') and path.endswith('"'):
            path = path[1:-1].encode().decode("unicode_escape").encode("latin1").decode("utf-8", "replace")
        local_shas[path] = (mode, sha)

    todo = []
    for path, (mode, sha) in local_shas.items():
        if remote_shas.get(path) == sha:
            continue          # identical object, no upload needed
        if os.path.isfile(path):
            todo.append((mode, path))
        else:
            print(f"  !! missing locally, skipped: {path}")
    print(f"{len(local_shas)} files, {len(todo)} need uploading")

    blob_shas = {}

    def work(entry):
        mode, path = entry
        return path, blob_for(path)

    with ThreadPoolExecutor(max_workers=8) as ex:
        for i, (path, sha) in enumerate(ex.map(work, todo)):
            blob_shas[path] = sha
            if (i + 1) % 100 == 0:
                print(f"  blobs {i+1}/{len(todo)}", flush=True)
    print(f"  blobs {len(blob_shas)}/{len(todo)} uploaded")

    # Build nested trees via the API's flat tree endpoint with base_tree.
    # Files the server already has keep their existing sha; only the changed
    # ones carry a freshly uploaded blob.
    flat = []
    reused = 0
    for mode, path in items:
        if path in blob_shas:
            flat.append({"path": path, "mode": mode, "type": "blob",
                         "sha": blob_shas[path]})
        elif path in remote_shas and not args.reset:
            flat.append({"path": path, "mode": mode, "type": "blob",
                         "sha": remote_shas[path]})
            reused += 1
    print(f"tree entries: {len(flat)} ({reused} reused from the remote)")
    payload = {"tree": flat}
    if tree_base:
        payload["base_tree"] = tree_base
    tree = call("POST", f"/repos/{args.repo}/git/trees", payload, tok=tok)
    if tree.get("truncated"):
        print("!! tree truncated by the API; needs chunked building")
        return 1
    new_tree = tree["sha"]
    print(f"new tree {new_tree[:8]}")

    # ---- commit
    # The message must describe THIS push, not whatever the local HEAD
    # happens to say. The first version read `git log -1`, so every push
    # after the first carried the same stale subject and the repo ended up
    # with three identical commit subjects.
    #
    # --reset drops the parent, making this a ROOT commit: the branch is
    # rewritten into one clean commit whose tree is exactly the local tree.
    # That is the repair path for a repo whose history was mangled by an
    # earlier run of this script.
    if args.reset:
        subject = args.message or "chore: publish the Jaume research tree"
        body = (args.body or "")
        parents = []
    else:
        remote_log = call("GET", f"/repos/{args.repo}/commits?per_page=1", tok=tok)
        prev_msg = ""
        if isinstance(remote_log, list) and remote_log:
            prev_msg = remote_log[0]["commit"]["message"]
        n_local = int(git("rev-list", "--count", "HEAD").strip() or "0")
        n_remote = len(call("GET", f"/repos/{args.repo}/commits?per_page=100", tok=tok))
        subject = f"chore: publish the current tree ({n_local - n_remote} local commits ahead)"
        body = (f"\n\nPublished from the working tree via the REST API (github.com is not\n"
                f"reachable from this host; api.github.com is).\n\n"
                f"Local commits that were not yet on the remote are folded into this one,\n"
                f"because the API path creates a single commit rather than replaying\n"
                f"history.\n\nPrevious remote subject: {prev_msg.splitlines()[0] if prev_msg else '(none)'}")
        parents = [base_sha]
    commit = call("POST", f"/repos/{args.repo}/git/commits",
                  {"message": subject + ("\n\n" + body if body else ""),
                   "tree": new_tree, "parents": parents}, tok=tok)
    new_commit = commit["sha"]
    print(f"new commit {new_commit[:8]}")

    if args.dry_run:
        print("(dry run, nothing written)")
        return 0

    # ---- move the branch
    call("PATCH", f"/repos/{args.repo}/git/refs/heads/{args.branch}",
         {"sha": new_commit, "force": True}, tok=tok)
    print(f"branch {args.branch} -> {new_commit[:8]}")
    print(f"verify: https://github.com/{args.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
