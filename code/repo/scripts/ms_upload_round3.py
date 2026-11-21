"""Upload the staged experiment record to ModelScope.

Uploads one top-level folder at a time so a transient failure does not lose the
whole round, and records per-folder outcomes so a partial upload is visible
rather than silently reported as success.
"""
import os as _os
PROJECT_ROOT = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import os
import sys
import time

from modelscope.hub.api import HubApi

REPO = "LightNeko/Jaume"
SRC = fr"{PROJECT_ROOT}\_upload3"

token = open(os.path.expanduser("~/.modelscope_token"), encoding="utf-8").read().strip()
api = HubApi()
api.login(token)


def upload_once(local, path_in_repo, ignore_patterns):
    api.upload_folder(
        repo_id=REPO,
        folder_path=local,
        path_in_repo=path_in_repo,
        ignore_patterns=ignore_patterns,
    )


def main():
    only = sys.argv[1:] or sorted(
        d for d in os.listdir(SRC) if os.path.isdir(os.path.join(SRC, d))
    )
    ignore = ["*.bin", "*.pt", "*.pth", "tokenizer/*", "__pycache__/*"]
    ok, bad = [], []
    for name in only:
        local = os.path.join(SRC, name)
        if not os.path.isdir(local):
            continue
        t0 = time.time()
        for attempt in range(1, 4):
            try:
                upload_once(local, name, ignore)
                dt = time.time() - t0
                n = sum(len(f) for _, _, f in os.walk(local))
                print(f"[up] OK   {name:22s} {n:4d} files  {dt:6.1f}s", flush=True)
                ok.append(name)
                break
            except Exception as exc:  # noqa: BLE001
                print(f"[up] retry {attempt}/3 {name}: "
                      f"{type(exc).__name__}: {str(exc)[:160]}", flush=True)
                time.sleep(5 * attempt)
        else:
            print(f"[up] FAIL {name}", flush=True)
            bad.append(name)
    print(f"\n[up] done: {len(ok)} ok, {len(bad)} failed")
    if bad:
        print(f"[up] failed: {bad}")


if __name__ == "__main__":
    main()
