"""Push to a ModelScope git repository, resolving the API-token -> git-token step.

Why this exists
---------------
ModelScope has TWO credentials and they are not interchangeable:

  * an **access token** (`ms-...`), created at
    https://modelscope.cn/my/myaccesstoken -- used by the SDK/HTTP API;
  * a **git token**, obtained by exchanging the access token for one via
    `HubApi().login()`, and used as the HTTP Basic password for git.

Using the access token directly as the git password fails with
"HTTP Basic: Access denied ... or your account has 2FA enabled".

Two more details that this script handles:

  * `HubApi.login()` returns `(git_token, cookies)` -- the reverse of what its
    own docstring says ("Returns: cookies ... git_token").
  * The system Git Credential Manager intercepts modelscope.cn and needs an
    OAuth provider that does not exist for it. It must be disabled for the
    push, which is done with `-c credential.helper=` (an empty value RESETS
    the helper list -- two empty values in a row leave no helper at all).

The token is read from the environment and never written to disk:

    MODELSCOPE_API_TOKEN   (or pass --token)
    ~/.modelscope_token   (fallback)

Usage:
    python scripts/push_to_modelscope.py --repo LightNeko/Jaume
    python scripts/push_to_modelscope.py --repo LightNeko/Jaume --branch master
"""

import argparse
import io
import os
import shutil
import subprocess
import sys
import tempfile


def read_token(explicit):
    if explicit:
        return explicit.strip()
    env = os.environ.get("MODELSCOPE_API_TOKEN")
    if env:
        return env.strip()
    path = os.path.expanduser("~/.modelscope_token")
    if os.path.exists(path):
        with io.open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    raise SystemExit(
        "no token: pass --token, set $MODELSCOPE_API_TOKEN, "
        "or write ~/.modelscope_token"
    )


def git_token_for(access_token):
    """Exchange the ModelScope access token for the git token."""
    from modelscope.hub.api import HubApi

    ret = HubApi().login(access_token=access_token)
    if isinstance(ret, (tuple, list)) and ret:
        first = ret[0]
        if isinstance(first, str):
            return first
    raise SystemExit("could not obtain a git token from the access token")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="e.g. LightNeko/Jaume")
    ap.add_argument("--branch", default="master")
    ap.add_argument("--token", default=None)
    ap.add_argument("--commit", default=None,
                    help="push this commit sha instead of the local branch. "
                         "Needed when the commit was built with plumbing "
                         "(commit-tree) on top of the remote head, so that the "
                         "weights already on the remote are preserved.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    token = read_token(args.token)
    gittok = git_token_for(token)
    print("git token acquired (length %d)" % len(gittok))

    url = "https://modelscope.cn/models/%s.git" % args.repo
    tmp = tempfile.mkdtemp(prefix="ms_git_")
    try:
        askpass = os.path.join(tmp, "askpass.bat")
        with io.open(askpass, "w", encoding="utf-8", newline="\r\n") as f:
            f.write("@echo off\r\n")
            f.write('echo %1 | findstr /i "username" >nul && (echo LightNeko && exit /b 0)\r\n')
            f.write("echo " + gittok + "\r\n")

        env = dict(os.environ)
        env["GIT_TERMINAL_PROMPT"] = "0"
        env["GIT_ASKPASS"] = askpass
        env["GIT_ASKPASS_REQUIRE"] = "force"
        for k in ("GIT_CONFIG_COUNT", "GIT_CONFIG_KEY_0", "GIT_CONFIG_VALUE_0"):
            env.pop(k, None)

        cmd = [
            "git",
            "-c", "credential.helper=",   # resets the helper list -> GCM off
            "-c", "core.askPass=" + askpass,
        ]
        if args.commit:
            cmd += ["push", url, f"{args.commit}:refs/heads/{args.branch}"]
        else:
            cmd += ["push", url, args.branch]
        if args.dry_run:
            print("dry-run:", " ".join(cmd[:1] + cmd[5:]))
            return 0

        r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        out = ((r.stdout or "") + (r.stderr or "")).replace(gittok, "***GIT_TOKEN***")
        print(out.strip())
        return r.returncode
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
