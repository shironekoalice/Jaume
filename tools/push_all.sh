#!/usr/bin/env bash
# Publish this tree to the code host the 950 application points at.
#
# WHY THIS IS NEEDED. The HiDevLab 950 application form asks for a
# "开源项目链接" restricted to GitHub / GitCode. The reviewer opens it. The
# GitHub mirror was a code-only skeleton (code/, scripts/, tests/ — no docs/,
# no prototype/), so opening it showed something unrelated to what the
# application describes. ModelScope already carries the full research
# record but is a MODEL page, not a code link, so it does not satisfy the
# form.
#
# So: push the current tree (docs/ + prototype/ + code/) to GitHub.
#
# CREDENTIALS — a GitHub PAT with `repo` scope
# (<https://github.com/settings/tokens>). Supply it any of these ways:
#
#   export GITHUB_TOKEN=ghp_xxx
#   echo ghp_xxx > ~/.github_token
#   or paste it at the prompt (input is not echoed, not written to disk,
#   not left in shell history)
#
# Usage:
#   bash tools/push_all.sh              # GitHub only
#   bash tools/push_all.sh --modelscope # also ModelScope (weights + logs)
#
# ModelScope needs a DIFFERENT credential: an access token (ms-...) from
# <https://modelscope.cn/my/myaccesstoken>. It cannot be used as the git
# password directly; code/repo/scripts/push_to_modelscope.py exchanges it for
# a git token via HubApi().login().
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DO_MS=0
[ "${1:-}" = "--modelscope" ] && DO_MS=1

read_token() {
  if [ -n "${GITHUB_TOKEN:-}" ]; then echo "$GITHUB_TOKEN"; return; fi
  if [ -f ~/.github_token ]; then cat ~/.github_token; return; fi
  printf 'GitHub PAT (repo scope, https://github.com/settings/tokens): ' >&2
  stty -echo 2>/dev/null || true
  read -r t < /dev/tty
  stty echo 2>/dev/null || true
  echo >&2
  echo "$t"
}

TOKEN="$(read_token)"
[ -n "$TOKEN" ] || { echo "!! no token; aborting" >&2; exit 1; }

git remote add github https://github.com/shironekoalice/Jaume.git 2>/dev/null || \
  git remote set-url github https://github.com/shironekoalice/Jaume.git

echo ">> pushing master -> main on GitHub ..."
# Emptying the credential-helper list stops the system Git Credential Manager
# from intercepting the host and asking for an OAuth provider that does not
# exist for it. The PAT goes in as the password with the token as username,
# which is what GitHub accepts.
GIT_TERMINAL_PROMPT=0 git -c credential.helper= \
  push "https://x-access-token:${TOKEN}@github.com/shironekoalice/Jaume.git" \
  master:main --force-with-lease
unset TOKEN

if [ "$DO_MS" = "1" ]; then
  echo ">> pushing to ModelScope ..."
  python "$REPO_ROOT/code/repo/scripts/push_to_modelscope.py" \
    --repo LightNeko/Jaume --branch master
fi

echo
echo ">> verify: https://github.com/shironekoalice/Jaume"
echo ">> expect: recent commit history, and docs/ + prototype/ in the file list"
