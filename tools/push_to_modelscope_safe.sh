#!/usr/bin/env bash
# Publish to ModelScope WITHOUT deleting the weights.
#
# THE PROBLEM. The two remotes want opposite things:
#
#   GitHub      code only. Weights, datasets and logs must NOT be there, so
#              the local index deliberately excludes them.
#   ModelScope  the weights, the datasets and the training logs MUST be
#              there, because that repo IS the model page. It already holds
#              574 weight/data/log files.
#
# A plain `git push` sends the DIFFERENCE between the local index and the
# remote, so with the code-oriented .gitignore it would DELETE 600 files
# from ModelScope, weights included.
#
# THE METHOD. Build the commit with plumbing, without ever checking out the
# weights:
#
#   1. take origin/master's tree as the base
#   2. overlay, in a TEMPORARY INDEX, only the paths that are new/changed
#      locally and that belong in the model page
#   3. commit-tree that, with origin/master as the parent
#   4. push the resulting commit
#
# Nothing is checked out, so git-lfs never tries to smudge a 1.1 GB weight
# file (which fails here because the LFS endpoint is unreachable). The
# weights stay exactly as they are on the remote: same blobs, same paths.
#
# Usage:
#   export MODELSCOPE_API_TOKEN=ms-...
#   bash tools/push_to_modelscope_safe.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MS_REPO="LightNeko/Jaume"
BRANCH="master"

echo ">> fetching the current ModelScope head..."
git fetch origin "$BRANCH" 2>&1 | tail -1
BASE="$(git rev-parse origin/$BRANCH)"
echo ">> base commit: $BASE"

KEEP_PREFIXES="docs/ code/ prototype/ tools/ results/ configs/ scripts/ tests/ jaume/ jaume_tokenizer/"
KEEP_FILES="README.md PLAN.md SPEC.md UPLOAD_MANIFEST.json .gitattributes .gitignore omni_notes.txt docvqa_env.py"

want() {
  local p="$1"
  for k in $KEEP_FILES; do [ "$p" = "$k" ] && return 0; done
  for pref in $KEEP_PREFIXES; do
    case "$p" in "$pref"*) return 0 ;; esac
  done
  return 1
}

REMOTE_FILES="$(git ls-tree -r origin/$BRANCH --name-only | LC_ALL=C sort)"
LOCAL_FILES="$(git ls-files | LC_ALL=C sort)"

CHANGED="$(comm -13 <(printf '%s\n' "$REMOTE_FILES") <(printf '%s\n' "$LOCAL_FILES") || true)"
REMOVED="$(comm -23 <(printf '%s\n' "$REMOTE_FILES") <(printf '%s\n' "$LOCAL_FILES") || true)"

ADD=""
for p in $CHANGED; do want "$p" && ADD="$ADD $p"; done

# Files that exist on the remote AND locally but whose CONTENT differs must
# also be updated — `comm` only reports added/removed paths, so an edited
# README or doc looks unchanged and would be silently skipped. Compare the
# blob shas instead of the path lists for those.
OVERWRITE=""
while read -r p; do
  [ -z "$p" ] && continue
  want "$p" || continue
  [ -f "$p" ] || continue
  remote_sha="$(git ls-tree -r origin/$BRANCH -- "$p" | awk '{print $3}')"
  [ -z "$remote_sha" ] && continue
  local_sha="$(git hash-object -- "$p")"
  if [ "$remote_sha" != "$local_sha" ]; then
    OVERWRITE="$OVERWRITE $p"
  fi
done <<< "$REMOTE_FILES"
[ -n "$OVERWRITE" ] && echo ">> existing files to update: $(printf '%s\n' $OVERWRITE | wc -l)"
for p in $OVERWRITE; do echo "     $p"; done

echo ">> remote-only files that MUST be preserved: $(printf '%s\n' "$REMOVED" | wc -l)"
echo ">> local additions to publish:               $(printf '%s\n' "$ADD" | wc -l)"
echo ">> existing files to update:                 $(printf '%s\n' "$OVERWRITE" | wc -l)"

# ---- build the new tree in a temporary index, starting from the remote's
INDEX="$(mktemp)"
trap 'rm -f "$INDEX"' EXIT
export GIT_INDEX_FILE="$INDEX"
git read-tree "$BASE"

# overlay the new/changed files
printf '%s\n' $ADD $OVERWRITE | while read -r p; do
  [ -z "$p" ] && continue
  [ -f "$p" ] || continue
  blob="$(git hash-object -w -- "$p")"
  git update-index --add --cacheinfo "100644,$blob,$p"
done

NEW_TREE="$(git write-tree)"
unset GIT_INDEX_FILE

echo ">> new tree: $NEW_TREE"

MSG="docs(actor): add the Actor research record, the reproduction scripts and the Ascend 950 application guide

  * docs/jaume-actor-prototype.md            architecture prototype (perception + solver)
  * docs/jaume-actor-backbone-selection.md   backbone selection on real ScreenSpot
  * docs/jaume-actor-handoff.md               machine handoff guide
  * docs/ascend-950-application.md            the 950PR application write-up
  * prototype/actor/                          46 scripts; check_consistency.py re-checks
                                              all 52 reported numbers against their json
  * README.md                                 the Ascend section and the reproduce block

  The weights, datasets and logs already in this repo are untouched."

NEW_COMMIT="$(printf '%s\n' "$MSG" | git commit-tree "$NEW_TREE" -p "$BASE")"
echo ">> new commit: $NEW_COMMIT (parent $BASE)"

# ---- verify nothing was lost before pushing
echo ">> verifying the new tree keeps every weight path..."
for p in 350m-mlm/pytorch_model.bin 350m-causal-mlm/final/pytorch_model.bin 350m-stage3/pytorch_model.bin; do
  if git ls-tree -r "$NEW_COMMIT" --name-only | grep -qx "$p"; then
    echo "   OK  $p"
  else
    echo "   !! MISSING $p -- aborting before push"
    exit 1
  fi
done
BEFORE="$(git ls-tree -r origin/$BRANCH --name-only | wc -l)"
AFTER="$(git ls-tree -r "$NEW_COMMIT" --name-only | wc -l)"
echo ">> file count before $BEFORE, after $AFTER"
if [ "$AFTER" -lt "$BEFORE" ]; then
  echo "   !! the new tree has FEWER files -- aborting before push"
  exit 1
fi

echo ">> pushing to ModelScope ..."
python "$REPO_ROOT/code/repo/scripts/push_to_modelscope.py" --repo "$MS_REPO" --branch "$BRANCH" --commit "$NEW_COMMIT"
echo ">> done: https://modelscope.cn/models/$MS_REPO"
