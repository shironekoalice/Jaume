"""Sync trained checkpoints to a (private) ModelScope model repo.

Token sources (first match wins):
  1. --token argument
  2. $MODELSCOPE_TOKEN env var
  3. ~/.modelscope_token file (plain text)

Examples:
  # one checkpoint -> one repo
  python scripts/sync_modelscope.py \
      --model output/pipeline/350m-stage3/final --repo <username>/jaume-embedding-350m

  # several checkpoints, each into its own repo
  python scripts/sync_modelscope.py \
      --model output/pipeline/350m-stage3/final --repo <u>/jaume-embedding-350m \
      --model output/350m-decision --repo <u>/jaume-decision-350m

The repo is created as PRIVATE if it does not exist yet; existing repos are
reused and updated in place.
"""
import argparse
import os
import sys


def get_token(args):
    token = args.token or os.environ.get("MODELSCOPE_TOKEN")
    if not token:
        token_path = os.path.expanduser("~/.modelscope_token")
        if os.path.exists(token_path):
            token = open(token_path).read().strip()
    if not token:
        sys.exit("No token: pass --token, set $MODELSCOPE_TOKEN, "
                 "or write it to ~/.modelscope_token\n"
                 "Create one at https://modelscope.cn -> 账号设置 -> 访问令牌")
    return token


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", required=True,
                   help="checkpoint dir (repeatable)")
    p.add_argument("--repo", action="append", required=True,
                   help="target repo id <username>/<name> (repeatable, "
                        "aligned with --model order)")
    p.add_argument("--token", default=None)
    p.add_argument("--license", default="Apache-2.0")
    args = p.parse_args()
    if len(args.model) != len(args.repo):
        sys.exit("--model and --repo counts differ")

    from modelscope.hub.api import HubApi

    token = get_token(args)
    api = HubApi()
    api.login(token)
    print("[sync] logged in")

    for model_dir, repo_id in zip(args.model, args.repo):
        if not os.path.isdir(model_dir):
            sys.exit(f"missing checkpoint dir: {model_dir}")
        try:
            api.create_repo(repo_id, visibility=1, exist_ok=True,
                            license=args.license)
            print(f"[sync] repo ready (private): {repo_id}")
        except Exception as e:
            print(f"[sync] create_repo note: {e}")
        size = sum(os.path.getsize(os.path.join(r, f))
                   for r, _, fs in os.walk(model_dir) for f in fs)
        print(f"[sync] uploading {model_dir} ({size/1e9:.2f} GB) -> {repo_id}")
        api.push_model(model_id=repo_id, model_dir=model_dir)
        print(f"[sync] done: https://modelscope.cn/models/{repo_id}")


if __name__ == "__main__":
    main()
