"""Backfill/parse training logs into TensorBoard events, idempotently.

The running pipeline writes to output/pipeline.log (all stages); older runs
have their own logs. This script tracks each log's consumed byte offset in
output/.tb_sync_state.json, converts new [mlm]/[jaume] step lines into
SummaryWriter scalars under the run's tb/ dir. Run it on a loop (cron or
`while true`) next to tensorboard.
"""
import glob
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
STATE_PATH = os.path.join(OUT, ".tb_sync_state.json")

# patterns cover both [mlm] and [jaume] log lines
STEP_RE = re.compile(
    r"step (\d+) loss ([\d.]+) lr ([\d.eE+-]+)"
    r"(?: tokens/s (\d+)| pairs/s (\d+))"
)

# log file -> [(run_dir, source_pattern)] ; runs live under output/
SOURCES = {
    "pipeline.log": "pipeline",          # stage determined by section headers
    "stage3-train.log": "350m-stage3-hardneg",
    "gc256-train.log": "350m-stage3-gc256",
    "mlm-train.log": "350m-mlm",
}

SECTION_RE = re.compile(r"=== \[\d+/\d+\][^:]*: (.+?) ===")
# map pipeline section titles to run dirs (capture excludes the "Stage N:" prefix)
SECTION_RUN = {
    "MLM pretraining": "pipeline/350m-mlm",
    "contrastive from MLM init (all entailment pairs)": "pipeline/350m-stage2",
    "retrain with hard negatives": "pipeline/350m-stage3",
}


def load_state():
    try:
        with open(STATE_PATH) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(state):
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, STATE_PATH)


_WRITERS = {}


def get_writer(run_name):
    if run_name not in _WRITERS:
        from torch.utils.tensorboard import SummaryWriter

        os.makedirs(os.path.join(OUT, run_name), exist_ok=True)
        _WRITERS[run_name] = SummaryWriter(log_dir=os.path.join(OUT, run_name, "tb"))
    return _WRITERS[run_name]


def sync():
    state = load_state()
    changed = False
    for log_name, default_run in SOURCES.items():
        path = os.path.join(OUT, log_name)
        if not os.path.exists(path):
            continue
        offset = state.get(log_name, 0)
        size = os.path.getsize(path)
        if size < offset:
            offset = 0  # truncated/rotated log
        with open(path, "rb") as f:
            f.seek(offset)
            new_text = f.read().decode("utf-8", errors="replace")
        if not new_text:
            continue
        new_offset = offset + len(new_text.encode("utf-8"))

        run_name = state.get(log_name + ":run", default_run)
        n = 0
        try:
            for line in new_text.splitlines():
                m = SECTION_RE.search(line)
                if m and m[1] in SECTION_RUN:
                    run_name = SECTION_RUN[m[1]]
                    state[log_name + ":run"] = run_name  # persists across increments
                m = STEP_RE.search(line)
                if not m:
                    continue
                step, loss, lr = int(m[1]), float(m[2]), float(m[3])
                get_writer(run_name).add_scalar("train/loss", loss, step)
                get_writer(run_name).add_scalar("train/lr", lr, step)
                if m[4]:
                    get_writer(run_name).add_scalar("train/tokens_per_s", float(m[4]), step)
                elif m[5]:
                    get_writer(run_name).add_scalar("train/pairs_per_s", float(m[5]), step)
                n += 1
        finally:
            for w in _WRITERS.values():
                w.flush()
        state[log_name] = new_offset
        changed = True
        if n:
            print(f"[tb_sync] {log_name}: +{n} points -> {run_name}/tb")
    if changed:
        save_state(state)


if __name__ == "__main__":
    once = "--once" in sys.argv
    while True:
        try:
            sync()
        except Exception as e:
            print(f"[tb_sync] error: {e}", flush=True)
        if once:
            break
        time.sleep(60)
