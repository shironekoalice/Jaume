"""Tests for --resume of the causal MLM training.

A 2.4-hour (or 27-hour) run must survive an interruption without silently
corrupting the training trajectory. These tests pin the properties that make
that safe:

  1. the checkpoint carries everything needed to continue;
  2. a resumed run reproduces the UNINTERRUPTED loss trajectory exactly;
  3. already-consumed corpus rows are not replayed;
  4. a checkpoint without trainer_state.pt is refused rather than resumed
     from a partially-known state.

They are slow (they actually train a tiny model), but they are the only
thing standing between a resumed run and a silently different one.
"""

import json
import os
import re
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "scripts", "train_mlm_causal.py")
CONFIG = os.path.join(REPO, "configs", "jaume-350m-causal.json")


def _corpus(path, n):
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"text": "row %d " % i * 5}) + "\n")


def _cmd(corpus, out, extra=()):
    return [
        sys.executable, SCRIPT,
        "--config", CONFIG,
        "--data", corpus,
        "--output", out,
        "--max-samples", "80",
        "--batch-size", "16",
        "--max-length", "32",
        "--save-every", "2",
        "--log-every", "1",
        "--num-workers", "0",
        *extra,
    ]


def _losses(text):
    out = {}
    for line in text.splitlines():
        m = re.search(r"step (\d+) loss ([0-9.]+)", line)
        if m:
            out[int(m.group(1))] = float(m.group(2))
    return out


def test_resume_reproduces_the_uninterrupted_trajectory():
    with tempfile.TemporaryDirectory() as d:
        corpus = os.path.join(d, "corpus.jsonl")
        _corpus(corpus, 80)
        out = os.path.join(d, "out-causal")   # 'causal' is required by the guard

        r1 = subprocess.run(_cmd(corpus, out), cwd=REPO, capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=1800)
        assert r1.returncode == 0, r1.stderr[-2000:]
        l1 = _losses(r1.stdout)
        assert len(l1) >= 5, l1   # 80 rows / batch 16 = 5 steps

        r2 = subprocess.run(_cmd(corpus, out, ["--resume", os.path.join(out, "step-2")]),
                            cwd=REPO, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=1800)
        assert r2.returncode == 0, r2.stderr[-2000:]
        l2 = _losses(r2.stdout)

        for step in range(3, 6):
            assert step in l1 and step in l2, (step, sorted(l1), sorted(l2))
            assert l1[step] == l2[step], (
                "step %d differs after resume: %r vs %r" % (step, l1[step], l2[step])
            )


def test_resume_does_not_replay_consumed_rows():
    with tempfile.TemporaryDirectory() as d:
        corpus = os.path.join(d, "corpus.jsonl")
        _corpus(corpus, 80)
        out = os.path.join(d, "out-causal")

        subprocess.run(_cmd(corpus, out), cwd=REPO, capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=1800)
        r = subprocess.run(_cmd(corpus, out, ["--resume", os.path.join(out, "step-2")]),
                           cwd=REPO, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=1800)
        assert r.returncode == 0, r.stderr[-2000:]
        m = re.search(r"resume: step (\d+), (\d+) samples already consumed.*?(\d+) samples left",
                      r.stdout)
        assert m, r.stdout[-2000:]
        step, consumed, left = int(m.group(1)), int(m.group(2)), int(m.group(3))
        assert consumed == step * 16
        assert consumed + left == 80


def test_resume_refuses_a_checkpoint_without_trainer_state():
    with tempfile.TemporaryDirectory() as d:
        corpus = os.path.join(d, "corpus.jsonl")
        _corpus(corpus, 80)
        out = os.path.join(d, "out-causal")
        bogus = os.path.join(d, "bogus")
        os.makedirs(bogus, exist_ok=True)
        r = subprocess.run(_cmd(corpus, out, ["--resume", bogus]), cwd=REPO,
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=600)
        assert r.returncode != 0
        assert "trainer_state.pt" in (r.stdout + r.stderr)


def test_output_dir_guard_requires_causal_in_the_name():
    with tempfile.TemporaryDirectory() as d:
        corpus = os.path.join(d, "corpus.jsonl")
        _corpus(corpus, 80)
        r = subprocess.run(_cmd(corpus, os.path.join(d, "out")), cwd=REPO,
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=600)
        assert r.returncode != 0
        assert "REFUSING TO RUN" in (r.stdout + r.stderr)
