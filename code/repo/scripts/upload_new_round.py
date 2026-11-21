"""Upload exactly the files that are new since the last round.

Skips per-checkpoint tokenizer copies (identical across runs, tens of MB each)
and raw corpora. Uploads only experiment records, code and docs.
"""
import os as _os
PROJECT_ROOT = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import os
import sys

from modelscope.hub.api import HubApi

REPO = "LightNeko/Jaume"
ROOT = fr"{PROJECT_ROOT}"

targets = [
    "docs/memory-breakthrough-diagnosis.md",
    "code/repo/scripts/diagnose_composition_learning.py",
    "code/repo/scripts/sweep_lr.py",
    "code/repo/scripts/run_multi.py",
    "code/repo/scripts/run_frozen_head.py",
    "code/repo/scripts/run_state_on_frozen.py",
    "output/comp_head/split0_seed0.json",
    "output/comp_multi/split0_seed0.json",
    "output/comp_state_frozen/split0_seed0.json",
    "output/comp_state_frozen/split0_seed1.json",
    "output/comp_state_frozen/split0_seed2.json",
    "output/comp_state_frozen/split1_seed0.json",
    "output/lr_sweep.json",
    # code added in this round that belongs with the earlier experiments
    "code/repo/scripts/diagnose_gate.py",
    "code/repo/scripts/test_gate_value.py",
    "code/repo/scripts/audit_task_necessity.py",
    "code/repo/scripts/check_evidence_lexical.py",
    "code/repo/scripts/audit_qasper_multi_evidence.py",
    "code/repo/scripts/make_qasper_composition.py",
    "code/repo/scripts/check_composition_baselines.py",
    "code/repo/scripts/check_a_leakage.py",
    "code/repo/scripts/check_b_fairness.py",
    "code/repo/scripts/make_comp_clean_splits.py",
    "code/repo/scripts/run_nine.py",
    "code/repo/scripts/aggregate_nine.py",
]

token = open(os.path.expanduser("~/.modelscope_token"), encoding="utf-8").read().strip()
api = HubApi()
api.login(token)

ok, fail = [], []
for rel in targets:
    full = os.path.join(ROOT, rel.replace("/", os.sep))
    if not os.path.exists(full):
        print(f"[skip] missing locally: {rel}")
        continue
    try:
        api.upload_file(
            path_or_fileobj=full,
            path_in_repo=rel,
            repo_id=REPO,
        )
        ok.append(rel)
        print(f"[ok] {rel}")
    except Exception as exc:  # noqa: BLE001
        fail.append(rel)
        print(f"[FAIL] {rel}: {type(exc).__name__}: {str(exc)[:120]}")

print(f"\nuploaded {len(ok)}, failed {len(fail)}")
if fail:
    print("failed:", fail)
