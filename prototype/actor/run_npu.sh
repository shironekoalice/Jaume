#!/usr/bin/env bash
# Launch the Jaume-Actor NPU training run.
#
# Usage (from the repo root):
#   bash prototype/actor/run_npu.sh <mode>
#
# <mode>:
#   check     environment pre-flight only (torch_npu, device, SigLIP2, data)
#   ui       train the GATED SCORER on seeclick, then evaluate on real DOM
#   probe    the cheap backbone-selection probe (linear probe on frozen feats)
#   maze     maze control training with the generic trunk (no NPU needed)
#
# WHY THE GATED SCORER AND NOT THE JAUME-350M BACKBONE
# docs/jaume-actor-backbone-selection.md measured the alternatives on real
# ScreenSpot with a held-out linear probe:
#
#   SigLIP2-base (93M)   top-1 0.8500   44 ms/crop
#   Qwen3-VL-2B           top-1 0.8056  481 ms/crop
#   Qwen2-VL-2B           top-1 0.6667  580 ms/crop
#
# The 93M tower wins on accuracy AND on latency (11x faster per crop), so
# the actor is SigLIP2 (frozen) + a small instruction-conditioned head. The
# Jaume-350M dual-tower path (model_jaume.py) is kept as a comparison, not
# as the default.
#
# The remote layout follows the platform's conventions: the code tree under a
# working root, outputs and logs under it too. Both are configurable so the
# script is not tied to one account's home directory.
set -euo pipefail

MODE="${1:-check}"
ROOT="${JAUME_REMOTE_ROOT:-/mnt/workspace}"
SRC_ROOT="$ROOT/Jaume/prototype/actor"
OUT_ROOT="$ROOT/output/jaume-actor"
LOG_ROOT="$ROOT/log/jaume-actor"
REPO_ROOT="$ROOT/Jaume"

# CANN lives under the user's home on the platform images. Locate it rather
# than hardcoding one user's path.
CANN_HOME="${JAUME_CANN_HOME:-$(ls -d /opt/home/*/Ascend 2>/dev/null | head -1 || true)}"
if [ -n "$CANN_HOME" ]; then
  source "$CANN_HOME/ascend-toolkit/set_env.sh" >/dev/null 2>&1 || true
  source "$CANN_HOME/nnal/atb/set_env.sh" >/dev/null 2>&1 || true
fi

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

case "$MODE" in
  check)
    python - <<'PY'
import os, torch, torch_npu
print("torch", torch.__version__)
print("npu_available", torch.npu.is_available())
if torch.npu.is_available():
    print("device", torch.npu.get_device_name(0))
    free, total = torch.npu.mem_get_info()
    print(f"HBM free {free/2**30:.1f} GiB / total {total/2**30:.1f} GiB")
# the jaume package lives in code/repo (see docs/jaume-actor-handoff.md)
sys_path = os.environ.get("JAUME_REPO_PKG", "%s/code/repo" % os.environ.get("JAUME_REMOTE_ROOT", "/mnt/workspace") + "/Jaume")
import sys
sys.path.insert(0, sys_path)
try:
    from jaume.modeling.config import JaumeConfig
    from jaume.modeling.backbone_causal import CausalBackbone
    print("jaume package OK at", sys_path)
except Exception as e:
    print("jaume package FAIL:", type(e).__name__, e)
root = os.environ.get("JAUME_REMOTE_ROOT", "/mnt/workspace") + "/Jaume"
for p in [root + "/prototype/actor/models",
          root + "/prototype/actor/data/seeclick",
          root + "/prototype/actor/data/screenspot"]:
    print(p, "EXISTS" if os.path.exists(p) else "MISSING")
PY
    ;;

  probe)
    cd "$SRC_ROOT"
    python -u backbone_probe.py --rows 300 --k 6 --epochs 8 \
      --out "$OUT_ROOT/probe.json" 2>&1 | tee "$LOG_ROOT/probe.log"
    ;;

  ui)
    cd "$SRC_ROOT"
    # train the gated head on seeclick (real UI elements as distractors)
    python -u train_gated_head.py \
      --rows "${ROWS:-800}" --k "${K:-6}" --epochs "${EPOCHS:-20}" \
      --out "$OUT_ROOT/gated_head.json" --ckpt "$OUT_ROOT/gated_head.pt" \
      2>&1 | tee "$LOG_ROOT/gated_head.log"
    # then evaluate it on REAL DOM candidates in a real browser
    python -u eval_browser_gated.py \
      --ckpt "$OUT_ROOT/gated_head.pt" \
      --out "$OUT_ROOT/browser_gated.json" \
      2>&1 | tee "$LOG_ROOT/browser_gated.log"
    ;;

  maze)
    cd "$SRC_ROOT"
    python -u train_maze_ce.py \
      --device npu --epochs "${EPOCHS:-30}" \
      --mazes "${MAZES:-3000}" --frames "${FRAMES:-60}" \
      --width "${WIDTH:-48}" --depth "${DEPTH:-4}" \
      --out "$OUT_ROOT/maze.json" 2>&1 | tee "$LOG_ROOT/maze.log"
    ;;

  *)
    echo "unknown mode: $MODE" >&2; exit 2;;
esac
