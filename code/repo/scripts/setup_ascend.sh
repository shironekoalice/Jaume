#!/usr/bin/env bash
# One-shot setup for the Jaume-Causal twin inside an Ascend CANN environment.
#
#   bash scripts/setup_ascend.sh [workdir] [--train] [--batch N] [--auto-batch]
#
# default workdir: /mnt/workspace/jaume   (persistent across reboots)
#
# Steps: fetch code+data from ModelScope -> verify assets -> run the NPU
# pre-flight check -> (optionally) start training.
#
# NOTE ON CARD-HOURS: an Ascend instance consumes card-hours only while it is
# RUNNING. Everything below is designed to reach a training step in a few
# minutes so an idle instance does not burn the quota.

set -euo pipefail

WORKDIR="${WORKDIR:-/mnt/workspace/jaume}"
REPO_ID="${MODELSCOPE_REPO:-LightNeko/Jaume}"
BRANCH="${MODELSCOPE_BRANCH:-master}"
BATCH="${BATCH:-32}"
DO_TRAIN=0
AUTO_BATCH=0

while [ $# -gt 0 ]; do
  case "$1" in
    --train)       DO_TRAIN=1 ;;
    --batch)       BATCH="$2"; shift ;;
    --auto-batch)  AUTO_BATCH=1 ;;
    --workdir)     WORKDIR="$2"; shift ;;
    *)             WORKDIR="$1" ;;
  esac
  shift || true
done

echo "=============================================================="
echo " Jaume-Causal setup on Ascend"
echo "   workdir : $WORKDIR"
echo "   repo    : $REPO_ID @ $BRANCH"
if [ "$AUTO_BATCH" -eq 1 ]; then BATCH_LABEL="$BATCH (auto)"; else BATCH_LABEL="$BATCH"; fi
echo "   batch   : $BATCH_LABEL"
echo "=============================================================="

# ---------------------------------------------------------------- 1. device
echo ""
echo "[1/5] checking torch_npu"
python - <<'PY'
import sys
try:
    import torch_npu, torch
    print("  torch      :", torch.__version__)
    print("  torch_npu  :", getattr(torch_npu, "__version__", "unknown"))
    print("  npu avail  :", torch.npu.is_available())
    print("  npu count  :", torch.npu.device_count())
    if torch.npu.is_available():
        p = torch.npu.get_device_properties(0)
        print("  device 0   : %s  %.1f GiB" % (p.name, p.total_memory / 1024 ** 3))
except Exception as e:
    print("  torch_npu not usable:", repr(e)[:200])
    sys.exit(1)
PY

# ---------------------------------------------------------------- 2. deps
echo ""
echo "[2/5] installing python deps"
python -m pip install -q --disable-pip-version-check modelscope 2>&1 | tail -2 || true
python -m pip install -q --disable-pip-version-check "transformers>=4.30" tensorboard 2>&1 | tail -2 || true

# ---------------------------------------------------------------- 3. fetch
echo ""
echo "[3/5] fetching code + data from ModelScope"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

if [ ! -d "$WORKDIR/Jaume/code/repo/scripts/train_mlm_causal.py" ]; then
  echo "  downloading snapshot (pulls the LFS corpus, ~470 MB)..."
  # allow_patterns keeps the released checkpoints (8.4 GB of pytorch_model.bin)
  # out of the download: this run trains a NEW causal twin from scratch, it
  # never loads those weights. Pulling them would cost ~8 GB of transfer and
  # ~8 GB of disk for nothing.
  python - <<PY
from modelscope.hub.snapshot_download import snapshot_download
p = snapshot_download(
    '$REPO_ID',
    revision='$BRANCH',
    allow_patterns=[
        'code/**',
        'data/mlm_corpus*.jsonl',
        'data/all_nli.jsonl',
        'data/scifact_blocks/**',
        'data/sickr-test.jsonl',
        'data/sts22-crosslingual-sts.jsonl',
        'docs/**',
        'README.md',
        'SPEC.md',
        'PLAN.md',
        'configs/**',
    ],
    ignore_patterns=[
        '**/pytorch_model.bin',
        '**/*.safetensors',
        '**/optimizer*',
        '**/rng_state*',
        '**/training_args.bin',
        'output/**',
        'logs/tb/**',
    ],
)
print("  snapshot:", p)
open("$WORKDIR/_snapshot_path.txt", "w").write(p)
PY
  SRC="$(cat "$WORKDIR/_snapshot_path.txt")"
  mkdir -p "$WORKDIR/Jaume"
  cp -r "$SRC"/. "$WORKDIR"/Jaume/
else
  echo "  already present, skipping download"
fi

REPO="$WORKDIR/Jaume"
echo "  repo dir: $REPO"

# ---------------------------------------------------------------- 4. assets
echo ""
echo "[4/5] verifying required assets"
FAIL=0
for f in \
    "$REPO/code/repo/scripts/train_mlm_causal.py" \
    "$REPO/code/repo/scripts/check_npu_env.py" \
    "$REPO/code/repo/configs/jaume-350m-causal.json" \
    "$REPO/code/repo/jaume/modeling/attention_causal.py" \
    "$REPO/code/repo/jaume/modeling/backbone_causal.py" \
    "$REPO/code/repo/jaume/modeling/nn_compat.py" \
    "$REPO/code/repo/jaume/device.py" \
    "$REPO/code/repo/jaume_tokenizer/tokenizer.json" \
    "$REPO/data/mlm_corpus.jsonl" \
    "$REPO/data/scifact_blocks/train.jsonl" \
    "$REPO/data/scifact_blocks/val.jsonl" \
    "$REPO/data/scifact_blocks/test.jsonl" ; do
  if [ -s "$f" ]; then
    printf "  ok      %s\n" "${f#$REPO/}"
  else
    printf "  MISSING %s\n" "${f#$REPO/}"
    FAIL=1
  fi
done
if [ "$FAIL" -ne 0 ]; then
  echo ""
  echo "  Some assets are missing or came back as LFS pointers."
  echo "  Fix inside the repo dir:  git lfs pull"
  exit 1
fi

# ---------------------------------------------------------------- 5. preflight
echo ""
echo "[5/5] NPU pre-flight check"
cd "$REPO/code/repo"
if [ "$AUTO_BATCH" -eq 1 ]; then
  # Largest batch that actually fits. The exit code of the pre-flight script
  # is the signal (it returns non-zero when the forward/backward OOMs), so it
  # must NOT be piped -- a pipe would report tail's status, not python's.
  FOUND=0
  for b in 128 96 64 48 32 16; do
    echo ""
    echo "  --- trying batch-size $b ---"
    LOG="/tmp/npu_check_$b.log"
    if python scripts/check_npu_env.py --batch-size "$b" --max-length 128 >"$LOG" 2>&1; then
      FOUND=1
      grep -E 'tok/s|ETA|VERDICT|PASS|FAIL' "$LOG" | tail -20
      BATCH="$b"
      echo "  -> using batch-size $BATCH"
      break
    else
      grep -E 'FAIL|OutOfMemory|VERDICT' "$LOG" | tail -6
      echo "  -> batch-size $b does not fit, stepping down"
    fi
  done
  echo ""
  if [ "$FOUND" -eq 1 ]; then
    echo "  selected batch-size: $BATCH"
    RC=0
  else
    echo "  no batch size in {128,96,64,48,32,16} fit on this device."
    echo "  Free device memory and lower --max-length, or train on a card with more VRAM."
    RC=1
  fi
else
  python scripts/check_npu_env.py --batch-size "$BATCH" --max-length 128
  RC=$?
fi

echo ""
echo "=============================================================="
if [ "$RC" -ne 0 ]; then
  echo " NOT READY -- fix the failing checks above, then re-run."
  echo "=============================================================="
  exit $RC
fi

if [ "$DO_TRAIN" -eq 1 ]; then
  echo ""
  echo " starting training (batch=$BATCH)"
  echo "=============================================================="
  cd "$REPO/code/repo"
  exec python scripts/train_mlm_causal.py \
      --config configs/jaume-350m-causal.json \
      --data "$REPO/data/mlm_corpus.jsonl" \
      --output "$WORKDIR/output/350m-causal-mlm" \
      --max-samples 300000 \
      --batch-size "$BATCH" \
      --max-length 128 \
      --num-workers 8 \
      --save-every 500 \
      --log-every 20
else
  echo " Ready. Start training with:"
  echo ""
  echo "   cd $REPO/code/repo"
  echo "   python scripts/train_mlm_causal.py \\"
  echo "       --config configs/jaume-350m-causal.json \\"
  echo "       --data $REPO/data/mlm_corpus.jsonl \\"
  echo "       --output $WORKDIR/output/350m-causal-mlm \\"
  echo "       --max-samples 300000 --batch-size $BATCH --num-workers 8 \\"
  echo "       --save-every 500 --log-every 20"
  echo ""
  echo "   (or re-run this script with --train to do it in one step)"
fi
echo "=============================================================="
exit 0
