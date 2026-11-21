#!/usr/bin/env bash
# JM-S fix experiments, grounded in Slot Attention (Locatello et al. 2020,
# arXiv:2006.15055). Root cause of the v1 failure: slots converged to identical
# reads (slot-state cosine 0.87, JS 0.04) because the softmax was over TOKENS
# computed independently per slot -> no competition between slots.
#
#   C1  slot competition (softmax over slots)          - the paper's core fix
#   C2  C1 + 3 iterative refinements + GRU update      - the full module
#   C3  C2 + diversity loss on slot states             - explicit anti-collapse
#   C4  C2 + 16 slots                                  - capacity, once slots specialise?
#   C5  C2 + slot-pool=attention                       - learned slot aggregation
#   C6  C2 without GRU (plain residual)                - is the GRU necessary?
#
# Same base (350m-mlm), data, steps, lr, max_length as the original sweep;
# seed 1 so each run pairs with B3-latent-s1.
set -u

PY='C:\Users\JingYue\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe'
export HF_ENDPOINT='https://hf-mirror.com'
cd /d/Jaume/code/repo || exit 1

run_arm () {
  local id="$1"; shift
  local out="D:/Jaume/output/mem/${id}-s1"
  if [ -f "${out}/final/config.json" ]; then
    echo "[fix] SKIP ${id} (exists)"
    return 0
  fi
  echo "===================================================="
  echo "[fix] START ${id}  $(date +%H:%M:%S)"
  "$PY" scripts/train_memory.py \
    --config configs/jaume-350m.json --pooling latent_memory --slot-heads 8 \
    --init-from D:/Jaume/350m-mlm --train-data D:/Jaume/data/all_nli.jsonl \
    --output "$out" --seed 1 --epochs 1 --batch-size 32 --grad-accum 4 \
    --lr 3e-4 --max-length 128 --log-every 200 --num-workers 0 "$@" 2>&1 \
    | grep -E 'memory\]|Error|Traceback' | tail -6
  echo "[fix] DONE ${id}  $(date +%H:%M:%S)"
}

run_arm C1-competition       --slot-competition
run_arm C2-competition-iter3 --slot-competition --slot-iterations 3
run_arm C3-comp-iter3-div    --slot-competition --slot-iterations 3 --slot-div-weight 0.1
run_arm C4-comp-iter3-16slot --slot-competition --slot-iterations 3 --num-slots 16
run_arm C5-comp-attnpool     --slot-competition --slot-iterations 3 --slot-pool attention
run_arm C6-comp-iter3-nogru  --slot-competition --slot-iterations 3 --no-slot-gru

echo "[fix] ALL DONE $(date +%H:%M:%S)"
