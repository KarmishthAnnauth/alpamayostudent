#!/bin/bash
# Phase 1: supervised distillation of the student's reasoning tower.
#   bash scripts/train_phase1.sh [--micro-batch N --grad-accum M]
set -euo pipefail

MB=""
for ((i = 1; i <= $#; i++)); do
    [ "${!i}" = "--micro-batch" ] && { j=$((i + 1)); MB="${!j}"; }
done

cd "$(dirname "$0")/.."
export PYTHONPATH="src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
mkdir -p logs
LOG="logs/phase1-$(date +%Y%m%d-%H%M%S).out"
exec > >(tee -a "$LOG") 2>&1

eval "$(python - <<'PY'
from distill.config import load_config
p = load_config().paths
print(f'CACHE="{p.cache_root}"; STUDENT="{p.student_repo}"; RUNS="{p.runs_root}"')
PY
)"

echo "== $(date -Is) commit=$(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
echo "   cache:   $CACHE"
echo "   student: $STUDENT"
echo "   runs:    $RUNS"
echo "   log:     $LOG"

[ -f "$STUDENT/config.json" ] || { echo "missing student at $STUDENT - run scripts/download_models.py" >&2; exit 1; }
for f in traj_tokenizer_spec.pt expert_layers.pt split_train.json split_val.json split_challenging.json; do
    [ -f "$CACHE/$f" ] || { echo "missing $CACHE/$f - run scripts/prepare_phase1_data.sh" >&2; exit 1; }
done
# ~60 GiB at the default micro-batch of 4.
MB="$MB" python - <<'PY'
import os, sys, torch
if not torch.cuda.is_available():
    sys.exit("no CUDA GPU visible")
p = torch.cuda.get_device_properties(0)
free = torch.cuda.mem_get_info()[0] / 2**30
print(f"   gpu:     {p.name}, {free:.0f} GiB free")
if not os.environ.get("MB") and free < 60:
    sys.exit(f"need ~60 GiB free at micro-batch 4; pass e.g. "
             f"--micro-batch 2 --grad-accum 16 to fit {free:.0f} GiB")
PY

echo "== training"
python -m distill.train_stage1 "$@"
echo "== $(date -Is) done; best checkpoint: $RUNS/stage1/best"
