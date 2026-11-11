#!/bin/bash
# Phase 1.5: GRPO on the chain-of-thought, starting from the phase 1 checkpoint.
#   bash scripts/train_phase1_5.sh [--init-ckpt PATH]
set -euo pipefail

INIT=""
for ((i = 1; i <= $#; i++)); do
    [ "${!i}" = "--init-ckpt" ] && { j=$((i + 1)); INIT="${!j}"; }
done

cd "$(dirname "$0")/.."
export PYTHONPATH="src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
mkdir -p logs
LOG="logs/phase1_5-$(date +%Y%m%d-%H%M%S).out"
exec > >(tee -a "$LOG") 2>&1

eval "$(python - <<'PY'
from distill.config import load_config
c = load_config()
p = c.paths
print(f'CACHE="{p.cache_root}"; STUDENT="{p.student_repo}"; RUNS="{p.runs_root}"; '
      f'DEFAULT_INIT="{c.stage1_rl.init_ckpt}"; RUN_NAME="{c.stage1_rl.run_name}"')
PY
)"
INIT="${INIT:-$DEFAULT_INIT}"

echo "== $(date -Is) commit=$(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
echo "   cache:   $CACHE"
echo "   student: $STUDENT"
echo "   init:    $INIT"
echo "   runs:    $RUNS"
echo "   log:     $LOG"

[ -f "$STUDENT/config.json" ] || { echo "missing student at $STUDENT - run scripts/download_models.py" >&2; exit 1; }
[ -e "$INIT" ] || { echo "missing phase 1 checkpoint at $INIT - run scripts/train_phase1.sh or pass --init-ckpt" >&2; exit 1; }
for f in traj_tokenizer_spec.pt split_train.json split_val.json; do
    [ -f "$CACHE/$f" ] || { echo "missing $CACHE/$f - run scripts/prepare_phase1_data.sh" >&2; exit 1; }
done
# ~60 GiB for groups of 16 rollouts.
python - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    sys.exit("no CUDA GPU visible")
p = torch.cuda.get_device_properties(0)
free = torch.cuda.mem_get_info()[0] / 2**30
print(f"   gpu:     {p.name}, {free:.0f} GiB free")
if free < 60:
    sys.exit(f"need ~60 GiB free, have {free:.0f}")
PY

echo "== training"
python -m distill.train_grpo_coc "$@"
echo "== $(date -Is) done; checkpoints: $RUNS/stage1_rl/$RUN_NAME/"
