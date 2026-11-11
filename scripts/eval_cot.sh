#!/bin/bash
# Score the student's and teacher's chain-of-thought against the driver on the validation set.
#   bash scripts/eval_cot.sh phase1   [--ckpt PATH]
#   bash scripts/eval_cot.sh phase1_5 [--step N | --ckpt PATH] [--init-ckpt PATH]
#   options: --split val  --n 100000 (all windows)
set -euo pipefail

PHASE="${1:-}"
case "$PHASE" in phase1|phase1_5) shift ;; *) echo "usage: $0 phase1|phase1_5 [options]" >&2; exit 1 ;; esac

CKPT="" INIT="" STEP="" ADAPTERS="" SPLIT="val" N=100000
while [ $# -gt 0 ]; do
    case "$1" in
        --ckpt) CKPT="$2"; shift 2 ;;
        --init-ckpt) INIT="$2"; shift 2 ;;
        --step) STEP="$2"; shift 2 ;;
        --split) SPLIT="$2"; shift 2 ;;
        --n) N="$2"; shift 2 ;;
        *) echo "unknown option $1" >&2; exit 1 ;;
    esac
done

cd "$(dirname "$0")/.."
export PYTHONPATH="src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

eval "$(python - <<'PY'
from distill.config import load_config
c = load_config()
print(f'RUNS="{c.paths.runs_root}"; RL_INIT="{c.stage1_rl.init_ckpt}"; RUN_NAME="{c.stage1_rl.run_name}"')
PY
)"

ARGS=()
if [ "$PHASE" = phase1 ]; then
    CKPT="${CKPT:-$RUNS/stage1/best}"
    NAME="phase1"
elif [ -n "$STEP" ]; then
    # GRPO adapters from one evaluation step, on top of the phase 1 checkpoint.
    CKPT="${INIT:-$RL_INIT}"
    ADAPTERS="$RUNS/stage1_rl/$RUN_NAME/step-$(printf %04d "$STEP")"
    [ -d "$ADAPTERS" ] || { echo "missing adapters at $ADAPTERS" >&2; exit 1; }
    ARGS+=(--adapters "$ADAPTERS")
    NAME="phase1_5-step$(printf %04d "$STEP")"
else
    CKPT="${CKPT:-$RUNS/stage1_rl/$RUN_NAME/best}"
    NAME="phase1_5"
fi
[ -e "$CKPT" ] || { echo "missing checkpoint at $CKPT" >&2; exit 1; }

OUT="$RUNS/eval"
mkdir -p "$OUT"
DUMP="$OUT/cot-$NAME-$SPLIT.jsonl"

echo "== $(date -Is) $NAME on $SPLIT"
echo "   checkpoint: $CKPT${ADAPTERS:+ + $ADAPTERS}"
echo "   output:     $DUMP"

python scripts/05b_eval_coc.py --ckpt "$CKPT" ${ARGS[@]+"${ARGS[@]}"} --split "$SPLIT" --n "$N" \
    --route-hint --dump "$DUMP" --out "$OUT/cot-$NAME-$SPLIT.json"
python scripts/05f_coc_gt_score.py --dump "$DUMP" --split "$SPLIT" | tee "$OUT/cot-$NAME-$SPLIT-driver.txt"
