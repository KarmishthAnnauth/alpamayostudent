#!/bin/bash
# Build the labelled PhysicalAI-AV cache for phase 1 / 1.5.
#   bash scripts/prepare_phase1_data.sh [N=5000] [LIMIT]
set -euo pipefail

N="${1:-5000}"
LIMIT="${2:-}"

cd "$(dirname "$0")/.."
export PYTHONPATH="src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

eval "$(python - <<'PY'
from distill.config import load_config
p = load_config().paths
print(f'CACHE="{p.cache_root}"; TEACHER="{p.teacher_repo}"; STUDENT="{p.student_repo}"')
PY
)"

echo "== $(date -Is) N=$N LIMIT=${LIMIT:-none} commit=$(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
echo "   cache:   $CACHE"
echo "   teacher: $TEACHER"
echo "   student: $STUDENT"

for d in "$TEACHER" "$STUDENT"; do
    [ -f "$d/config.json" ] || { echo "missing model at $d - run scripts/download_models.py" >&2; exit 1; }
done
if [ -z "${HF_TOKEN:-}" ] && ! python -c "from huggingface_hub import get_token; assert get_token()" 2>/dev/null; then
    echo "no Hugging Face token: export HF_TOKEN or run 'hf auth login'" >&2; exit 1
fi
# Teacher needs ~23 GiB free.
python - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    sys.exit("no CUDA GPU visible")
p = torch.cuda.get_device_properties(0)
free = torch.cuda.mem_get_info()[0] / 2**30
print(f"   gpu:     {p.name}, {free:.0f} GiB free")
if free < 23:
    sys.exit(f"need >23 GiB free on {p.name}, have {free:.0f}")
PY

mkdir -p "$CACHE"
if [ -f "$CACHE/traj_tokenizer_spec.pt" ] && [ -f "$CACHE/expert_layers.pt" ]; then
    echo "== 00_verify: done already"
else
    python scripts/00_verify.py
fi

if [ -f "$CACHE/curated_$N.json" ]; then
    echo "== 01_curate: curated_$N.json exists"
else
    python scripts/01_curate.py --n "$N"
fi

# Resumable: skips windows already labelled.
python scripts/02_label.py --n "$N" ${LIMIT:+--limit "$LIMIT"}
echo "== $(date -Is) phase 1 data ready in $CACHE"
