#!/bin/bash
# Teacher labeling without SLURM, pinned to the Ada. Resumable.
# Usage: nohup bash scripts/run_label.sh 5000 > logs/label.out 2>&1 &
set -euo pipefail

N="${1:-5000}"
LIMIT="${2:-}"

cd "$(dirname "$0")/.."
mkdir -p logs
source .venv-a2e/bin/activate
export PYTHONPATH="src:${PYTHONPATH:-}"
export HF_TOKEN="$(cat ~/.config/alpamayo2edge/hf_token)"
export HF_HUB_CACHE=/bulk/users/vla/alpamayo2edge/hf-cache
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# Pin the Ada (the Blackwell is reserved for training).
ADA_UUID="$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader \
           | awk -F', ' '/RTX 6000 Ada/{print $1; exit}')"
if [ -z "$ADA_UUID" ]; then
    echo "no RTX 6000 Ada visible" >&2; exit 1
fi
export CUDA_VISIBLE_DEVICES="$ADA_UUID"

echo "== $(date -Is) host=$(hostname) pid=$$ N=$N LIMIT=${LIMIT:-none}"
echo "== commit $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
python -c "import torch,sys
p=torch.cuda.get_device_properties(0); free,tot=(x/2**30 for x in torch.cuda.mem_get_info())
print(f'   pinned {p.name}  {tot:.0f} GiB total, {free:.0f} GiB free')
ok = 'Ada' in p.name and free > 23
sys.exit(0 if ok else f'need >23 GiB free on the Ada, have {free:.0f}')"

CACHE=$(python -c "import sys; sys.path.insert(0,'src')
from distill.config import load_config; print(load_config().paths.cache_root)")
[ -f "$CACHE/curated_$N.json" ] || python scripts/01_curate.py --n "$N"

python scripts/02_label.py --n "$N" ${LIMIT:+--limit "$LIMIT"}
echo "== $(date -Is) labeling done"
