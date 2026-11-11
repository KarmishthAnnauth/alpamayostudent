#!/bin/bash
# Teacher labeling via SLURM, pinned to the Ada. Resumable.
# Usage: sbatch scripts/sbatch_label.sh 5000 [LIMIT]
#SBATCH --job-name=a2e-label
#SBATCH --partition=main
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

N="${1:-5000}"
LIMIT="${2:-}"

cd "${SLURM_SUBMIT_DIR:-$HOME/projects/alpamayo2edge}"
mkdir -p logs
source .venv-a2e/bin/activate
export PYTHONPATH="src:${PYTHONPATH:-}"
export HF_TOKEN="$(cat ~/.config/alpamayo2edge/hf_token)"
export HF_HUB_CACHE=/bulk/users/vla/alpamayo2edge/hf-cache
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Pin the Ada.
ADA_UUID="$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader \
           | awk -F', ' '/RTX 6000 Ada/{print $1; exit}')"
if [ -z "$ADA_UUID" ]; then
    echo "no RTX 6000 Ada visible" >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES="$ADA_UUID"

echo "== $(date -Is) job=${SLURM_JOB_ID:-?} host=$(hostname) restarts=${SLURM_RESTART_COUNT:-0} N=$N LIMIT=${LIMIT:-none}"
echo "== commit $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
nvidia-smi -L
python -c "import torch,sys
p=torch.cuda.get_device_properties(0); free,tot=(x/2**30 for x in torch.cuda.mem_get_info())
print(f'   pinned {p.name}  {tot:.0f} GiB total, {free:.0f} GiB free')
ok = 'Ada' in p.name and free > 23
sys.exit(0 if ok else f'wrong card or low mem: {p.name}, {free:.0f} GiB free')"

CACHE=$(python -c "import sys; sys.path.insert(0,'src')
from distill.config import load_config; print(load_config().paths.cache_root)")
if [ ! -f "$CACHE/curated_$N.json" ]; then
    echo "== curated_$N.json missing - running curation (deterministic, seed=0)"
    python scripts/01_curate.py --n "$N"
fi

python scripts/02_label.py --n "$N" ${LIMIT:+--limit "$LIMIT"}
echo "== $(date -Is) labeling done"
