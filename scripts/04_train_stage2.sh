#!/bin/bash
# Stage 2: train the flow head on Bench2Drive, then run the post-run checks.
# Usage: sbatch scripts/04_train_stage2.sh  (A2E_POST=0 skips the checks)
#SBATCH --job-name=a2e-stage2
#SBATCH --partition=main
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-22:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
echo "Job ${SLURM_JOB_ID:-<none>} on $(hostname), started $(date -Is)"
echo "Commit:         $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git repo')"
source ~/envs/alpamayo2edge/bin/activate
export HF_TOKEN="$(cat ~/.config/alpamayo2edge/hf_token)"
export HF_HUB_CACHE=/bulk/users/$USER/alpamayo2edge/hf-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
BW_UUID="$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader \
           | awk -F', ' '/RTX PRO 6000/ && !d {print $1; d=1}')"
[ -n "$BW_UUID" ] || { echo "no RTX PRO 6000 visible" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="$BW_UUID"
python3 - <<'GUARD'
import sys, torch
p = torch.cuda.get_device_properties(0)
free, total = (x / 2**30 for x in torch.cuda.mem_get_info())
print(f"   device 0 = {p.name}  {total:.1f} GiB total, {free:.1f} GiB free")
if "RTX PRO 6000" not in p.name:
    sys.exit(f"pinned the wrong card: {p.name}")
if free < 60:
    sys.exit(f"only {free:.1f} GiB free; need ~60 GiB")
GUARD
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
RUN="$(python3 -c "from distill.config import load_config; print(load_config('configs/default.yaml').stage2.get('run_name', 'run'))" 2>/dev/null | tail -1)"
echo "run_name:       $RUN"
python3 -m distill.train_stage2 --config configs/default.yaml "$@"
if [ "${A2E_POST:-1}" = 1 ]; then
    echo "=== post-run checks for $RUN ($(date -Is)) ==="
    bash scripts/04b_post_stage2.sh "$RUN"
fi
