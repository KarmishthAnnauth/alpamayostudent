#!/bin/bash
# Stage 1: distill the teacher reasoner into the student's AR tower.
#     sbatch scripts/03_train_stage1.sh
#SBATCH --job-name=a2e-stage1
#SBATCH --partition=main
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-20:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

echo "Job ${SLURM_JOB_ID:-<none>} on $(hostname), started $(date -Is)"
echo "Submitted from: ${SLURM_SUBMIT_DIR:-$PWD}"
echo "GPU:            $(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd', ')"
echo "Commit:         $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git repo')"
echo "Scratch:        ${SLURM_SCRATCH:-<none>}"
echo "----------------------------------------------------------------"

source ~/envs/alpamayo2edge/bin/activate
export HF_TOKEN="$(cat ~/.config/alpamayo2edge/hf_token)"
export HF_HUB_CACHE=/bulk/users/$USER/alpamayo2edge/hf-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Pin the Blackwell by UUID.
BW_UUID="$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader \
           | awk -F', ' '/RTX PRO 6000/ && !d {print $1; d=1}')"
if [ -z "$BW_UUID" ]; then
    echo "no RTX PRO 6000 visible - is this a lab account?" >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES="$BW_UUID"

python3 - <<'GUARD'
import os, sys, torch
print("   CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("   CPUs allocated       =", os.environ.get("SLURM_CPUS_PER_TASK"))
if not torch.cuda.is_available():
    sys.exit("no CUDA device visible to torch")
p = torch.cuda.get_device_properties(0)
free, total = (x / 2**30 for x in torch.cuda.mem_get_info())
print(f"   device 0             = {p.name}  {total:.1f} GiB total, {free:.1f} GiB free")
if "RTX PRO 6000" not in p.name:
    sys.exit(f"pinned the wrong card: {p.name}")
if free < 65:
    sys.exit(f"only {free:.1f} GiB free on {p.name}; need ~65 GiB "
             f"(job 232 peak 58.3 GiB at micro_batch 4, 4 cameras).")
GUARD

# Best-effort W&B log tail.
JOB_LOG="logs/${SLURM_JOB_NAME:-a2e-stage1}-${SLURM_JOB_ID:-manual}.out"
WANDB_RUN="stage1-job${SLURM_JOB_ID:-manual}"
if python3 -c "import wandb" 2>/dev/null; then
    python3 scripts/wandb_tail.py "$JOB_LOG" --name "$WANDB_RUN" &
    WANDB_TAIL_PID=$!
    finish_wandb() {
        kill "$WANDB_TAIL_PID" 2>/dev/null || true
        wait "$WANDB_TAIL_PID" 2>/dev/null || true
        python3 scripts/wandb_tail.py "$JOB_LOG" --name "$WANDB_RUN" --once || true
    }
    trap finish_wandb EXIT
    echo "wandb sidecar: run $WANDB_RUN (pid $WANDB_TAIL_PID)"
else
    echo "wandb sidecar: skipped (wandb not importable)"
fi

echo "----------------------------------------------------------------"
PYTHONPATH=src python3 -m distill.train_stage1 --config configs/default.yaml
echo "----------------------------------------------------------------"
echo "Finished $(date -Is)"
echo "Check what this actually used:  seff ${SLURM_JOB_ID:-<jobid>}"
