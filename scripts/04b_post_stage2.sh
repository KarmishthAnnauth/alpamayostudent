#!/bin/bash
# Post-run checks for a stage-2 run: val minADE, CoC regeneration + scoring, open-loop L2.
# Usage: sbatch scripts/04b_post_stage2.sh <run_name>
#SBATCH --job-name=a2e-post-stage2
#SBATCH --partition=main
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
RUN=${1:?run_name}
echo "Job ${SLURM_JOB_ID:-<none>} on $(hostname), started $(date -Is); commit $(git rev-parse --short HEAD); run $RUN"
source ~/envs/alpamayo2edge/bin/activate
export HF_TOKEN="$(cat ~/.config/alpamayo2edge/hf_token)"
export HF_HUB_CACHE=/bulk/users/$USER/alpamayo2edge/hf-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
BW_UUID="$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader \
           | awk -F', ' '/RTX PRO 6000/ && !d {print $1; d=1}')"
export CUDA_VISIBLE_DEVICES="$BW_UUID"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
CK=/data/vla/alpamayo2edge/runs/stage2/$RUN/best
[ -e "$CK" ] || { echo "no best checkpoint under $CK" >&2; exit 1; }
echo "best -> $(readlink -f "$CK")"
python3 scripts/05k_flow_minade.py --ckpt "$CK" --split val --out "runs/flow-minade-$RUN-best-valfull.json"
COC="runs/coc_probe/coc_$RUN-best_val.jsonl"
rm -f "$COC"
python3 scripts/05j_b2d_coc.py generate --ckpt "$CK" --split val --out "$COC"
python3 scripts/05j_b2d_coc.py score --split val --out "$COC" --json "runs/coc_probe/coc_$RUN-best_val-score.json"
python3 scripts/05m_sample_select.py --ckpt "$CK" --ks 1 --out "runs/sampler_probe/$RUN-best-b2dL2-val.json"
