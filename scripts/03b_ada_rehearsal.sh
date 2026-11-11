#!/bin/bash
# Stage-1 rehearsal on the shared Ada GPU, outside SLURM.
# Usage: bash scripts/03b_ada_rehearsal.sh
set -euo pipefail

# Effective batch 32.
MB=${MB:-2}
GA=${GA:-16}
# Fraction of the card this process may allocate.
MEM_FRAC=${MEM_FRAC:-0.88}
# Refuse to start if others hold more than this (MiB).
OTHERS_MAX_MIB=${OTHERS_MAX_MIB:-4000}

echo "Ada rehearsal, started $(date -Is)"
echo "Commit:  $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git repo')"
echo "Batch:   micro_batch $MB x grad_accum $GA = $((MB * GA))"
echo "Cap:     ${MEM_FRAC} of the card"
echo "----------------------------------------------------------------"

source ~/envs/alpamayo2edge/bin/activate
export HF_TOKEN="$(cat ~/.config/alpamayo2edge/hf_token)"
export HF_HUB_CACHE=/bulk/users/$USER/alpamayo2edge/hf-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}

ADA_UUID="$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader \
            | awk -F', ' '/RTX 6000 Ada/ && !d {print $1; d=1}')"
if [ -z "$ADA_UUID" ]; then
    echo "no RTX 6000 Ada visible" >&2
    exit 1
fi

# Everyone else's footprint on the Ada, before we add ours.
OTHERS_MIB="$(nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader \
              | awk -F', ' -v u="$ADA_UUID" '$1==u {s += $2} END {print s+0}')"
echo "others currently on the Ada: ${OTHERS_MIB} MiB"
if [ "$OTHERS_MIB" -gt "$OTHERS_MAX_MIB" ]; then
    echo "REFUSING: others hold ${OTHERS_MIB} MiB (> ${OTHERS_MAX_MIB})." >&2
    echo "Someone else's job arrived after this run was sized. Wait for the" >&2
    echo "Blackwell instead of squeezing them - see the header." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$ADA_UUID"
export A2E_MEM_FRAC="$MEM_FRAC"

python3 - <<'GUARD'
import os, sys, torch
p = torch.cuda.get_device_properties(0)
free, total = (x / 2**30 for x in torch.cuda.mem_get_info())
print(f"   device 0 = {p.name}  {total:.1f} GiB total, {free:.1f} GiB free")
if "RTX 6000 Ada" not in p.name:
    sys.exit(f"pinned the wrong card: {p.name}")
GUARD

# Memory cap applied via sitecustomize shim.
export PYTHONPATH="src:scripts/_ada_shim${PYTHONPATH:+:$PYTHONPATH}"

echo "----------------------------------------------------------------"
python3 -m distill.train_stage1 --config configs/default.yaml \
        --micro-batch "$MB" --grad-accum "$GA" "$@"
echo "----------------------------------------------------------------"
echo "Finished $(date -Is)"
