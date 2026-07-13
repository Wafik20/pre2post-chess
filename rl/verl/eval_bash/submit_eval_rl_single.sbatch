#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200|h100|a100
#SBATCH --time=04:00:00
#SBATCH --mem=300G
#SBATCH --output=slurm_logs/chess_eval_rl_%j.log
#SBATCH --account=<your-slurm-account>
#SBATCH --mail-type=END

# SLURM wrapper that evaluates every checkpoint of one RL run.
#
# Pass CHECKPOINT_BASE (and optionally OUTPUT_DIR / RES_LENGTH / TEMPERATURE /
# N_SAMPLES / STEPS) via --export, e.g.:
#   sbatch --export=ALL,CHECKPOINT_BASE=/path/to/checkpoints_hf_format \
#          submit_eval_rl_single.sbatch
#
# CHECKPOINT_BASE should point to the checkpoints_hf_format/ directory for a run.

echo "Starting job $SLURM_JOB_NAME (ID: $SLURM_JOB_ID)"
echo "CHECKPOINT_BASE: $CHECKPOINT_BASE"

source ~/.bashrc

# Package root = two levels up from this script (rl/verl/eval_bash -> rl/verl).
PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PKG_ROOT}"
mkdir -p slurm_logs

mkdir -p "${SCRATCH}/.triton-cache" "${SCRATCH}/torchinductor"
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

conda activate "${CONDA_ENV:-chess_rl}"

echo "Activated conda environment"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

if [ -z "${CHECKPOINT_BASE:-}" ]; then
    echo "ERROR: CHECKPOINT_BASE not set. Pass it via --export."
    exit 1
fi

# Auto-discover HF checkpoint steps when the caller hasn't pre-set STEPS.
if [ -z "${STEPS:-}" ]; then
    STEPS=$(ls -d "$CHECKPOINT_BASE"/global_step_* 2>/dev/null \
        | xargs -n1 basename | sed 's/global_step_//' | sort -n | tr '\n' ' ')
fi
export STEPS

if [ -z "$STEPS" ]; then
    echo "ERROR: No global_step_* directories found under $CHECKPOINT_BASE"
    exit 1
fi
echo "Steps to evaluate: $STEPS"

# OUTPUT_DIR / RES_LENGTH / TEMPERATURE / N_SAMPLES may be passed in; otherwise
# run_eval_all_ckps_rl.sh applies its own defaults.
export CHECKPOINT_BASE
[ -n "${OUTPUT_DIR:-}" ] && export OUTPUT_DIR
export RES_LENGTH TEMPERATURE N_SAMPLES

bash eval_bash/run_eval_all_ckps_rl.sh

echo ""
echo "Done."
