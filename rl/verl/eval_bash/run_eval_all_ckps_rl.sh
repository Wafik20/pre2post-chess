#!/bin/bash
# =============================================================================
# Run evaluation across all checkpoints of one RL run
# =============================================================================
#
# Drives verl_eval.sh once per checkpoint step. Continues even if one step fails.
#
# Usage:
#   CHECKPOINT_BASE=/path/to/checkpoints_hf_format bash run_eval_all_ckps_rl.sh
#   STEPS="0 100 200" TEMPERATURE=0.6 N_SAMPLES=8 bash run_eval_all_ckps_rl.sh
#   EVAL_DATASETS="test_B1_multi_turn" bash run_eval_all_ckps_rl.sh
#
# =============================================================================

# Don't use set -e - we want to continue even if one checkpoint fails

# Script directory (defined early for path resolution)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"   # = the verl package root (rl/verl)

# Base path to the HF-format checkpoints of the run to evaluate. Each step is
# expected at $CHECKPOINT_BASE/global_step_<N>. Override via env var.
CHECKPOINT_BASE=${CHECKPOINT_BASE:?"Set CHECKPOINT_BASE to a checkpoints_hf_format directory"}

# Export settings for verl_eval.sh (can override via env vars)
export RES_LENGTH=${RES_LENGTH:-2560}
export TEMPERATURE=${TEMPERATURE:-1}
export N_SAMPLES=${N_SAMPLES:-16}

# Auto-generate output directory name from settings
EXPERIMENT_BASE=$(dirname "$CHECKPOINT_BASE")
# Convert response length to short form (8192 -> 8k, 16384 -> 16k)
RES_LEN_SHORT=$((RES_LENGTH / 1024))k
# Convert temperature to string (1 -> t1, 0.6 -> t0_6)
TEMP_STR=$(echo "$TEMPERATURE" | sed 's/\./_/')
# Build output dir name
export OUTPUT_DIR=${OUTPUT_DIR:-"$EXPERIMENT_BASE/eval_results_${RES_LEN_SHORT}_t${TEMP_STR}_n${N_SAMPLES}"}

# Checkpoint steps to evaluate (space-separated). Override via STEPS env var.
STEPS=${STEPS:-"100 120"}
EVAL_DATASETS=${EVAL_DATASETS:-"test_B1_multi_turn,test_B2_multi_turn,test_B3_multi_turn,test_B4_multi_turn,test_B5_multi_turn"}
# Evaluation data directory (not shipped with this code release; see README).
export EVAL_DATA_DIR=${EVAL_DATA_DIR:-"$WORKSPACE_DIR/data/eval_thinking/"}
export GPUS=${GPUS:-"0"}
export N_GPUS=${N_GPUS:-1}
export GPU_MEMORY=${GPU_MEMORY:-0.8}
export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-32}

echo "=============================================="
echo "  Running Evaluation Across All Checkpoints"
echo "=============================================="
echo "Checkpoints:  $CHECKPOINT_BASE"
echo "Steps:        $STEPS"
echo "Datasets:     $EVAL_DATASETS"
echo "Generation:   ${RES_LENGTH} tokens, temp=${TEMPERATURE}, n=${N_SAMPLES}"
echo "GPUs:         $GPUS ($N_GPUS)"
echo "=============================================="
echo ""

# Track progress
TOTAL=$(echo "$STEPS" | wc -w)
CURRENT=0
FAILED_STEPS=""
SUCCEEDED_STEPS=""

for STEP in $STEPS; do
    CURRENT=$((CURRENT + 1))
    MODEL_PATH="$CHECKPOINT_BASE/global_step_$STEP"

    echo ""
    echo "=============================================="
    echo "  [$CURRENT/$TOTAL] Evaluating: global_step_$STEP"
    echo "=============================================="

    if [ ! -d "$MODEL_PATH" ]; then
        echo "WARNING: Checkpoint not found: $MODEL_PATH"
        echo "Skipping..."
        FAILED_STEPS="$FAILED_STEPS $STEP(not_found)"
        continue
    fi

    # Run evaluation (continue even if it fails)
    export MODEL_PATH
    if bash "$SCRIPT_DIR/verl_eval.sh"; then
        echo ""
        echo "Completed: global_step_$STEP"
        SUCCEEDED_STEPS="$SUCCEEDED_STEPS $STEP"
    else
        echo ""
        echo "FAILED: global_step_$STEP"
        FAILED_STEPS="$FAILED_STEPS $STEP"
    fi

    # Brief pause to allow GPU memory cleanup between runs
    echo "Waiting 10s for GPU cleanup..."
    sleep 10
    echo ""
done

echo ""
echo "=============================================="
echo "  All evaluations complete!"
echo "=============================================="
echo ""
echo "Succeeded:$SUCCEEDED_STEPS"
if [ -n "$FAILED_STEPS" ]; then
    echo "Failed:$FAILED_STEPS"
fi
echo ""
