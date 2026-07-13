#!/bin/bash
# Generate Chain-of-Thought (CoT) SFT data by sharded puzzle-sequence search
# (plain shell, no SLURM).
#
# Usage:
#   bash generate_cot.sh [stockfish|random|model]
#
# This is the non-SLURM port of submit_sharded_jobs_puzzle_seq.sh. The original
# submitted one sbatch job per shard, each running run_puzzle_seq_<mode>.sbatch;
# here the per-shard python command is inlined and run in the foreground loop.
#
# Override paths/knobs via environment variables:
#   DATA_PATH       input CSV of puzzles/games
#   OUTPUT_DIR      where generated CoT shards are written
#   STOCKFISH_PATH  path to the Stockfish binary
#   MODEL_PATH      (model mode) HF policy checkpoint
#   CONFIG_PATH     (model mode) config snapshot for the checkpoint
#   TOTAL_SAMPLES / SAMPLES_PER_JOB / START_OFFSET / NUM_SF_WORKERS

set -euo pipefail

# cot_generation/ is the parent of this scripts/ dir; puzzle_sequence_generator.py lives there.
COT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${COT_DIR}/.." && pwd)"

MODE="${1:-stockfish}"          # stockfish | random | model

TOTAL_SAMPLES="${TOTAL_SAMPLES:-91619}"
SAMPLES_PER_JOB="${SAMPLES_PER_JOB:-500}"
START_OFFSET="${START_OFFSET:-0}"
NUM_SF_WORKERS="${NUM_SF_WORKERS:-12}"   # parallel Stockfish workers for tree annotation
SEEDS=(0)

STOCKFISH_PATH="${STOCKFISH_PATH:-${REPO_ROOT}/Stockfish/src/stockfish}"

# Defaults per mode (override DATA_PATH / OUTPUT_DIR / MODEL_PATH / CONFIG_PATH via env).
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/chess/puzzles_train.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/data/sft/cot_data_puzzle_seq}"
MODEL_PATH="${MODEL_PATH:-}"
CONFIG_PATH="${CONFIG_PATH:-}"
MODEL_TAG="${MODEL_TAG:-policy}"

NUM_JOBS=$(( (TOTAL_SAMPLES - START_OFFSET + SAMPLES_PER_JOB - 1) / SAMPLES_PER_JOB ))
mkdir -p "${OUTPUT_DIR}"
cd "${COT_DIR}"
mkdir -p logs

echo "=========================================="
echo "Puzzle-sequence CoT generation"
echo "Mode: $MODE | Total: $TOTAL_SAMPLES | Per shard: $SAMPLES_PER_JOB | Shards: $NUM_JOBS"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

run_shard() {
  local job_name="$1"; shift
  echo "Running: ${job_name}"
  python puzzle_sequence_generator.py "$@" 2>&1 | tee "logs/${job_name}.out"
  echo "Done: ${job_name}"
}

if [ "$MODE" = "model" ]; then
  if [ -z "$MODEL_PATH" ] || [ -z "$CONFIG_PATH" ]; then
    echo "ERROR: model mode requires MODEL_PATH and CONFIG_PATH env vars"; exit 1
  fi
fi

for SEED in "${SEEDS[@]}"; do
  echo "---- SEED=$SEED ----"
  for i in $(seq 0 $((NUM_JOBS - 1))); do
    SKIP=$((START_OFFSET + i * SAMPLES_PER_JOB))
    REMAINING=$((TOTAL_SAMPLES - SKIP))
    [ $REMAINING -le 0 ] && break
    NUM_SAMPLES=$(( REMAINING < SAMPLES_PER_JOB ? REMAINING : SAMPLES_PER_JOB ))

    if [ "$MODE" = "stockfish" ]; then
      run_shard "puzzle_seq_stf_s${SEED}_k${SKIP}" \
        --data_path "$DATA_PATH" --output_path "$OUTPUT_DIR" \
        --sampling_policy stockfish --candidate_selection policy \
        --num_candidates 8 --num_trajectories 10 --trajectory_depth 20 \
        --inject_answer --max_puzzle_moves 8 \
        --traversal_methods dfs_policy dfs_verifier bfs_policy bfs_verifier \
        --evaluation_policy minimax \
        --num_samples "$NUM_SAMPLES" --num_skip_samples "$SKIP" \
        --stockfish_path "$STOCKFISH_PATH" --stockfish_time 0.2 \
        --stockfish_depth_work 8 --stockfish_depth_ref 18 --stockfish_top_k 15 \
        --stop_entropy 0.4 --budget_mode difficulty_adaptive \
        --budget_min 100 --budget_max 300 --rating_min 500 --rating_max 3000 \
        --win_threshold 300 --num_workers "$NUM_SF_WORKERS" \
        --temperature 1.0 --seed "$SEED" --data_name "puzzle_v4"

    elif [ "$MODE" = "random" ]; then
      run_shard "puzzle_seq_random_s${SEED}_k${SKIP}" \
        --data_path "$DATA_PATH" --output_path "$OUTPUT_DIR" \
        --sampling_policy random --candidate_selection stockfish \
        --num_candidates 10 --num_trajectories 5 --num_trajectories_answer 10 \
        --trajectory_depth 5 --depth_mode random --depth_mean 5.0 --depth_sigma 0.4 \
        --inject_answer --max_puzzle_moves 5 \
        --traversal_methods dfs_policy dfs_verifier bfs_policy bfs_verifier \
        --evaluation_policy minimax \
        --num_samples "$NUM_SAMPLES" --num_skip_samples "$SKIP" \
        --stockfish_path "$STOCKFISH_PATH" --stockfish_time 0.1 \
        --win_threshold 300 --num_workers "$NUM_SF_WORKERS" \
        --seed "$SEED" --data_name "puzzles_train"

    elif [ "$MODE" = "model" ]; then
      run_shard "puzzle_seq_${MODEL_TAG}_s${SEED}_k${SKIP}" \
        --data_path "$DATA_PATH" --output_path "$OUTPUT_DIR" \
        --sampling_policy hf_model --model_path "$MODEL_PATH" --config_path "$CONFIG_PATH" \
        --candidate_selection policy \
        --num_candidates 8 --num_trajectories_min 2 --num_trajectories 10 \
        --num_trajectories_mean 8.0 --num_trajectories_sigma 0.2 --num_trajectories_answer 8 \
        --trajectory_depth 20 --depth_mode random --depth_mean 5.0 --depth_sigma 0.4 \
        --inject_answer --max_puzzle_moves 8 \
        --traversal_methods dfs_policy dfs_verifier bfs_policy bfs_verifier \
        --evaluation_policy minimax \
        --num_samples "$NUM_SAMPLES" --num_skip_samples "$SKIP" \
        --stockfish_path "$STOCKFISH_PATH" --stockfish_time 0.1 \
        --win_threshold 300 --num_workers "$NUM_SF_WORKERS" \
        --temperature 1.0 --device "cuda" --seed "$SEED" --data_name "puzzle_v4"

    else
      echo "ERROR: Unknown MODE='$MODE'. Use 'stockfish', 'random', or 'model'."; exit 1
    fi
  done
done

echo "=========================================="
echo "All shards done. Output in: $OUTPUT_DIR"
echo "=========================================="
