#!/bin/bash
# =============================================================================
# VERL Evaluation Script
# =============================================================================
# Evaluates a single checkpoint on the given datasets using verl's val_only mode.
#
# Usage:
#   MODEL_PATH=/path/to/ckpt bash verl_eval.sh           # evaluate one checkpoint
#   N_GPUS=2 GPUS="4,5" bash verl_eval.sh                # use 2 GPUs
#   N_GPUS=4 GPUS="0,1,2,3" bash verl_eval.sh            # use 4 GPUs
#   RES_LENGTH=32768 N_SAMPLES=8 bash verl_eval.sh       # different generation settings
#   EVAL_DATASETS="test_B1_multi_turn" bash verl_eval.sh # single dataset
#
# This is the single-checkpoint engine. To sweep many checkpoints, drive it
# with run_eval_all_ckps_rl.sh.
#
# Three rollout modes are toggled by env vars (see project README for details):
#   MULTI_TURN (interactive_mode) : True = multi-turn rollout in the chess env;
#                                   the model emits an env-call token, the env
#                                   replies, generation continues. False = one
#                                   single-shot generation per prompt.
#   THINKING   (thinking_mode)    : True = model produces a reasoning / CoT phase
#                                   before its answer. False = answer directly.
#   TTS        (tts_mode)         : True = test-time scaling: force a fixed
#                                   THINKING_TOKEN_BUDGET of thinking tokens
#                                   before the answer phase begins.
# =============================================================================

set -e

# Resolve paths from this script's location (no hardcoded absolute paths).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"   # = the verl package root (rl/verl)

# =============================================================================
#  GPU SETTINGS (Must be set BEFORE anything else)
# =============================================================================

# Prevent Ray from connecting to existing cluster - force local instance
export RAY_ADDRESS=""

# NCCL timeout - 3 hours for 32k generation (default 30min is not enough)
# Read by verl/workers/fsdp_workers.py at init_process_group
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-10800}

# GPU configuration - configurable via env vars
export CUDA_VISIBLE_DEVICES="${GPUS:-0}"
N_GPUS=${N_GPUS:-1}
GPU_MEMORY=${GPU_MEMORY:-0.8}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-8}

# Verify GPUs are visible
echo "Using GPUs: $CUDA_VISIBLE_DEVICES (expecting $N_GPUS GPUs)"

# =============================================================================
#  CONFIGURATION
# =============================================================================

# Model checkpoint path (HF format). Override via env var.
MODEL_PATH=${MODEL_PATH:?"Set MODEL_PATH to an HF-format checkpoint directory"}

# Generation settings
RES_LENGTH=${RES_LENGTH:-1536}           # Response length: 8192, 16384, 32768
TEMPERATURE=${TEMPERATURE:-1}            # 1=sampling enabled
N_SAMPLES=${N_SAMPLES:-8}                # Number of samples per prompt

# Datasets to evaluate (comma-separated; resolved under EVAL_DATA_DIR)
EVAL_DATASETS=${EVAL_DATASETS:-"test_B1_multi_turn,test_B2_multi_turn,test_B3_multi_turn,test_B4_multi_turn,test_B5_multi_turn"}
# Reward function
REWARD_TYPE=${REWARD_TYPE:-"RULE_BASED"}
MULTI_TURN=${MULTI_TURN:-True}
export REWARD_MODEL_TYPE=$REWARD_TYPE

# Rollout backend: vllm (default) or sglang. Opt in with ROLLOUT_NAME=sglang.
ROLLOUT_NAME=${ROLLOUT_NAME:-vllm}

# =============================================================================
#  AUTO-DERIVED PATHS
# =============================================================================

STEP_NUM=$(basename "$MODEL_PATH" | sed 's/global_step_//')
EXPERIMENT_BASE=$(dirname "$(dirname "$MODEL_PATH")")
OUTPUT_DIR=${OUTPUT_DIR:-"$EXPERIMENT_BASE/eval_results"}
# Format: step_50_8192len_t0.6_n8
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"step_${STEP_NUM}_${RES_LENGTH}len_t${TEMPERATURE}_n${N_SAMPLES}"}
THINKING=${THINKING:-"True"}
TTS=${TTS:-"False"}
THINKING_TOKEN_BUDGET=${THINKING_TOKEN_BUDGET-"1024"}
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2048}"
SEED=${SEED:-0}
# =============================================================================
#  TTS-aware response/context length
# =============================================================================
# In TTS mode, the worker (fsdp_workers.py:generate_sequences_tts) interprets
#   total_budget = thinking_budget + answer_length
# and forces the thinking phase to fill thinking_budget tokens before the
# answer phase begins. To keep total context within the model's hard ceiling
# (and to make different thinking budgets a clean thinking/answer split rather
# than a budget-vs-fixed-answer race), set:
#   answer_length = MODEL_MAX - PROMPT_LEN - THINKING_TOKEN_BUDGET
#   max_model_len = MODEL_MAX
# In non-TTS mode, fall back to the original RES_LENGTH semantics.
PROMPT_LEN=${PROMPT_LEN:-512}
MODEL_MAX=${MODEL_MAX:-3072}

if [ "$TTS" = "True" ]; then
    ANS_LEN=$(( MODEL_MAX - PROMPT_LEN - THINKING_TOKEN_BUDGET ))
    if [ "$ANS_LEN" -le 0 ]; then
        echo "ERROR: ANS_LEN=$ANS_LEN <= 0 (MODEL_MAX=$MODEL_MAX, PROMPT_LEN=$PROMPT_LEN, THINKING_TOKEN_BUDGET=$THINKING_TOKEN_BUDGET)"
        echo "Reduce THINKING_TOKEN_BUDGET or raise MODEL_MAX."
        exit 1
    fi
    EFFECTIVE_RESPONSE_LEN=$ANS_LEN
    EFFECTIVE_MAX_MODEL_LEN=$MODEL_MAX
    EFFECTIVE_MAX_BATCHED_TOKENS=$(( 2048 + MODEL_MAX ))
    echo "TTS config: MODEL_MAX=$MODEL_MAX PROMPT_LEN=$PROMPT_LEN THINKING_TOKEN_BUDGET=$THINKING_TOKEN_BUDGET ANS_LEN=$ANS_LEN"
else
    EFFECTIVE_RESPONSE_LEN=$RES_LENGTH
    EFFECTIVE_MAX_MODEL_LEN=$(( PROMPT_LEN + RES_LENGTH ))
    EFFECTIVE_MAX_BATCHED_TOKENS=$(( 2048 + RES_LENGTH ))
fi
# =============================================================================
#  BUILD DATASET PATHS
# =============================================================================

# Evaluation data and reward function live under the package root. The data/
# directory is not shipped with this code release — point EVAL_DATA_DIR at your
# own copy (see the project README).
EVAL_DATA_DIR=${EVAL_DATA_DIR:-"$WORKSPACE_DIR/data/puzzles"}
REWARD_FUNCTION=${REWARD_FUNCTION:-"$WORKSPACE_DIR/reward_function_multiturn.py"}

build_eval_files() {
    local files=""
    IFS=',' read -ra DATASETS <<< "$EVAL_DATASETS"
    for dataset in "${DATASETS[@]}"; do
        dataset=$(echo "$dataset" | xargs)
        local path="$EVAL_DATA_DIR/${dataset}.parquet"
        if [ -f "$path" ]; then
            [ -z "$files" ] && files="'$path'" || files="$files,'$path'"
        else
            echo "WARNING: Dataset not found: $path" >&2
        fi
    done
    echo "[$files]"
}

eval_files=$(build_eval_files)

# =============================================================================
#  SETUP
# =============================================================================

VALIDATION_DIR="$OUTPUT_DIR/$EXPERIMENT_NAME/generations"
LOG_FILE="$OUTPUT_DIR/$EXPERIMENT_NAME/eval.log"

mkdir -p "$VALIDATION_DIR"

# Use V1 engine (faster than XFORMERS fallback to V0)
unset VLLM_ATTENTION_BACKEND

# =============================================================================
#  PRINT CONFIG
# =============================================================================

echo ""
echo "=============================================="
echo "  VERL Evaluation"
echo "=============================================="
echo "Model:        $MODEL_PATH"
echo "Step:         $STEP_NUM"
echo "Output:       $OUTPUT_DIR/$EXPERIMENT_NAME"
echo ""
echo "Generation:   ${RES_LENGTH} tokens, temp=${TEMPERATURE}, n=${N_SAMPLES}"
echo "Datasets:     $EVAL_DATASETS"
echo "GPUs:         $CUDA_VISIBLE_DEVICES ($N_GPUS)"
echo "Rollout:      $ROLLOUT_NAME"
echo "NCCL Timeout: ${NCCL_TIMEOUT}s ($((NCCL_TIMEOUT / 3600))h)"
echo "Reward:       $REWARD_TYPE"
echo "=============================================="
echo ""

# =============================================================================
#  RUN EVALUATION
# =============================================================================

# SGLang launches its scheduler/detokenizer as separate subprocesses with a bare
# python3 that does NOT inherit torch's bundled CUDA libs, so it fails to load
# libcudart.so.12 and the parent hangs until the job times out. Expose the pip
# nvidia/*/lib dirs on LD_LIBRARY_PATH so those subprocesses can find the CUDA
# runtime. Conditional on sglang — vLLM runs in-process and is unaffected.
if [ "$ROLLOUT_NAME" = "sglang" ]; then
    _NV_LIBS=$(python3 -c "import os,glob,nvidia; b=os.path.dirname(nvidia.__file__); print(':'.join(sorted(glob.glob(b+'/*/lib'))))" 2>/dev/null)
    if [ -n "$_NV_LIBS" ]; then
        export LD_LIBRARY_PATH="$_NV_LIBS:${LD_LIBRARY_PATH:-}"
        echo "SGLang: prepended nvidia libs to LD_LIBRARY_PATH"
    fi
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$eval_files \
    data.val_files=$eval_files \
    data.train_batch_size=64 \
    data.max_prompt_length=512 \
    data.trust_remote_code=True \
    data.max_response_length=$EFFECTIVE_RESPONSE_LEN \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.name=$ROLLOUT_NAME \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY \
    actor_rollout_ref.rollout.max_model_len=$EFFECTIVE_MAX_MODEL_LEN \
    actor_rollout_ref.rollout.interactive_mode.enable=$MULTI_TURN \
    actor_rollout_ref.rollout.thinking_mode.enable=$THINKING \
    actor_rollout_ref.rollout.tts_mode.enable=$TTS \
    actor_rollout_ref.rollout.max_num_seqs=$MAX_NUM_SEQS \
    +actor_rollout_ref.rollout.interactive_mode.thinking_budget=$THINKING_TOKEN_BUDGET \
    actor_rollout_ref.model.trust_remote_code=True \
    +actor_rollout_ref.rollout.seed=$SEED \
    actor_rollout_ref.rollout.max_num_batched_tokens=$EFFECTIVE_MAX_BATCHED_TOKENS \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.val_kwargs.n=$N_SAMPLES \
    actor_rollout_ref.rollout.val_kwargs.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.val_kwargs.do_sample=$([ "$TEMPERATURE" != "0" ] && echo "True" || echo "False") \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    algorithm.use_kl_in_reward=False \
    reward_model.enable=False \
    reward_model.reward_manager=batch \
    custom_reward_function.path=$REWARD_FUNCTION \
    custom_reward_function.name=compute_score_batch \
    trainer.val_only=True \
    trainer.val_before_train=True \
    trainer.resume_mode=disable \
    trainer.total_epochs=1 \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.logger='["console"]' \
    trainer.project_name=$EXPERIMENT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.validation_data_dir=$VALIDATION_DIR \
    2>&1 | tee "$LOG_FILE"

# Capture the exit code from the python command (not tee)
EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "Done! Results: $VALIDATION_DIR"
else
    echo "FAILED with exit code $EXIT_CODE"
    echo "Check log: $LOG_FILE"
fi
echo ""

exit $EXIT_CODE
