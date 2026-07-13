#!/bin/bash
# Run multi-turn SFT on a pretrained checkpoint (plain shell, no SLURM).
#
# Usage:
#   bash run_sft.sh
#
# Override paths/knobs via environment variables, e.g.:
#   PRETRAIN_ROOT=/data/checkpoints/C_6p5e19 \
#   TRAIN_DATA_DIR=/data/sft/cot_data \
#   NUM_GPUS=2 WANDB_ENTITY=my-team \
#   bash run_sft.sh
#
# This is the non-SLURM port of the original train_sft_multi_turn.sh: instead of
# submitting an sbatch job per (spec, cot_type) it runs `accelerate launch`
# directly in the foreground.

set -euo pipefail

# Repo root = directory of this script (contains config/, scripts/, training/, ...).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# Configuration  (every value below can be overridden via an env variable)
# ============================================================================

# Root holding the pretrain checkpoints. Each model lives at
#   ${PRETRAIN_ROOT}/{total_compute}_{modelsize}_alpha{alpha}/final/
PRETRAIN_ROOT="${PRETRAIN_ROOT:-${REPO_ROOT}/checkpoints/C_6p5e19}"

# Generated CoT SFT data (output of the CoT generation step).
TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-${REPO_ROOT}/data/sft/cot_data_puzzle_seq}"
DATA_NAME="${DATA_NAME:-generated_cot}"

# Weights & Biases team/entity. Leave empty to use your default wandb account.
WANDB_ENTITY="${WANDB_ENTITY:-}"
NUM_GPUS="${NUM_GPUS:-2}"
LR="3e-4"

CONFIG="${REPO_ROOT}/config/configs/qwen_multiturn_sft/sft_config_multi_path_2048.yaml"
MAP_CONFIG_FILE="${REPO_ROOT}/sft_max_train_files_map.json"

# ============================================================================
# What to train
# ============================================================================

# One entry per pretrain checkpoint: "total_compute|modelsize|alpha|beta".
pretrain_specs=(
  "6p5e19|680m|0.750|0.030"
)

# CoT variants to train. The three arrays are index-aligned (one column = one run).
cot_fields=("cot_by_method.trajectory_sep.cot_format_no_labels")
cot_types=("trajectory_sep_no_labels")
block_sizes=("3072")

# ============================================================================
# Helpers
# ============================================================================

model_id_from_spec() {       # SFT run-name id (includes modelsize): compute,size,alpha,beta
  printf "C%s_%s_alpha%s_beta%s" "$1" "$2" "$3" "$4"
}

pretrain_dir_from_spec() {   # pretrain checkpoint dir name: compute,size,alpha
  printf "%s_%s_alpha%s" "$1" "$2" "$3"
}

# Pick the directory that actually contains the HF config.json.
resolve_model_dir() {
  local dir="$1"
  if [[ -f "${dir}/config.json" ]]; then
    echo "${dir}"
  elif [[ -f "${dir}/hf_model/config.json" ]]; then
    echo "${dir}/hf_model"
  else
    echo "[WARN] HF model files not found at ${dir} — trainer will resolve from spec" >&2
    echo "${dir}"
  fi
}

# Echo the first weights file that exists among the candidates (empty if none).
resolve_weights_file() {
  local candidate
  for candidate in "$@"; do
    [[ -f "${candidate}" ]] && { echo "${candidate}"; return; }
  done
}

# Look up max_train_files for a spec from MAP_CONFIG_FILE, falling back to the
# configured default (and warning on stderr) when there is no exact match.
max_train_files_for_spec() {
  python - "$MAP_CONFIG_FILE" "$@" <<'PY'
import json, pathlib, sys

cfg_path = pathlib.Path(sys.argv[1])
total_compute, modelsize, alpha, beta = sys.argv[2:6]
cfg = json.loads(cfg_path.read_text())
spec = [total_compute, modelsize, alpha, beta]

for item in cfg.get("entries_by_spec", []):
    if [str(item.get(k)) for k in ("total_compute", "modelsize", "alpha", "beta")] == spec:
        print(int(item["max_train_files"]))
        break
else:
    default = cfg.get("default_max_train_files")
    if default is None:
        raise SystemExit(f"No max_train_files mapping for ({','.join(spec)})")
    print(f"[WARN] No exact spec mapping for {total_compute}_{modelsize}_alpha{alpha}_"
          f"beta{beta}; using default_max_train_files={int(default)}", file=sys.stderr)
    print(int(default))
PY
}

# ============================================================================
# Main loop: one SFT run per (pretrain spec) x (CoT variant)
# ============================================================================

export PYTHONWARNINGS="ignore"
export NCCL_TIMEOUT=1800000
export TORCH_NCCL_BLOCKING_WAIT=1

for spec in "${pretrain_specs[@]}"; do
  IFS="|" read -r total_compute modelsize alpha beta <<<"${spec}"

  model_id="$(model_id_from_spec "${total_compute}" "${modelsize}" "${alpha}" "${beta}")"
  pretrain_dir="${PRETRAIN_ROOT}/$(pretrain_dir_from_spec "${total_compute}" "${modelsize}" "${alpha}")/final"

  pretrained_model="$(resolve_model_dir "${pretrain_dir}")"
  pretrained_weights="$(resolve_weights_file \
    "${pretrain_dir}/model.safetensors"     "${pretrain_dir}/pytorch_model.bin" \
    "${pretrained_model}/model.safetensors" "${pretrained_model}/pytorch_model.bin")"

  max_train_files="$(max_train_files_for_spec "${total_compute}" "${modelsize}" "${alpha}" "${beta}")"

  for j in "${!cot_fields[@]}"; do
    cot_field="${cot_fields[$j]}"
    cot_type="${cot_types[$j]}"
    block_size="${block_sizes[$j]}"

    # Assemble trainer arguments. Array form keeps this readable and space-safe;
    # optional flags are appended only when their value is non-empty.
    args=(
      --config             "${CONFIG}"
      --lr                 "${LR}"
      --pretrained-model   "${pretrained_model}"
      --naming-scheme      pretrain_spec
      --total-compute      "${total_compute}"
      --modelsize          "${modelsize}"
      --alpha              "${alpha}"
      --beta               "${beta}"
      --cot-field          "${cot_field}"
      --cot-type           "${cot_type}"
      --max-train-files    "${max_train_files}"
    )
    [[ -n "${pretrained_weights}" ]] && args+=(--pretrained-weights "${pretrained_weights}")
    [[ -n "${block_size}" ]]         && args+=(--block-size "${block_size}")
    [[ -n "${TRAIN_DATA_DIR}" ]]     && args+=(--train-files "${TRAIN_DATA_DIR}")
    [[ -n "${DATA_NAME}" ]]          && args+=(--data-name "${DATA_NAME}")
    [[ -n "${WANDB_ENTITY}" ]]       && args+=(--wandb-entity "${WANDB_ENTITY}")

    echo "=========================================="
    echo "[SFT] model_id=${model_id}"
    echo "[SFT] pretrain_dir=${pretrain_dir}"
    echo "[SFT] max_train_files=${max_train_files} cot_type=${cot_type}"
    echo "=========================================="

    accelerate launch --num_processes "${NUM_GPUS}" \
      "${REPO_ROOT}/scripts/train/run_sft.py" "${args[@]}"
  done
done
