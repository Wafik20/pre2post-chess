#!/bin/bash

# Downloads the SFT model folders used as RL starting points by
# sweep_multi_turn.sh, one per pretrain spec.
#
# Set HF_REPO to the HuggingFace repo that holds your SFT checkpoints
# (organized as <repo>/<model_id>/...).

set -euo pipefail

HF_REPO="${HF_REPO:-<your-hf-org>/sft_trajectory_no_labels}"
COT_TYPE="${COT_TYPE:-trajectory_sep_no_labels}"
SFT_ROOT="${SFT_ROOT:-../../sft_checkpoints/sft}"
LOCAL_BASE="${SFT_ROOT}/${COT_TYPE}"

pretrain_specs=(
  "6p5e18|200m|0.050|0.045"
  "6p5e18|200m|0.200|0.045"
  "6p5e18|200m|0.400|0.045"
  "6p5e18|200m|0.750|0.045"
  "6p5e18|200m|0.950|0.045"
  "6p5e18|50m|0.750|0.045"
  "6p5e18|50m|0.400|0.045"
)

model_id_from_spec() {
  printf "C%s_%s_alpha%s_beta%s" "$1" "$2" "$3" "$4"
}

mkdir -p "${LOCAL_BASE}"

for spec in "${pretrain_specs[@]}"; do
  IFS="|" read -r total_compute modelsize alpha beta <<<"${spec}"
  model_id="$(model_id_from_spec "${total_compute}" "${modelsize}" "${alpha}" "${beta}")"
  dest="${LOCAL_BASE}/${model_id}"

  echo "[download] ${HF_REPO}/${model_id}  ->  ${dest}"

  huggingface-cli download "${HF_REPO}" \
    --include "${model_id}/**" \
    --local-dir "${LOCAL_BASE}" \
    --repo-type model

  echo "[done]     ${model_id}"
done

echo "[download] All models downloaded to ${LOCAL_BASE}"
