#!/usr/bin/env bash
# analyze.sh — Stage 2 (metrics + figures).
#
# Reads the raw_scores.parquet files produced by collect.sh and writes the full
# policy-evolution figure pack + summary tables for one model config:
#
#     $OUT_DIR/
#     ├── 01_sharpening_alpha.png        move-space sharpening  (KL-optimal α*)
#     ├── 01b_sharpening_metrics.png     explained-sharpening / residual
#     ├── 01c_sharpening_distributions.png
#     ├── 02_trace_sensitivity.png       TS_rl vs RL step (SFT/pretrain anchors)
#     ├── 03_rl_sft_trace_collapse.png   D_marg, D_trace, C_trace
#     ├── 04_entropy_pretrain_vs_sft.png entropy decomposition
#     ├── 05_category_summary.png        per-state transition categories
#     ├── 06*/07*_category_per_*.png     category counts per bin / per step
#     ├── sharpening_fits.csv            one row per (stage, rl_step)
#     ├── per_step_summary.csv           one row per RL step (all scalars)
#     └── per_state_categories.parquet   one row per (puzzle, rl_step)
#
# CPU-only and fast: per-puzzle "state" objects are cached under $OUT_DIR/cache/,
# so re-runs (e.g. with different --top-k) reuse them instead of re-reading the
# multi-GB raw_scores.
#
# Env vars (override the defaults — these reproduce results/.../policy_analysis_topk3_B5):
#   TOTAL_COMPUTE / MODELSIZE / ALPHA / BETA   config fields  (default 6p5e18 / 50m / 0.200 / 0.023)
#   RESULTS_TAG       suffix on the config id  (default _n128)
#   RESULTS_ROOT      root of the raw_scores tree            (default ./results)
#   RL_STEPS          space-separated RL steps (default: all discovered)
#   TOP_K             k in the top-K transition spec         (default 3)
#   EXAMPLE_BRACKET   restrict rendered examples to one bin  (default test_B5)
#   N_EXAMPLES        example states per category per step   (default 5)
#   EPS_TAIL          tail threshold ε_tail                  (default 0.05)
#   OUT_DIR           output directory   (default $RESULTS_ROOT/<cfg>/policy_analysis)
#   SKIP_CATEGORIES   "true" to skip per-state board renders (faster)
#
# Example (reproduces the paper figure pack):
#   RESULTS_ROOT=./results TOP_K=3 EXAMPLE_BRACKET=test_B5 \
#   RL_STEPS="50 100 250 500 750" bash analyze.sh

set -euo pipefail
cd "$(dirname "$0")"

TOTAL_COMPUTE=${TOTAL_COMPUTE:-6p5e18}
MODELSIZE=${MODELSIZE:-50m}
ALPHA=${ALPHA:-0.200}
BETA=${BETA:-0.023}
RESULTS_TAG=${RESULTS_TAG:-_n128}
RESULTS_ROOT=${RESULTS_ROOT:-./results}
TOP_K=${TOP_K:-3}
EXAMPLE_BRACKET=${EXAMPLE_BRACKET:-test_B5}
N_EXAMPLES=${N_EXAMPLES:-5}
EPS_TAIL=${EPS_TAIL:-0.05}

echo "=== analyze.sh (Stage 2) ==="
echo "  config        = ${TOTAL_COMPUTE}_${MODELSIZE}_alpha${ALPHA}_beta${BETA}${RESULTS_TAG}"
echo "  RESULTS_ROOT  = $RESULTS_ROOT"
echo "  rl_steps      = ${RL_STEPS:-<all discovered>}"
echo "  top_k         = $TOP_K   example_bracket = $EXAMPLE_BRACKET"
echo

extra=()
[[ -n "${RL_STEPS:-}" ]]        && extra+=(--rl-steps ${RL_STEPS})
[[ -n "${OUT_DIR:-}" ]]         && extra+=(--out-dir "$OUT_DIR")
[[ -n "${EXAMPLE_BRACKET:-}" ]] && extra+=(--example-bracket "$EXAMPLE_BRACKET")
[[ "${SKIP_CATEGORIES:-false}" == "true" ]] && extra+=(--skip-categories)

python run_policy_analysis.py \
    --total-compute "$TOTAL_COMPUTE" \
    --modelsize     "$MODELSIZE" \
    --alpha         "$ALPHA" \
    --beta          "$BETA" \
    --results-tag   "$RESULTS_TAG" \
    --results-root  "$RESULTS_ROOT" \
    --top-k         "$TOP_K" \
    --n-examples    "$N_EXAMPLES" \
    --eps-tail      "$EPS_TAIL" \
    "${extra[@]}"

echo
echo "done — see the policy_analysis output dir under $RESULTS_ROOT/"
