#!/usr/bin/env bash
# collect.sh — Stage 1 (data collection).
#
# Teacher-forces every model checkpoint over its own 128-rollout eval and writes
# one raw_scores.parquet per (stage, step):
#
#     $RESULTS_ROOT/$CFG/<stage>/<step>/raw_scores.parquet
#
# These parquet files are the only input Stage 2 (analyze.sh) needs.
#
# What to provide
# ---------------
# A manifest file (default ./manifest.tsv) with one line per checkpoint you want
# scored.  Columns are whitespace-separated; blank lines and #-comments ignored:
#
#     # stage      step        model_path                 rollout_jsonl
#     pretrain     step_final  /ckpts/pretrain            /ckpts/pretrain/0.jsonl
#     sft          step_final  /ckpts/sft                 /ckpts/sft/0.jsonl
#     rl           step_50     /ckpts/rl/global_step_50   /ckpts/rl/step_50/0.jsonl
#     rl           step_100    /ckpts/rl/global_step_100  /ckpts/rl/step_100/0.jsonl
#
#   stage         one of {pretrain, sft, rl}
#   step          subdirectory name; use "step_final" for pretrain/sft and
#                 "step_<N>" for each RL training step
#   model_path    HuggingFace checkpoint dir (must contain config.json + vocab.json)
#   rollout_jsonl the model's 128-rollout eval generation file (generation/0.jsonl)
#
# Env vars (override the defaults):
#   CFG          config id; names the output subtree   (default 6p5e18_50m_alpha0.200_beta0.023_n128)
#   RESULTS_ROOT root of the raw_scores tree            (default ./results)
#   PARQUET_DIR  dir of eval parquets carrying extra_info (FEN, PuzzleId, env_replies, Rating)
#   MANIFEST     manifest file                          (default ./manifest.tsv)
#   BATCH_SIZE   micro-batch for move scoring           (default 256)
#
# Requirements: a GPU, plus `transformers`, `torch`, `python-chess`, `pandas`,
# `pyarrow`.  See README.md for how the 128-rollout evals are produced upstream.
#
# Example:
#   CFG=6p5e18_50m_alpha0.200_beta0.023_n128 \
#   PARQUET_DIR=/data/eval_parquets \
#   MANIFEST=./manifest.tsv bash collect.sh

set -euo pipefail
cd "$(dirname "$0")"

CFG=${CFG:-6p5e18_50m_alpha0.200_beta0.023_n128}
RESULTS_ROOT=${RESULTS_ROOT:-./results}
MANIFEST=${MANIFEST:-./manifest.tsv}
BATCH_SIZE=${BATCH_SIZE:-256}
: "${PARQUET_DIR:?set PARQUET_DIR to the directory of eval parquets (with extra_info)}"

[[ -f "$MANIFEST" ]] || { echo "[collect] manifest not found: $MANIFEST" >&2; exit 1; }

echo "=== collect.sh (Stage 1) ==="
echo "  CFG          = $CFG"
echo "  RESULTS_ROOT = $RESULTS_ROOT"
echo "  PARQUET_DIR  = $PARQUET_DIR"
echo "  MANIFEST     = $MANIFEST"
echo

n=0
while read -r stage step model_path rollout_jsonl _rest; do
    # skip comments / blank lines
    [[ -z "${stage:-}" || "${stage:0:1}" == "#" ]] && continue

    out="$RESULTS_ROOT/$CFG/$stage/$step/raw_scores.parquet"
    if [[ -s "$out" ]]; then
        echo "[skip] $stage/$step (raw_scores already exists: $out)"
        continue
    fi
    echo "[score] $stage/$step"
    echo "        model   = $model_path"
    echo "        rollout = $rollout_jsonl"
    mkdir -p "$(dirname "$out")"
    python measure_policy.py \
        --model-type   "$stage" \
        --model-path   "$model_path" \
        --rollout-data "$rollout_jsonl" \
        --parquet-dir  "$PARQUET_DIR" \
        --output-path  "$out" \
        --batch-size   "$BATCH_SIZE"
    n=$((n + 1))
done < "$MANIFEST"

echo
echo "done — scored $n checkpoint(s).  Raw scores under $RESULTS_ROOT/$CFG/"
echo "Next: run  bash analyze.sh"
