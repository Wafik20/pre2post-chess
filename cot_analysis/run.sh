#!/usr/bin/env bash
# Single-entry replication script.
#
# Inputs (override via env vars):
#   CHECKPOINTS  root directory containing sft/<tag>/eval_results and
#                rl/<tag>/eval_results_2k_t1_n16/ subtrees   (default: ../checkpoints)
#   RUN_TAG      which run folder under both sft/ and rl/    (default: C6p5e18_20m_alpha0.200_beta0.008)
#   OUT          output directory for parquets and figures   (default: ../results/<RUN_TAG>)
#   STOCKFISH    path to Stockfish binary                    (default: stockfish on PATH)
#   STOCKFISH_DEPTH  search depth                            (default: 8)
#   N_WORKERS    parallel Stockfish processes                (default: 8)
#   GRID         "rows cols" for the figure grid             (default: 1xN strip)
#
# Example:
#   STOCKFISH=/opt/stockfish CHECKPOINTS=./checkpoints bash run.sh

set -euo pipefail
cd "$(dirname "$0")"

CHECKPOINTS=${CHECKPOINTS:-../checkpoints}
RUN_TAG=${RUN_TAG:-C6p5e18_20m_alpha0.200_beta0.008}
OUT=${OUT:-../results/${RUN_TAG}}
STOCKFISH=${STOCKFISH:-stockfish}
STOCKFISH_DEPTH=${STOCKFISH_DEPTH:-8}
N_WORKERS=${N_WORKERS:-8}

echo "=== run.sh ==="
echo "  CHECKPOINTS   = $CHECKPOINTS"
echo "  RUN_TAG       = $RUN_TAG"
echo "  OUT           = $OUT"
echo "  STOCKFISH     = $STOCKFISH (depth $STOCKFISH_DEPTH)"
echo "  N_WORKERS     = $N_WORKERS"
echo

python analyze.py \
    --checkpoints "$CHECKPOINTS" \
    --run-tag     "$RUN_TAG" \
    --stockfish   "$STOCKFISH" \
    --stockfish-depth "$STOCKFISH_DEPTH" \
    --n-workers   "$N_WORKERS" \
    --out         "$OUT"

GRID_ARG=()
if [[ -n "${GRID:-}" ]]; then
    GRID_ARG=(--grid $GRID)
fi

python plot.py \
    --data "$OUT" \
    --tags "$RUN_TAG" \
    --out  "$OUT" \
    "${GRID_ARG[@]}"

echo
echo "done — see $OUT/"
