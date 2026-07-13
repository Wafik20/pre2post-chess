#!/usr/bin/env bash
# Download HuggingFace checkpoints for multiple model sizes and alphas.
# Repo format: chess-pre-to-post/C9e18_{size}_alpha{alpha}_beta{beta}
#
# Usage:
#   ./download_hf_checkpoints.sh
#
# Edit the arrays below to specify which sizes, alphas, and betas to download.
#
# huggingface-cli download is incremental (checks file hashes), so only
# new/changed files are fetched even when the local folder already exists.
# fix_tokenizer.py is run after every download to patch tokenizer.py files;
# it safely skips files that are already patched.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIX_TOKENIZER="$SCRIPT_DIR/fix_tokenizer.py"
FIX_ARCHITECTURE="$SCRIPT_DIR/fix_architecture.py"
FIX_ROPE_THETA="$SCRIPT_DIR/fix_rope_theta.py"

# ── Configuration ────────────────────────────────────────────────────────────

HF_ORG="chess-pre-to-post"
# TOTAL_COMPUTE=6p5e19
TOTAL_COMPUTE=6p5e18
LOCAL_BASE="./checkpoints/C_${TOTAL_COMPUTE}"

# Model sizes (e.g. 400m, 1b, 7b)
# SIZES=("50m" "100m" "200m" "410m" "680m") #"400m" "800m" "200m"  "100m" "410m" "50m"  "410m"   "200m" "100m" "50m"
# SIZES=("50m" "32m" "20m" "5m" "10m")
SIZES=("20m")
#"5m" 
# Alpha values — use the exact string that appears in the repo name
# ALPHAS=("0.050" "0.200"  "0.400" ) #"0.049" "0.970" "0.750" "0.950"
# ALPHAS=("0.200"  "0.400" "0.050" "0.750" "1.000") #"0.950" "0.050" "0.200"  "0.400"


# ── Download loop ─────────────────────────────────────────────────────────────

mkdir -p "$LOCAL_BASE"

for size in "${SIZES[@]}"; do
  for alpha in "${ALPHAS[@]}"; do
    REPO_NAME_1="${TOTAL_COMPUTE}_${size}_alpha${alpha}"
    REPO_NAME="${size}_C_${TOTAL_COMPUTE}_alpha${alpha}"
    REPO_ID="${HF_ORG}/${REPO_NAME}"
    LOCAL_DIR="${LOCAL_BASE}/${REPO_NAME_1}"

    if [[ -d "$LOCAL_DIR" ]]; then
      echo "[update] $REPO_ID — syncing updates into $LOCAL_DIR"
    else
      echo "[download] $REPO_ID → $LOCAL_DIR"
    fi
    mkdir -p "$LOCAL_DIR"

    hf download \
      "$REPO_ID" \
      --repo-type model \
      --local-dir "$LOCAL_DIR" \

    echo "[patch] applying fix_tokenizer.py to $LOCAL_DIR"
    python3 "$FIX_TOKENIZER" "$LOCAL_DIR"

    echo "[patch] applying fix_architecture.py to $LOCAL_DIR"
    python3 "$FIX_ARCHITECTURE" "$LOCAL_DIR"

    echo "[patch] applying fix_rope_theta.py to $LOCAL_DIR"
    python3 "$FIX_ROPE_THETA" "$LOCAL_DIR"

    echo "[done] $REPO_NAME"
    echo ""
  done
done

echo "All downloads complete. Files saved under: $LOCAL_BASE"
