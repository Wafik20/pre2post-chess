#!/bin/bash
#
# Batch convert VERL FSDP checkpoints to HuggingFace format
#
# Traverses all model subdirectories under LARGE_DIR and converts
# every checkpoint found to HuggingFace format using verl.model_merger.
#
# POST-CONVERSION CLEANUP:
#   After successful HF conversion, the original FSDP checkpoint directory
#   (global_step_N/) is deleted for all steps EXCEPT the latest (highest step).
#   The latest checkpoint is always kept intact.
#
# SAFETY: Deletion only happens after verified successful HF conversion.
#         The latest checkpoint is NEVER deleted.
#         If conversion fails, the original FSDP checkpoint is kept untouched.
#

set -u  # catch undefined variables; do NOT use set -e (we handle errors per-step)

# =============================================================================
# USER CONFIGURATION
# =============================================================================
# Point this at the directory that holds your RL run folders, i.e. the
# SAVE_ROOT/<hparam_tag> directory produced by sweep_multi_turn.sh. Each
# direct child is expected to contain a checkpoints/ subdirectory.

LARGE_DIR="${LARGE_DIR:-../results/rl/trajectory_sep_no_labels/multi_turn_lr1e-5_bs256_kl0.001_res3584}"

# =============================================================================
# TRAVERSAL + CONVERSION
# =============================================================================

echo "=========================================="
echo "Batch HF Conversion for all models under:"
echo "  $LARGE_DIR"
echo "=========================================="
echo ""

if [ ! -d "$LARGE_DIR" ]; then
    echo "ERROR: Large directory does not exist: $LARGE_DIR"
    exit 1
fi

# Collect all model subdirectories (non-recursive, direct children)
model_dirs=()
for d in "$LARGE_DIR"/*/; do
    [ -d "$d" ] && model_dirs+=("$d")
done

if [ ${#model_dirs[@]} -eq 0 ]; then
    echo "ERROR: No subdirectories found under $LARGE_DIR"
    exit 1
fi

echo "Found ${#model_dirs[@]} model folder(s):"
for d in "${model_dirs[@]}"; do
    echo "  - $(basename $d)"
done
echo ""

total_converted=0
total_skipped=0
total_failed=0
total_deleted=0

for model_dir in "${model_dirs[@]}"; do
    model_name=$(basename "$model_dir")
    checkpoint_base_dir="${model_dir}checkpoints"
    hf_output_base_dir="${model_dir}checkpoints_hf_format"

    echo "------------------------------------------"
    echo "Model: $model_name"
    echo "------------------------------------------"

    if [ ! -d "$checkpoint_base_dir" ]; then
        echo "  WARNING: No checkpoints/ directory found, skipping."
        echo ""
        continue
    fi

    # Find all available global steps, sorted numerically
    available_steps=$(ls -d "$checkpoint_base_dir"/global_step_* 2>/dev/null \
        | xargs -n 1 basename | sed 's/global_step_//' | sort -n)

    if [ -z "$available_steps" ]; then
        echo "  WARNING: No global_step_* checkpoints found in $checkpoint_base_dir, skipping."
        echo ""
        continue
    fi

    # Identify the latest (highest) step — never delete this one
    latest_step=$(echo "$available_steps" | tail -1)

    echo "  Checkpoints found: $(echo $available_steps | tr '\n' ' ')"
    echo "  Latest step:       $latest_step  (FSDP copy will be kept)"
    echo "  Output dir:        $hf_output_base_dir"
    echo ""

    mkdir -p "$hf_output_base_dir"

    for step in $available_steps; do
        source_step_dir="$checkpoint_base_dir/global_step_${step}"
        source_checkpoint_dir="$source_step_dir/actor"
        target_hf_dir="$hf_output_base_dir/global_step_${step}"

        is_latest=false
        [ "$step" = "$latest_step" ] && is_latest=true

        if [ ! -d "$source_checkpoint_dir" ]; then
            echo "  WARNING: actor/ dir missing for step $step: $source_checkpoint_dir"
            ((total_failed++)) || true
            continue
        fi

        # Check if already converted
        already_converted=false
        if [ -f "$target_hf_dir/model.safetensors" ] || \
           [ -f "$target_hf_dir/pytorch_model.bin" ] || \
           [ -f "$target_hf_dir/model.safetensors.index.json" ] || \
           ls "$target_hf_dir"/model-*-of-*.safetensors >/dev/null 2>&1 || \
           ls "$target_hf_dir"/pytorch_model-*-of-*.bin >/dev/null 2>&1; then
            already_converted=true
        fi

        if $already_converted; then
            echo "  [SKIP] step $step: already converted"
            ((total_skipped++)) || true
        else
            echo "  [CONVERT] step $step..."
            echo "    Source: $source_checkpoint_dir"
            echo "    Target: $target_hf_dir"

            mkdir -p "$target_hf_dir"

            log_file="/tmp/convert_${model_name}_step_${step}.log"

            # Run conversion; capture exit code explicitly (pipefail not set, so tee won't mask it)
            set -o pipefail
            python -m verl.model_merger merge \
                --backend fsdp \
                --local_dir "$source_checkpoint_dir" \
                --target_dir "$target_hf_dir" \
                2>&1 | tee "$log_file"
            convert_exit=$?
            set +o pipefail

            # Verify conversion by checking for model weight files
            hf_weights_found=false
            if [ -f "$target_hf_dir/model.safetensors" ] || \
               [ -f "$target_hf_dir/pytorch_model.bin" ] || \
               [ -f "$target_hf_dir/model.safetensors.index.json" ] || \
               ls "$target_hf_dir"/model-*-of-*.safetensors >/dev/null 2>&1 || \
               ls "$target_hf_dir"/pytorch_model-*-of-*.bin >/dev/null 2>&1; then
                hf_weights_found=true
            fi

            if [ $convert_exit -eq 0 ] && $hf_weights_found; then
                echo "  [OK]   step $step: conversion successful -> $target_hf_dir"
                ((total_converted++)) || true
                already_converted=true
            else
                echo "  [FAIL] step $step: conversion failed (exit=$convert_exit, weights_found=$hf_weights_found)"
                echo "         Original FSDP checkpoint kept: $source_step_dir"
                echo "         Log: $log_file"
                ((total_failed++)) || true
                # Clean up empty/partial HF output dir so re-runs start fresh
                rm -rf "$target_hf_dir"
                # Do NOT touch the original FSDP checkpoint
                continue
            fi
            echo ""
        fi

        # Cleanup: delete the FSDP checkpoint if converted and NOT the latest step
        if $already_converted && ! $is_latest; then
            if [ -d "$source_step_dir" ]; then
                echo "  [DELETE] Removing FSDP checkpoint (not latest): $source_step_dir"
                rm -rf "$source_step_dir"
                ((total_deleted++)) || true
            fi
        elif $already_converted && $is_latest; then
            echo "  [KEEP]   Keeping FSDP checkpoint (latest step $step)"
        fi
        echo ""
    done
done

echo "=========================================="
echo "Batch Conversion Complete"
echo "=========================================="
echo "  Converted: $total_converted"
echo "  Skipped:   $total_skipped  (already done)"
echo "  Failed:    $total_failed"
echo "  Deleted:   $total_deleted  FSDP checkpoints removed (non-latest only)"
echo ""
