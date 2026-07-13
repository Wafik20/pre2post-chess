"""Pipeline: eval JSONLs → per-rollout metrics → per-prompt parquet.

Expects the following directory layout under `--checkpoints`:

  checkpoints/
    sft/<run_tag>/eval_results/step_*/generations/0.jsonl    (step 0 = SFT)
    rl/<run_tag>/eval_results_2k_t1_n16/step_K_*/generations/0.jsonl

The pipeline:
  1. Glob the JSONL paths and read all rollouts (one per line).
  2. Filter to the held-out test sets {test_B1, ..., test_B5}.
  3. Compute tree-based metrics on every rollout.
  4. Collect the unique FENs we need to score and run Stockfish once
     per FEN (parallel across processes). The cache is shared across
     all steps and rows.
  5. Compute Stockfish-based metrics (candidate / opponent / commit
     quality + target presence) for every rollout.
  6. Aggregate rollouts to per-prompt means; add the `commit_quality`
     z-score composite.
  7. Save rollout_df.parquet and prompt_df.parquet.

Run with `--help` for all flags.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import compute_all_metrics_for_row
from quality import (
    collect_fens_for_quality, build_position_cache,
    compute_role_quality, compute_commit_quality_components,
    compute_target_presence,
)


TEST_DATA_SOURCES = {"test_B1", "test_B2", "test_B3", "test_B4", "test_B5"}


# =============================================================================
# Path discovery
# =============================================================================

def find_jsonl_paths(checkpoints_dir: Path, run_tag: str) -> dict[int, Path]:
    """Return {step: jsonl_path}.

    SFT baseline gets step = 0. RL checkpoints keep their training step.
    """
    paths: dict[int, Path] = {}

    sft_glob = checkpoints_dir / "sft" / run_tag / "eval_results" / "step_*" / "generations" / "0.jsonl"
    matches = sorted(glob.glob(str(sft_glob)))
    if matches:
        paths[0] = Path(matches[0])

    rl_pat = checkpoints_dir / "rl" / run_tag / "eval_results_2k_t1_n16" / "step_*_*" / "generations" / "0.jsonl"
    for p in sorted(glob.glob(str(rl_pat))):
        # extract integer step from .../step_<K>_<suffix>/...
        parts = p.split(os.sep)
        step_dir = parts[-3]                # e.g. "step_500_2560len_t1_n16"
        step = int(step_dir.split("_")[1])
        paths[step] = Path(p)

    return paths


# =============================================================================
# JSONL loading
# =============================================================================

def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def filter_test_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("data_source") in TEST_DATA_SOURCES]


# =============================================================================
# Per-rollout metric computation
# =============================================================================

def compute_metrics_for_row(row: dict, step: int, cache: dict) -> dict:
    rec: dict = {
        "step": step,
        "data_source": row.get("data_source"),
        "input": row.get("input"),
        "output": row.get("output"),
        "score": row.get("score"),
        "outcome_score": row.get("outcome_score"),
        "extracted_moves": row.get("extracted_moves"),
        "target_moves": row.get("target_moves"),
    }
    rec.update(compute_all_metrics_for_row(row))           # tree-based
    rec.update(compute_target_presence(row))               # target presence
    rec.update(compute_role_quality(row, cache))           # Stockfish role quality
    rec.update(compute_commit_quality_components(row, cache))  # commit components
    return rec


# =============================================================================
# Per-prompt aggregation + composite z-score
# =============================================================================

def _zscore(s: pd.Series) -> pd.Series:
    mu, sd = s.mean(skipna=True), s.std(skipna=True, ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return s * 0.0
    return (s - mu) / sd


def aggregate_to_prompt(rollout_df: pd.DataFrame) -> pd.DataFrame:
    """Average each numeric metric over the rollouts of the same prompt,
    then add the commit_quality z-score composite.
    """
    numeric_cols = [c for c in rollout_df.columns
                    if pd.api.types.is_numeric_dtype(rollout_df[c]) and c != "step"]
    grouped = rollout_df.groupby(["step", "data_source", "input"], as_index=False)
    prompt_df = grouped[numeric_cols].mean()

    # commit_quality = composite z-score (higher = better) over per-prompt rows
    prompt_df["commit_quality"] = (
        -_zscore(prompt_df["selected_norm_rank_self"])
        - _zscore(prompt_df["selection_rank_gap_self"])
    )
    return prompt_df


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", type=Path, required=True,
                    help="Root directory containing sft/ and rl/ subfolders.")
    ap.add_argument("--run-tag", type=str, required=True,
                    help="e.g. C6p5e18_20m_alpha0.200_beta0.008")
    ap.add_argument("--stockfish", type=str,
                    default=os.environ.get("STOCKFISH", "stockfish"),
                    help="Path to the Stockfish binary.")
    ap.add_argument("--stockfish-depth", type=int, default=8)
    ap.add_argument("--n-workers", type=int,
                    default=int(os.environ.get("N_WORKERS", "8")))
    ap.add_argument("--out", type=Path, required=True,
                    help="Output directory for parquets.")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] discovering JSONLs under {args.checkpoints}")
    paths = find_jsonl_paths(args.checkpoints, args.run_tag)
    if not paths:
        print("  no eval JSONLs found", file=sys.stderr); return 1
    print("  found steps:", sorted(paths.keys()))

    print(f"[2/5] loading + filtering to test_B*")
    rows_by_step: dict[int, list[dict]] = {}
    for step, p in paths.items():
        rows = filter_test_rows(load_jsonl(p))
        rows_by_step[step] = rows
        print(f"  step {step:>4}  rows={len(rows):>6}   {p}")
    all_rows = [r for rs in rows_by_step.values() for r in rs]

    print(f"[3/5] collecting unique FENs for Stockfish")
    fens = collect_fens_for_quality(all_rows)
    print(f"  {len(fens):,} unique FENs")

    print(f"[4/5] Stockfish multipv @ depth={args.stockfish_depth}  "
          f"workers={args.n_workers}")
    cache = build_position_cache(fens, args.stockfish_depth, args.n_workers,
                                  args.stockfish)
    print(f"  cache size: {len(cache):,}")

    print(f"[5/5] computing per-rollout metrics")
    records: list[dict] = []
    for step, rows in rows_by_step.items():
        for r in rows:
            records.append(compute_metrics_for_row(r, step, cache))
        print(f"  step {step:>4}  metrics done ({len(rows)} rollouts)")

    rollout_df = pd.DataFrame.from_records(records)
    prompt_df = aggregate_to_prompt(rollout_df)

    rollout_df.to_parquet(args.out / "rollout_df.parquet")
    prompt_df.to_parquet(args.out / "prompt_df.parquet")
    print(f"\nwrote {args.out/'rollout_df.parquet'}  ({len(rollout_df)} rows)")
    print(f"wrote {args.out/'prompt_df.parquet'}  ({len(prompt_df)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
