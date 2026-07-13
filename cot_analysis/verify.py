"""Compare a freshly-computed prompt_df.parquet against a reference one.

The intended use is: if you have an alternative implementation's output
parquet, this lets you check that the per-prompt metrics agree.

Outputs a short report of (a) per-step mean differences and (b) per-prompt
agreement (correlation + mean absolute diff) on the inner join of (step,
input).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = [
    # tree / behavior / length
    "candidate_count", "num_nodes", "max_depth",
    "effective_branching", "width_depth_index",
    "revisit_rate", "dfs_consistency",
    "reasoning_char_count", "n_ordered_visits",
    # target presence
    "target_in_root_candidates", "target_in_any_depth",
    # Stockfish quality
    "mean_candidate_norm_rank", "best_candidate_norm_rank",
    "mean_opponent_norm_rank",  "best_opponent_norm_rank",
    "candidate_illegal_rate",   "opponent_illegal_rate",
    "selected_norm_rank_self",  "selection_rank_gap_self",
    "commit_quality",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new",       type=Path, required=True,
                    help="prompt_df.parquet produced by this tool.")
    ap.add_argument("--reference", type=Path, required=True,
                    help="prompt_df.parquet to compare against.")
    args = ap.parse_args()

    new = pd.read_parquet(args.new)
    ref = pd.read_parquet(args.reference)
    steps = sorted(set(new.step.unique()) & set(ref.step.unique()))
    print(f"shared steps: {steps}")

    print("\n=== Per-step mean comparison ===")
    print(f"{'metric':32s} {'step':>5s} {'new':>10s} {'ref':>10s} {'diff':>10s}")
    for m in METRICS:
        if m not in new.columns or m not in ref.columns:
            continue
        for s in steps:
            n = new[new.step == s][m].mean()
            r = ref[ref.step == s][m].mean()
            if pd.isna(n) and pd.isna(r):
                continue
            print(f"{m:32s} {s:>5d} {n:>10.4f} {r:>10.4f} {n - r:>+10.5f}")

    print("\n=== Per-prompt agreement (joined on (step, input)) ===")
    print(f"{'metric':32s} {'n':>6s} {'mean|d|':>10s} {'max|d|':>10s} {'corr':>6s}")
    new_s = new[new.step.isin(steps)]
    ref_s = ref[ref.step.isin(steps)]
    key = ["step", "input"]
    for m in METRICS:
        if m not in new.columns or m not in ref.columns:
            continue
        j = (ref_s[key + [m]].rename(columns={m: "ref"})
             .merge(new_s[key + [m]].rename(columns={m: "new"}),
                    on=key, how="inner"))
        j = j.replace([np.inf, -np.inf], np.nan).dropna()
        if len(j) < 10:
            continue
        d = (j["new"] - j["ref"]).abs()
        print(f"{m:32s} {len(j):>6d} {d.mean():>10.5f} {d.max():>10.5f} "
              f"{j['new'].corr(j['ref']):>6.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
