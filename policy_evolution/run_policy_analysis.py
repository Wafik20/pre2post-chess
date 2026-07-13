#!/usr/bin/env python3
"""
run_policy_analysis.py — end-to-end policy-evolution analysis for ONE model.

Inputs (already on disk, produced by launch_measure_policy.py + compute_metrics.py):
  results/<cfg>/{pretrain,sft,rl}/step_*/raw_scores.parquet

Outputs (under results/<cfg>/policy_analysis/):
  01_sharpening_kl.png       α* (KL fit) and residual across stages
  02_trace_sensitivity.png    TS_sft (anchor) and TS_rl across rl_step
  03_rl_sft_trace_collapse.png    D_marg, D_trace, C_trace across rl_step
  04_entropy_pretrain_vs_sft.png  pretrain H vs SFT per-trace H (scatter + ECDF)
  05_category_summary.png     count/step + count/step/bin + pass@k/cat
  per_step_summary.csv        one row per rl_step
  per_state_categories.parquet    one row per (puzzle, rl_step)
  cache/state_<stage>_<step>.pkl  per-puzzle states (re-used across runs)
  categories/<cat>/examples/step_<n>_puzzle_<id>.png  + manifest.csv

Usage:
  python run_policy_analysis.py \\
      --total-compute 6p5e18 --modelsize 50m --alpha 0.200 --beta 0.023 \\
      [--rl-steps 100 400 600 800] \\
      [--n-examples 5] [--alpha-max 5.0] \\
      [--delta-M 1.0 --eps-tail 0.05 --eps-high 0.5 --K-tail 5] \\
      [--skip-categories]   # skip per-state example renders (fast mode)
"""

from __future__ import annotations
import argparse, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
import policy_analysis_lib as L


# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-compute", required=True, help="e.g. 6p5e18")
    ap.add_argument("--modelsize",     required=True, help="e.g. 50m")
    ap.add_argument("--alpha",         required=True, help="pretrain α, e.g. 0.200")
    ap.add_argument("--beta",          required=True, help="SFT β, e.g. 0.023")
    ap.add_argument("--rl-steps", nargs="*", type=int, default=None,
                     help="explicit RL steps; default = all discovered")
    ap.add_argument("--out-dir", default=None, help="override output dir")
    ap.add_argument("--results-root", default=None,
                     help="Root of the raw_scores tree (overrides the RESULTS_ROOT "
                          "env var and the ./results default).")
    ap.add_argument("--n-examples", type=int, default=5,
                     help="number of representative states per category per RL step")
    ap.add_argument("--alpha-max", type=float, default=5.0)
    ap.add_argument("--eps-tail",  type=float, default=0.05,
                     help="Tail threshold ε_tail for splitting tail_discovery vs topk_correction")
    ap.add_argument("--results-tag", default="",
                     help="Append to cfg_id for the input results directory; matches "
                          "launch_measure_policy.py's --results-tag (e.g. '_n128').")
    ap.add_argument("--example-bracket", default=None,
                     help="Filter example puzzle selection to one bracket "
                          "(e.g. 'test_B5').  Default: pick from all brackets.")
    ap.add_argument("--top-k",     type=int,   default=1,
                     help="k in T_θ^k (top-K cut-off).  k=1 = strict spec; k>1 relaxes "
                          "tail_discovery/topk_correction to count states where GT only "
                          "reached the top-K (not necessarily top-1).")
    ap.add_argument("--skip-categories", action="store_true",
                     help="skip per-state example PNG renders (faster)")
    return ap.parse_args()


CATEGORIES = ["gt_amplification", "topk_correction", "tail_discovery",
              "wrong_mode_amplification", "gt_regression", "other"]
# Coordinated palette — six maximally distinct hues, "other" muted
CAT_COLORS = {
    "gt_amplification":         "#1a9850",   # green  — RL sharpens correct mode
    "topk_correction":          "#3690c0",   # blue   — RL promotes a non-tail GT to top-1
    "tail_discovery":           "#fdae61",   # orange — RL promotes a tail GT to top-1
    "wrong_mode_amplification": "#d7191c",   # red    — RL sharpens same wrong mode
    "gt_regression":            "#54278f",   # purple — RL drops GT out of top-1
    "other":                    "#999999",   # gray   — stable / wrong-mode switches / no-Δ
}

def cat_label(cat: str, top_k: int = 1, multiline: bool = False) -> str:
    """Display label for a category (paper-friendly title-case).

    multiline=True splits long labels across two lines (for narrow subplot titles).
    """
    if multiline:
        return {
            "gt_amplification":         "Ground Truth\nAmplification",
            "topk_correction":          f"Top-{top_k} Correction",
            "tail_discovery":           "Tail Discovery",
            "wrong_mode_amplification": "Wrong Mode\nAmplification",
            "gt_regression":            "Ground Truth\nRegression",
            "other":                    "Other",
        }.get(cat, cat.replace("_", " ").title())
    return {
        "gt_amplification":         "Ground Truth Amplification",
        "topk_correction":          f"Top-{top_k} Correction",
        "tail_discovery":           "Tail Discovery",
        "wrong_mode_amplification": "Wrong Mode Amplification",
        "gt_regression":            "Ground Truth Regression",
        "other":                    "Other",
    }.get(cat, cat.replace("_", " ").title())

# Global plot style — same recipe as scaling_law_fit/tools/plot_AB_relations.py
PLOT_STYLE = {
    "font.family":     "serif",
    "font.serif":      ["DejaVu Serif", "Liberation Serif", "Times", "serif"],
    "font.size":        14,
    "axes.titlesize":   15,
    "axes.labelsize":   14,
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
    "legend.fontsize":  11,
    "axes.linewidth":   1.0,
    "grid.linewidth":   0.6,
    "lines.linewidth":  1.8,
    "savefig.dpi":      170,
    "axes.grid":        True,
    "grid.linestyle":  "--",
    "grid.alpha":       0.4,
}


def _save_both(fig_or_none, base_path: Path) -> None:
    """Save current figure as both .png and .pdf using bbox_inches='tight'."""
    base_path = Path(base_path)
    if base_path.suffix.lower() in (".png", ".pdf"):
        base_path = base_path.with_suffix("")
    plt.savefig(base_path.with_suffix(".png"), bbox_inches="tight")
    plt.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
# Common marker/line kwargs for trajectory lines
LINE_KW   = dict(marker="o", ms=7, markeredgecolor="black", markeredgewidth=0.7, lw=1.8)
DASH_KW   = dict(marker="s", ms=6, markeredgecolor="black", markeredgewidth=0.6, lw=1.5, ls="--", alpha=0.85)
LEG_KW    = dict(frameon=True, framealpha=0.92, edgecolor="#cccccc")


def main():
    args = parse_args()
    if args.results_root:
        L.RESULTS_ROOT = Path(args.results_root)
    cfg = L.config_id(args.total_compute, args.modelsize, args.alpha, args.beta) + (args.results_tag or "")
    cfg_dir = L.RESULTS_ROOT / cfg
    if not cfg_dir.exists():
        sys.exit(f"[ERR] no results dir: {cfg_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else cfg_dir / "policy_analysis"
    cache_dir = out_dir / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")
    plt.rcParams.update(PLOT_STYLE)
    plt.rcParams["figure.dpi"] = 110

    print(f"[run_policy_analysis] cfg={cfg}")
    print(f"[run_policy_analysis] out_dir={out_dir}")

    # ── Load per-puzzle states (cached) ──────────────────────────────────
    t0 = time.time()
    pre = L.build_or_load_state("pretrain", "step_final", cfg, cache_dir)
    sft = L.build_or_load_state("sft",      "step_final", cfg, cache_dir)
    if pre is None or sft is None:
        sys.exit(f"[ERR] missing pretrain or SFT raw_scores for {cfg}")
    print(f"[load] pretrain {len(pre)} puzzles, sft {len(sft)} puzzles  ({time.time()-t0:.1f}s)")

    rl_steps = args.rl_steps if args.rl_steps is not None else L.discover_rl_steps(cfg)
    rl_states = {}
    for s in rl_steps:
        t0 = time.time()
        st = L.build_or_load_state("rl", f"step_{s}", cfg, cache_dir)
        if st is None:
            print(f"[skip] rl step_{s} (no raw_scores)")
            continue
        rl_states[s] = st
        print(f"[load] rl step_{s} {len(st)} puzzles  ({time.time()-t0:.1f}s)")
    if not rl_states:
        print("[warn] no RL steps available — only pretrain/SFT plots will be produced")

    # ── Restrict to matched B1-B5 state set ──────────────────────────────
    # RL rollouts only cover test_B1..test_B5; SFT additionally covers test_B0.
    # Drop B0 from pretrain/SFT so all stages compare over the same coverage.
    matched_bins = {"test_B1", "test_B2", "test_B3", "test_B4", "test_B5"}
    pre = pre[pre["data_source"].isin(matched_bins)].reset_index(drop=True)
    sft = sft[sft["data_source"].isin(matched_bins)].reset_index(drop=True)
    for _s in list(rl_states.keys()):
        rl_states[_s] = rl_states[_s][rl_states[_s]["data_source"].isin(matched_bins)].reset_index(drop=True)
    print(f"[filter] restricted to B1-B5: pretrain {len(pre)}, sft {len(sft)} states; "
          f"rl per-step ~{len(next(iter(rl_states.values()))) if rl_states else 0}")

    # ── Sharpening fits — global + per-state, KL + logit-linear ──────────
    print("\n[sharpening] global + per-state, KL projection + logit-linear")

    def _summarize(joined, source_col, target_col, label):
        kl_g  = L.fit_alpha_kl(joined, source_col=source_col, target_col=target_col,
                                 alpha_max=args.alpha_max)
        kl_p  = L.fit_alpha_kl_per_state(joined, source_col=source_col, target_col=target_col,
                                            alpha_max=args.alpha_max)
        log_g = L.fit_logit_linear_global(joined, source_col=source_col, target_col=target_col)
        log_p = L.fit_logit_linear_per_state(joined, source_col=source_col, target_col=target_col)
        row = {
            # KL global
            "kl_alpha_global":     kl_g["alpha_star"],
            "kl_residual_global":  kl_g["residual_jsd"],
            "kl_baseline_global":  kl_g["baseline_jsd"],
            "kl_explained_global": kl_g["explained_sharp"],
            "kl_boundary":         kl_g["boundary"],
            # KL per-state (median + IQR)
            "kl_alpha_med":        float(kl_p["alpha_star"].median()),
            "kl_alpha_q25":        float(kl_p["alpha_star"].quantile(0.25)),
            "kl_alpha_q75":        float(kl_p["alpha_star"].quantile(0.75)),
            "kl_residual_med":     float(kl_p["residual_jsd"].median()),
            "kl_residual_q25":     float(kl_p["residual_jsd"].quantile(0.25)),
            "kl_residual_q75":     float(kl_p["residual_jsd"].quantile(0.75)),
            "kl_baseline_med":     float(kl_p["baseline_jsd"].median()),
            "kl_explained_med":    float(kl_p["explained_sharp"].median()),
            "kl_explained_q25":    float(kl_p["explained_sharp"].quantile(0.25)),
            "kl_explained_q75":    float(kl_p["explained_sharp"].quantile(0.75)),
            # Logit global
            "logit_slope_global":  log_g["slope"],
            "logit_inter_global":  log_g["intercept"],
            "logit_r2_global":     log_g["r2"],
            "logit_n_points":      log_g["n_points"],
            # Logit per-state (median + IQR)
            "logit_slope_med":     float(log_p["slope"].median()),
            "logit_slope_q25":     float(log_p["slope"].quantile(0.25)),
            "logit_slope_q75":     float(log_p["slope"].quantile(0.75)),
            "logit_r2_med":        float(log_p["r2"].median()),
            "logit_r2_q25":        float(log_p["r2"].quantile(0.25)),
            "logit_r2_q75":        float(log_p["r2"].quantile(0.75)),
        }
        print(f"  {label}: KL α*={row['kl_alpha_global']:.3f} (med={row['kl_alpha_med']:.3f}, IQR=[{row['kl_alpha_q25']:.3f},{row['kl_alpha_q75']:.3f}])  "
              f"explained={row['kl_explained_global']:.3f} (med={row['kl_explained_med']:.3f})  "
              f"logit a={row['logit_slope_global']:.3f} (med={row['logit_slope_med']:.3f})  "
              f"R²={row['logit_r2_global']:.3f} (med={row['logit_r2_med']:.3f})")
        return row

    sharp_rows = []
    per_state_kl  = {}    # rl_step (or 0 for pre→sft) → DataFrame
    per_state_log = {}

    def _store_per_state(joined, source_col, target_col, key):
        per_state_kl[key]  = L.fit_alpha_kl_per_state(joined, source_col=source_col,
                                                         target_col=target_col,
                                                         alpha_max=args.alpha_max)
        per_state_log[key] = L.fit_logit_linear_per_state(joined, source_col=source_col,
                                                             target_col=target_col)

    j_pre_sft = L._join_states(sft, pre).rename(columns={"p_sft": "p_pre", "p_rl": "p_sft"})
    sharp_rows.append({"stage": "pretrain→sft", "rl_step": 0,
                        **_summarize(j_pre_sft, "p_pre", "p_sft", "pretrain→sft")})
    _store_per_state(j_pre_sft, "p_pre", "p_sft", 0)
    for s, rl in rl_states.items():
        j = L._join_states(rl, sft)
        sharp_rows.append({"stage": f"sft→rl_{s}", "rl_step": s,
                            **_summarize(j, "p_sft", "p_rl", f"sft→rl_{s}")})
        _store_per_state(j, "p_sft", "p_rl", s)
    sharp_df = pd.DataFrame(sharp_rows)
    sharp_df.to_csv(out_dir / "sharpening_fits.csv", index=False)

    # ── Trace sensitivity ────────────────────────────────────────────────
    print("\n[TS] computing trace sensitivity (KL form)")
    TS_pre = L.aggregate_TS(pre)["mean"]
    TS_sft = L.aggregate_TS(sft)["mean"]
    print(f"  TS_pretrain (≈0)  = {TS_pre:.5f}")
    print(f"  TS_sft            = {TS_sft:.5f}")
    TS_rl = {s: L.aggregate_TS(st)["mean"] for s, st in rl_states.items()}
    for s, v in TS_rl.items():
        print(f"  TS_rl  step_{s:<5} = {v:.5f}")

    # ── Trace collapse ───────────────────────────────────────────────────
    print("\n[collapse] computing D_marg, D_trace, C_trace per RL step")
    collapse_per_step = {}
    for s, rl in rl_states.items():
        cdf = L.aggregate_collapse(rl, sft)
        collapse_per_step[s] = cdf
        print(f"  step_{s:<5}: D_marg={cdf['d_marg'].mean():.4f}  D_trace={cdf['d_trace'].mean():.4f}  "
              f"C_trace={cdf['c_trace'].mean():.4f}  frac_nearest_correct={cdf['nearest_trace_modal_eq_g'].mean():.3f}")

    # ── Categorize per (puzzle, rl_step) ─────────────────────────────────
    print("\n[categorize] assigning categories at every RL step")
    cat_rows = []
    for s, rl in rl_states.items():
        cdf = L.categorize_all(sft, rl, eps_tail=args.eps_tail, top_k=args.top_k)
        cdf["rl_step"] = s
        cat_rows.append(cdf)
    if cat_rows:
        cat_all = pd.concat(cat_rows, ignore_index=True)
        cat_all.to_parquet(out_dir / "per_state_categories.parquet")
        # Counts per (rl_step, category)
        cat_counts = cat_all.groupby(["rl_step", "category"]).size().unstack(fill_value=0)
        for c in CATEGORIES:
            if c not in cat_counts.columns: cat_counts[c] = 0
        cat_counts = cat_counts[CATEGORIES]
        print("  counts per RL step:"); print(cat_counts)
    else:
        cat_all = pd.DataFrame()
        cat_counts = pd.DataFrame()

    # ── Per-step summary CSV ─────────────────────────────────────────────
    print("\n[summary] building per-step summary CSV")
    summary_rows = []
    for s, rl in rl_states.items():
        cdf = collapse_per_step[s]
        sub = cat_all[cat_all["rl_step"] == s] if len(cat_all) else pd.DataFrame()
        passk_overall = L.pass_at_k_for_states(rl.assign(n_correct_rl=rl["n_correct"], n_traces_rl=rl["n_traces"]),
                                                k_list=(1, 8, 16)) if len(rl) else {}
        row = dict(rl_step=s, n_states=len(rl),
                   alpha_star=sharp_df.loc[sharp_df["rl_step"]==s, "kl_alpha_global"].iloc[0]
                                if (sharp_df["rl_step"]==s).any() else np.nan,
                   residual_jsd=sharp_df.loc[sharp_df["rl_step"]==s, "kl_residual_global"].iloc[0]
                                if (sharp_df["rl_step"]==s).any() else np.nan,
                   explained_sharp=sharp_df.loc[sharp_df["rl_step"]==s, "kl_explained_global"].iloc[0]
                                if (sharp_df["rl_step"]==s).any() else np.nan,
                   TS_rl=TS_rl[s], TS_sft_anchor=TS_sft, TS_pretrain_anchor=TS_pre,
                   d_marg=cdf["d_marg"].mean(), d_trace=cdf["d_trace"].mean(),
                   c_trace=cdf["c_trace"].mean(),
                   frac_collapse_correct=cdf["nearest_trace_modal_eq_g"].mean(),
                   **{f"count_{c}": int((sub["category"]==c).sum()) for c in CATEGORIES},
                   **passk_overall)
        # Pass@k per category
        for c in CATEGORIES:
            sub_c = sub[sub["category"]==c]
            pk = L.pass_at_k_for_states(sub_c, k_list=(1, 8, 16)) if len(sub_c) else {"pass@1": np.nan, "pass@8": np.nan, "pass@16": np.nan}
            for k, v in pk.items():
                row[f"{c}__{k}"] = v
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows).sort_values("rl_step")
    summary_df.to_csv(out_dir / "per_step_summary.csv", index=False)

    # ── Plot 01: α / a trajectories (KL + logit-linear, global + per-state) ─
    print("\n[plot] 01_sharpening_alpha.png")
    rl_only = sharp_df[sharp_df["stage"].str.startswith("sft→rl")].sort_values("rl_step")
    pre_sft = sharp_df[sharp_df["stage"]=="pretrain→sft"].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0))
    # Left: KL α*
    ax = axes[0]
    ax.plot(rl_only["rl_step"], rl_only["kl_alpha_global"], color="#4c72b0",
             label=r"$\alpha^\star$  (global KL)", **LINE_KW)
    ax.plot(rl_only["rl_step"], rl_only["kl_alpha_med"], color="#4c72b0",
             ls="--", marker="s", ms=6, mec="black", mew=0.6, lw=1.5, alpha=0.85,
             label=r"$\alpha^\star$  (per-state median)")
    ax.fill_between(rl_only["rl_step"], rl_only["kl_alpha_q25"], rl_only["kl_alpha_q75"],
                     color="#4c72b0", alpha=0.18, label="per-state IQR")
    ax.axhline(1, color="#666", ls=":", lw=1, label=r"$\alpha=1$  (no sharpening)")
    ax.axhline(pre_sft["kl_alpha_global"], color="#dd8452", ls="--", lw=1.4,
                 label=f"pretrain → SFT   $\\alpha^\\star$={pre_sft['kl_alpha_global']:.2f}")
    ax.set(xlabel="RL step", ylabel=r"$\alpha^\star$  (KL-optimal)")
    ax.set_title("Probability-space KL fit", pad=8)
    ax.legend(loc="best", **LEG_KW)
    # Right: Centered log-prob slope β
    ax = axes[1]
    ax.plot(rl_only["rl_step"], rl_only["logit_slope_global"], color="#55a868",
             label=r"$\beta$  (global)", **LINE_KW)
    ax.plot(rl_only["rl_step"], rl_only["logit_slope_med"], color="#55a868",
             ls="--", marker="s", ms=6, mec="black", mew=0.6, lw=1.5, alpha=0.85,
             label=r"$\beta$  (per-state median)")
    ax.fill_between(rl_only["rl_step"], rl_only["logit_slope_q25"], rl_only["logit_slope_q75"],
                     color="#55a868", alpha=0.18, label="per-state IQR")
    ax.axhline(1, color="#666", ls=":", lw=1, label=r"$\beta=1$  (identity)")
    ax.axhline(pre_sft["logit_slope_global"], color="#dd8452", ls="--", lw=1.4,
                 label=f"pretrain → SFT   $\\beta$={pre_sft['logit_slope_global']:.2f}")
    ax.set(xlabel="RL step", ylabel=r"slope  $\beta$")
    ax.set_title(r"Centered log-prob fit:  "
                  r"$\widetilde{\log\pi}_{rl} = \beta\,\widetilde{\log\pi}_{sft}$"
                  "\n($\\widetilde{\\log\\pi}(a) = \\log\\pi(a) - \\overline{\\log\\pi}$)",
                  pad=6)
    ax.legend(loc="best", **LEG_KW)
    fig.suptitle(cfg, fontweight="bold", y=1.01)
    plt.tight_layout(); plt.savefig(out_dir / "01_sharpening_alpha.png", bbox_inches="tight")
    plt.close()

    # ── Plot 01b: goodness-of-fit (KL bars + explained, logit R²) ─────────
    print("[plot] 01b_sharpening_metrics.png")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0))
    steps_arr = rl_only["rl_step"].values
    n = len(steps_arr); x = np.arange(n); w = 0.36

    # Left: KL — overlayed bars (residual / baseline) + line for explained_sharp on twin axis
    ax = axes[0]
    bars_res = ax.bar(x - w/2, rl_only["kl_residual_global"], w,
                        color="#c44e52", edgecolor="black", lw=0.7,
                        label="residual JSD (global)")
    bars_base = ax.bar(x + w/2, rl_only["kl_baseline_global"], w,
                        color="#7f7f7f", edgecolor="black", lw=0.7, alpha=0.85,
                        label="baseline JSD (global)")
    # Per-state median residual / baseline drawn as small markers at the top of each bar
    ax.scatter(x - w/2, rl_only["kl_residual_med"], marker="_", s=80,
                c="black", zorder=5, label="residual JSD (per-state median)")
    ax.scatter(x + w/2, rl_only["kl_baseline_med"], marker="_", s=80,
                c="black", zorder=5)
    ax.set_xticks(x); ax.set_xticklabels([str(s) for s in steps_arr])
    ax.set_xlabel("RL step")
    ax.set_ylabel("JSD")
    ax.set_title("KL fit: residual / baseline / explained", pad=8)
    ax2 = ax.twinx()
    ax2.plot(x, rl_only["kl_explained_global"], color="#1a9850",
              marker="o", ms=8, mec="black", mew=0.7, lw=2.0,
              label="explained_sharp (global)")
    ax2.plot(x, rl_only["kl_explained_med"], color="#1a9850",
              ls="--", marker="s", ms=6, mec="black", mew=0.6, lw=1.5, alpha=0.85,
              label="explained_sharp (per-state median)")
    ax2.fill_between(x, rl_only["kl_explained_q25"], rl_only["kl_explained_q75"],
                       color="#1a9850", alpha=0.18, label="per-state IQR")
    ax2.set_ylabel("explained_sharp", color="#1a9850")
    ax2.tick_params(axis="y", labelcolor="#1a9850")
    ax2.set_ylim(-0.05, 1.0)
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8.5, **LEG_KW)

    # Right: Centered log-prob R² (global + per-state median + IQR)
    ax = axes[1]
    ax.plot(rl_only["rl_step"], rl_only["logit_r2_global"], color="#dd8452",
             label=r"$R^2$  (global)", **LINE_KW)
    ax.plot(rl_only["rl_step"], rl_only["logit_r2_med"], color="#dd8452",
             ls="--", marker="s", ms=6, mec="black", mew=0.6, lw=1.5, alpha=0.85,
             label=r"$R^2$  (per-state median)")
    ax.fill_between(rl_only["rl_step"], rl_only["logit_r2_q25"], rl_only["logit_r2_q75"],
                     color="#dd8452", alpha=0.18, label="per-state IQR")
    ax.set(xlabel="RL step", ylabel=r"$R^2$")
    ax.set_title("Centered log-prob fit: $R^2$", pad=8)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="best", **LEG_KW)
    fig.suptitle(cfg, fontweight="bold", y=1.01)
    plt.tight_layout(); plt.savefig(out_dir / "01b_sharpening_metrics.png", bbox_inches="tight")
    plt.close()

    # ── Plot 01c: per-state distributions (violins across RL steps) ───────
    print("[plot] 01c_sharpening_distributions.png")
    rl_step_keys = sorted([k for k in per_state_kl if k != 0])
    if rl_step_keys:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        positions = list(range(len(rl_step_keys)))
        labels = [str(s) for s in rl_step_keys]

        def _violin(ax, data_lists, color, ylabel, title, ref=None):
            parts = ax.violinplot(data_lists, positions=positions, showmeans=False,
                                    showmedians=False, showextrema=False, widths=0.7)
            for body in parts["bodies"]:
                body.set_facecolor(color); body.set_edgecolor("black")
                body.set_alpha(0.55); body.set_linewidth(0.7)
            # Median + IQR + whiskers (5–95th)
            for i, d in enumerate(data_lists):
                if len(d) == 0: continue
                med  = float(np.median(d))
                q25  = float(np.quantile(d, 0.25))
                q75  = float(np.quantile(d, 0.75))
                p05  = float(np.quantile(d, 0.05))
                p95  = float(np.quantile(d, 0.95))
                ax.plot([positions[i]]*2, [p05, p95], color="black", lw=1.0, zorder=3)
                ax.plot([positions[i]]*2, [q25, q75], color="black", lw=3.5, zorder=4)
                ax.plot(positions[i], med, "o", color="white",
                          markeredgecolor="black", markeredgewidth=0.8, ms=6, zorder=5)
            ax.set_xticks(positions); ax.set_xticklabels(labels)
            ax.set_xlabel("RL step")
            ax.set_ylabel(ylabel)
            ax.set_title(title, pad=8)
            if ref is not None:
                ax.axhline(ref, color="#666", ls=":", lw=1)

        # Per-state KL α*
        kl_alpha_per_step = [per_state_kl[s]["alpha_star"].dropna().values for s in rl_step_keys]
        _violin(axes[0,0], kl_alpha_per_step, "#4c72b0",
                  ylabel=r"$\alpha^\star$  (per-state, KL fit)",
                  title=r"Per-state $\alpha^\star$  (KL)", ref=1.0)
        # Per-state KL explained_sharp
        kl_expl_per_step = [per_state_kl[s]["explained_sharp"].dropna().values for s in rl_step_keys]
        _violin(axes[0,1], kl_expl_per_step, "#1a9850",
                  ylabel="explained_sharp  (per-state)",
                  title="Per-state explained_sharp  (KL)", ref=0.0)
        axes[0,1].set_ylim(-0.5, 1.05)
        # Per-state centered-log slope β
        log_slope_per_step = [per_state_log[s]["slope"].dropna().values for s in rl_step_keys]
        _violin(axes[1,0], log_slope_per_step, "#55a868",
                  ylabel=r"slope  $\beta$  (per-state, centered log-prob)",
                  title=r"Per-state $\beta$  (centered log-prob)", ref=1.0)
        axes[1,0].set_ylim(-1.0, 5.0)
        # Per-state centered-log R²
        log_r2_per_step = [per_state_log[s]["r2"].dropna().values for s in rl_step_keys]
        _violin(axes[1,1], log_r2_per_step, "#dd8452",
                  ylabel=r"$R^2$  (per-state, centered log-prob)",
                  title=r"Per-state $R^2$  (centered log-prob)", ref=None)
        axes[1,1].set_ylim(0, 1.02)

        # Annotate panel-wide median + IQR text
        for ax, lists in zip(axes.flat,
                              [kl_alpha_per_step, kl_expl_per_step,
                               log_slope_per_step, log_r2_per_step]):
            terminal = lists[-1]
            if len(terminal):
                m = float(np.median(terminal))
                lo = float(np.quantile(terminal, 0.25))
                hi = float(np.quantile(terminal, 0.75))
                ax.text(0.99, 0.04,
                          f"terminal step (rl={rl_step_keys[-1]}):\n"
                          f"median = {m:.3f},  IQR = [{lo:.3f}, {hi:.3f}]",
                          transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                          bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                                      alpha=0.92, edgecolor="#cccccc"))

        fig.suptitle(f"Per-state distributions across RL steps   ({cfg})",
                       fontweight="bold", y=1.00)
        plt.tight_layout()
        plt.savefig(out_dir / "01c_sharpening_distributions.png", bbox_inches="tight")
        plt.close()

    # ── Plot 02: Trace sensitivity (with SFT @ rl_step=0 anchor) ─────────
    print("[plot] 02_trace_sensitivity.png")
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    if rl_states:
        steps_full = [0] + sorted(rl_states.keys())
        ts_full    = [TS_sft] + [TS_rl[s] for s in sorted(rl_states.keys())]
        ax.plot(steps_full, ts_full, color="#c44e52",
                  label="trajectory (SFT @ step 0 → RL)", **LINE_KW)
        ax.scatter([0], [TS_sft], color="#4c72b0", s=120, zorder=5,
                    edgecolor="black", linewidths=0.7,
                    label=f"SFT baseline   TS={TS_sft:.4f}")
    else:
        ax.scatter([0], [TS_sft], color="#4c72b0", s=120, edgecolor="black",
                    label=f"SFT  TS={TS_sft:.4f}")
    ax.axhline(TS_pre, color="#666", ls=":", lw=1, label=f"pretrain (≈0)   TS={TS_pre:.5f}")
    ax.set(xlabel="RL step  (0 = SFT)",
            ylabel=r"Trace sensitivity   $\mathbb{E}_{s_0}\,[(1/N)\sum_i \mathrm{KL}(\pi_i\,\|\,\bar\pi)]$")
    ax.set_title("Trace sensitivity", pad=8)
    ax.legend(loc="best", **LEG_KW)
    fig.suptitle(cfg, fontweight="bold", y=1.02)
    plt.tight_layout(); plt.savefig(out_dir / "02_trace_sensitivity.png", bbox_inches="tight")
    plt.close()

    # ── Plot 03: Trace collapse (with SFT @ rl_step=0 anchor) ────────────
    print("[plot] 03_rl_sft_trace_collapse.png")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    if rl_states:
        sft_self = L.intra_sft_collapse(sft)
        sorted_steps = sorted(rl_states.keys())
        steps   = [0] + sorted_steps
        D_marg  = [0.0]                        + [collapse_per_step[s]["d_marg"].mean()  for s in sorted_steps]
        D_trace = [sft_self["d_trace"].mean()] + [collapse_per_step[s]["d_trace"].mean() for s in sorted_steps]
        C_trace = [-sft_self["d_trace"].mean()] + [collapse_per_step[s]["c_trace"].mean() for s in sorted_steps]
        frac_correct = [sft_self["nearest_trace_modal_eq_g"].mean()] + \
                        [collapse_per_step[s]["nearest_trace_modal_eq_g"].mean() for s in sorted_steps]
        axes[0].plot(steps, D_marg,  color="#4c72b0",
                      label=r"$D_{marg}$  JSD($\pi_{rl},\bar\pi_{sft}$)", **LINE_KW)
        axes[0].plot(steps, D_trace, color="#dd8452",
                      label=r"$D_{trace}$  min-JSD to an SFT trace", **{**LINE_KW, "marker": "s"})
        axes[0].plot(steps, C_trace, color="#55a868",
                      label=r"$C_{trace}=D_{marg}-D_{trace}$", **{**LINE_KW, "marker": "^"})
        axes[0].axhline(0, color="#d7191c", ls="--", lw=1.0, alpha=0.6)
        axes[0].axvline(0, color="#666", ls=":", lw=1)
        axes[0].set(xlabel="RL step  (0 = SFT baseline)", ylabel="JSD")
        axes[0].set_title("Distance to SFT marginal vs. nearest trace", pad=8)
        axes[0].legend(loc="best", **LEG_KW)
        axes[1].plot(steps, frac_correct, color="#1a9850",
                      label="nearest SFT trace is GT-modal", **LINE_KW)
        axes[1].plot(steps, [1-f for f in frac_correct], color="#d7191c",
                      label="nearest SFT trace is wrong-modal", **DASH_KW)
        axes[1].axvline(0, color="#666", ls=":", lw=1)
        axes[1].set(ylim=(0, 1), xlabel="RL step  (0 = SFT baseline)",
                     ylabel="fraction of states")
        axes[1].set_title("Whose conditional did RL collapse onto?", pad=8)
        axes[1].legend(loc="best", **LEG_KW)
    fig.suptitle(cfg, fontweight="bold", y=1.01)
    plt.tight_layout(); plt.savefig(out_dir / "03_rl_sft_trace_collapse.png", bbox_inches="tight")
    plt.close()

    # ── Plot 04: 3-panel entropy distributions across all puzzles ─────────
    print("[plot] 04_entropy_pretrain_vs_sft.png")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True, sharex=True)
    panels = [
        (r"pretrain marginal $H(\bar\pi_{pre})$", pre["H_marg"].dropna().values,        "#4c72b0"),
        (r"SFT marginal $H(\bar\pi_{sft})$",      sft["H_marg"].dropna().values,        "#dd8452"),
        (r"SFT per-trace mean $\overline{H}(\pi_i)$", sft["mean_per_trace_H"].dropna().values, "#55a868"),
    ]
    for ax, (lbl, v, c) in zip(axes, panels):
        x = np.sort(v); y = np.arange(1, len(x)+1) / max(1, len(x))
        ax.plot(x, y, color=c, lw=2.0)
        ax.fill_between(x, 0, y, color=c, alpha=0.20)
        ax.set_xlabel("entropy (nats)")
        ax.set_ylabel("ECDF")
        ax.set_title(lbl, pad=8)
        ax.text(0.04, 0.96, f"mean = {v.mean():.3f}\nmedian = {np.median(v):.3f}\nN = {len(v)}",
                  transform=ax.transAxes, va="top", ha="left",
                  bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.92,
                              edgecolor="#cccccc"), fontsize=10)
    fig.suptitle(f"Entropy distribution across all puzzles   ({cfg})",
                   fontweight="bold", y=1.02)
    plt.tight_layout(); plt.savefig(out_dir / "04_entropy_pretrain_vs_sft.png", bbox_inches="tight")
    plt.close()

    # ── SFT pass@k baseline (for plot 05 anchor line) ─────────────────────
    sft_pass = L.pass_at_k_for_states(
        sft.assign(n_correct_rl=sft["n_correct"], n_traces_rl=sft["n_traces"]),
        k_list=(1, 8, 16))
    print(f"[summary] SFT baseline: pass@1={sft_pass['pass@1']:.3f}  pass@8={sft_pass['pass@8']:.3f}  pass@16={sft_pass['pass@16']:.3f}")

    # ── Plot 05: Category summary (count + pass@16, with SFT baseline) ────
    print("[plot] 05_category_summary.png")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    if len(cat_counts):
        ax = axes[0]
        for c in CATEGORIES:
            ax.plot(cat_counts.index, cat_counts[c], color=CAT_COLORS[c],
                     label=cat_label(c, args.top_k), **LINE_KW)
        ax.set(xlabel="RL step", ylabel="# states")
        ax.set_title("Count per RL step", pad=8)
        ax.legend(loc="best", **LEG_KW)
    if len(summary_df):
        ax = axes[1]
        for c in CATEGORIES:
            col = f"{c}__pass@16"
            if col in summary_df.columns:
                ax.plot(summary_df["rl_step"], summary_df[col],
                         color=CAT_COLORS[c], label=cat_label(c, args.top_k), **LINE_KW)
        ax.axhline(sft_pass["pass@16"], color="black", ls="--", lw=1.4,
                    label=f"SFT pass@16 (overall) = {sft_pass['pass@16']:.3f}")
        ax.set(ylim=(0, 1.02), xlabel="RL step",
                ylabel="pass@16  (avg over states in cat)")
        ax.set_title("pass@16 per category", pad=8)
        ax.legend(loc="best", **LEG_KW)
    fig.suptitle(cfg, fontweight="bold", y=1.02)
    plt.tight_layout(); plt.savefig(out_dir / "05_category_summary.png", bbox_inches="tight")
    plt.close()

    # ── Plot 08: Δpass@k decomposition by category (waterfall) ────────────
    # For each RL step, decompose Δpass@k (RL minus SFT, averaged over the
    # RL-scored states) into per-category contributions.  Positives stack up,
    # negatives stack down, net Δ marked with a horizontal tick.
    print("[plot] 08_pass_at_k_decomposition.png")
    if len(cat_all):
        # Per-state SFT pass@k lookup (states are keyed by puzzle_id; load_raw
        # already restricts to turn=0 and test_B* brackets).
        def _passk_per_state(state_df, k):
            return {row["puzzle_id"]: L.pass_at_k_unbiased(int(row["n_correct"]),
                                                             int(row["n_traces"]), k)
                     for _, row in state_df.iterrows()}
        ks = (1, 16)
        sft_passk_by_state = {k: _passk_per_state(sft, k) for k in ks}
        # Build contribution table: rows=rl_step, cols=category, one matrix per k
        contrib = {k: pd.DataFrame(0.0, index=sorted(rl_states.keys()),
                                       columns=CATEGORIES) for k in ks}
        net = {k: {} for k in ks}
        for s, rl_st in rl_states.items():
            # Per-state RL pass@k at this step
            rl_passk = {k: _passk_per_state(rl_st, k) for k in ks}
            sub = cat_all[cat_all["rl_step"] == s]
            n_total = max(1, len(sub))  # one row per (state, step)
            for k in ks:
                # Mean SFT pass@k over the same states (restricted to those
                # that appear in both sft and the rl-scored set)
                sft_vals, rl_vals = [], []
                for _, r in sub.iterrows():
                    pid = r["puzzle_id"]
                    if pid in sft_passk_by_state[k] and pid in rl_passk[k]:
                        sft_vals.append(sft_passk_by_state[k][pid])
                        rl_vals.append(rl_passk[k][pid])
                if not rl_vals:
                    continue
                # Net Δpass@k (mean over states)
                net[k][s] = float(np.mean(rl_vals) - np.mean(sft_vals))
                # Per-category contribution: sum of (rl - sft) over states in cat
                # divided by N_total → sums across categories = net Δpass@k
                for c in CATEGORIES:
                    mask = (sub["category"] == c).values
                    if mask.sum() == 0: continue
                    delta = (np.asarray(rl_vals) - np.asarray(sft_vals))[mask]
                    contrib[k].loc[s, c] = float(delta.sum()) / n_total

        fig, axes = plt.subplots(1, len(ks), figsize=(7.5 * len(ks), 4.6))
        if len(ks) == 1: axes = [axes]
        steps = sorted(rl_states.keys())
        x = np.arange(len(steps))
        bar_w = 0.7
        for ax, k in zip(axes, ks):
            cdf = contrib[k].loc[steps]
            pos_bot = np.zeros(len(steps))
            neg_bot = np.zeros(len(steps))
            for c in CATEGORIES:
                vals = cdf[c].values
                pos = np.where(vals > 0, vals, 0.0)
                neg = np.where(vals < 0, vals, 0.0)
                if pos.any():
                    ax.bar(x, pos, bar_w, bottom=pos_bot, color=CAT_COLORS[c],
                            label=cat_label(c, top_k=args.top_k), edgecolor="black", linewidth=0.4)
                    pos_bot = pos_bot + pos
                if neg.any():
                    ax.bar(x, neg, bar_w, bottom=neg_bot, color=CAT_COLORS[c],
                            edgecolor="black", linewidth=0.4,
                            label=cat_label(c, top_k=args.top_k) if not pos.any() else None)
                    neg_bot = neg_bot + neg
            # Net Δ marker
            net_vals = [net[k].get(s, 0.0) for s in steps]
            ax.plot(x, net_vals, "o-", color="black", lw=1.4, ms=5,
                    label=f"net Δpass@{k}")
            ax.axhline(0, color="black", lw=0.7)
            ax.set_xticks(x); ax.set_xticklabels([str(s) for s in steps])
            ax.set_xlabel("RL step")
            ax.set_ylabel(f"Δpass@{k} contribution (vs SFT)")
            ax.set_title(f"pass@{k} decomposition by category")
            ax.grid(True, axis="y", ls="--", alpha=0.4)
        # Dedup legend across panels — collect from first ax
        h, l = axes[0].get_legend_handles_labels()
        seen = set(); h2, l2 = [], []
        for hi, li in zip(h, l):
            if li and li not in seen:
                seen.add(li); h2.append(hi); l2.append(li)
        fig.legend(h2, l2, loc="lower center", ncol=min(4, len(l2)), bbox_to_anchor=(0.5, -0.05),
                    **LEG_KW)
        fig.suptitle(cfg, fontweight="bold", y=1.02)
        plt.tight_layout(); _save_both(fig, out_dir / "08_pass_at_k_decomposition")
        plt.close()

    # ── Plot 09: Per-category pass@16 trajectory ──────────────────────────
    print("[plot] 09_pass_at_k_per_category.png")
    if len(cat_all):
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        steps = sorted(rl_states.keys())
        for c in CATEGORIES:
            ys = []
            for s in steps:
                col = f"{c}__pass@16"
                if col in summary_df.columns:
                    v = summary_df.loc[summary_df["rl_step"] == s, col]
                    ys.append(v.iloc[0] if len(v) else np.nan)
                else:
                    ys.append(np.nan)
            ax.plot(steps, ys, marker="o", lw=1.6, ms=5, color=CAT_COLORS[c],
                    label=cat_label(c, top_k=args.top_k))
        ax.axhline(sft_pass["pass@16"], color="black", ls="--", lw=1.2,
                    label=f"SFT pass@16 = {sft_pass['pass@16']:.3f}")
        # Overall RL pass@16 trajectory
        overall = [summary_df.loc[summary_df["rl_step"] == s, "pass@16"].iloc[0]
                    if (summary_df["rl_step"] == s).any() else np.nan
                    for s in steps]
        ax.plot(steps, overall, marker="s", lw=2.2, color="black",
                label="RL pass@16 (overall)")
        ax.set(xlabel="RL step", ylabel="pass@16 (mean over states in cat)",
                ylim=(0, 1.02))
        ax.set_title("pass@16 per category vs RL step")
        ax.legend(loc="best", ncol=2, **LEG_KW)
        fig.suptitle(cfg, fontweight="bold", y=1.02)
        plt.tight_layout(); _save_both(fig, out_dir / "09_pass_at_k_per_category")
        plt.close()

    # ── Plot 06: Category counts per bracket (one subplot per bin) ────────
    # Drops B0 (typically empty in our eval set); each panel auto-scales y so
    # mid-difficulty brackets aren't dwarfed by easy ones.
    print("[plot] 06_category_per_bin.png")
    if len(cat_all):
        bins_show = [b for b in L.BRACKETS if b != "test_B0"]
        # Drop empty bins entirely
        steps = sorted(cat_all["rl_step"].unique())
        bins_show = [b for b in bins_show
                      if (cat_all["data_source"] == b).sum() > 0]
        n_panels = len(bins_show)
        if n_panels:
            ncols = min(n_panels, 5)
            nrows = (n_panels + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols,
                                      figsize=(5.4 * ncols, 5.0 * nrows),
                                      sharex=True, sharey=False)
            axes_flat = np.atleast_1d(axes).flatten()
            legend_handles = None
            for j, (ax, ds) in enumerate(zip(axes_flat, bins_show)):
                n_total = (cat_all["data_source"] == ds).sum() // max(1, len(steps))
                series_for_panel = {}
                for c in CATEGORIES:
                    s_arr = [((cat_all["rl_step"] == s)
                                & (cat_all["category"] == c)
                                & (cat_all["data_source"] == ds)).sum()
                              for s in steps]
                    series_for_panel[c] = s_arr
                    ax.plot(steps, s_arr, color=CAT_COLORS[c],
                             label=cat_label(c, args.top_k), **LINE_KW)
                local_max = max((max(v) for v in series_for_panel.values()), default=1)
                ax.set_ylim(0, max(1, local_max) * 1.15)
                ax.set_title(f"{ds.replace('test_','')}   (n ≈ {n_total})",
                              fontweight="bold", color="black", pad=4)
                ax.set_xlabel("RL step")
                if j % ncols == 0:
                    ax.set_ylabel("# Samples")
                else:
                    ax.set_ylabel("")
                if legend_handles is None:
                    legend_handles = ax.get_legend_handles_labels()
            for ax_unused in axes_flat[len(bins_show):]:
                ax_unused.axis("off")
            fig.legend(*legend_handles, loc="lower center",
                        ncol=len(CATEGORIES), bbox_to_anchor=(0.5, -0.02),
                        **LEG_KW)
            plt.tight_layout(rect=[0, 0.05, 1, 0.97])
            _save_both(fig, out_dir / "06_category_per_bin")
            plt.close()

    # ── Plot 07: bin distribution per category, evolution across RL steps ──
    # One panel per category; x = bracket; one line per RL step (viridis gradient).
    print("[plot] 07_category_per_step.png")
    if len(cat_all):
        bins_show = [b for b in L.BRACKETS if b != "test_B0" and (cat_all["data_source"]==b).sum() > 0]
        steps = sorted(cat_all["rl_step"].unique())
        if bins_show and steps:
            ncats = len(CATEGORIES)
            fig, axes = plt.subplots(1, ncats, figsize=(5.4*ncats, 5.6),
                                      sharex=True, sharey=False)
            axes_flat = np.atleast_1d(axes).flatten()
            cmap_steps = plt.cm.viridis(np.linspace(0.15, 0.95, len(steps)))
            for j, (ax, c) in enumerate(zip(axes_flat, CATEGORIES)):
                local_max = 1
                for i, s in enumerate(steps):
                    counts = [((cat_all["rl_step"] == s)
                                & (cat_all["category"] == c)
                                & (cat_all["data_source"] == ds)).sum()
                              for ds in bins_show]
                    ax.plot(range(len(bins_show)), counts,
                             marker="o", ms=10, mec="black", mew=1.0, lw=3.4,
                             color=cmap_steps[i], label=f"step {s}")
                    local_max = max(local_max, max(counts) if counts else 0)
                ax.set_xticks(range(len(bins_show)))
                ax.set_xticklabels([b.replace("test_", "") for b in bins_show])
                ax.set_xlabel("Puzzle Difficulty Bins")
                if j == 0:
                    ax.set_ylabel("# Samples")
                else:
                    ax.set_ylabel("")
                ax.set_title(cat_label(c, args.top_k, multiline=True),
                              color="black", pad=4, fontweight="bold")
                ax.set_ylim(0, max(1, local_max) * 1.15)
            handles, labels = axes_flat[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", ncol=len(steps),
                        bbox_to_anchor=(0.5, -0.02), **LEG_KW)
            plt.tight_layout(rect=[0, 0.07, 1, 0.97])
            _save_both(fig, out_dir / "07_category_per_step")
            plt.close()

    # ── Plot 06b: proportion of each bracket assigned to each category ────
    # AGGREGATION NOTE: nothing is aggregated across RL steps in plots 06/06b/07/07b.
    # Each x-axis tick is a single RL step.  Counts/proportions are computed per
    # (rl_step × bracket × category); a "proportion" is the fraction of THAT
    # BRACKET's puzzles at THAT STEP that were assigned to a given category
    # (sums to 1 across categories within each (step, bracket) cell).
    print("[plot] 06b_category_per_bin_fraction.{png,pdf}")
    if len(cat_all):
        bins_show = [b for b in L.BRACKETS if b != "test_B0" and (cat_all["data_source"]==b).sum() > 0]
        steps = sorted(cat_all["rl_step"].unique())
        if bins_show and steps:
            fig, axes = plt.subplots(2, 3, figsize=(20, 11), sharex=True, sharey=False)
            axes_flat = np.atleast_1d(axes).flatten()
            legend_handles = None
            for j, (ax, ds) in enumerate(zip(axes_flat, bins_show)):
                local_max = 0.0
                for c in CATEGORIES:
                    fracs = []
                    for s in steps:
                        n_total = ((cat_all["rl_step"] == s) & (cat_all["data_source"] == ds)).sum()
                        n_cat   = ((cat_all["rl_step"] == s) & (cat_all["data_source"] == ds)
                                    & (cat_all["category"] == c)).sum()
                        fracs.append(n_cat / n_total if n_total > 0 else 0.0)
                    ax.plot(steps, fracs, color=CAT_COLORS[c],
                             label=cat_label(c, args.top_k), **LINE_KW)
                    if fracs: local_max = max(local_max, max(fracs))
                n_total = (cat_all["data_source"] == ds).sum() // max(1, len(steps))
                ax.set_title(f"{ds.replace('test_','')}   (n ≈ {n_total})",
                              fontweight="bold", color="black", pad=4)
                ax.set_xlabel("RL step")
                # Only leftmost column gets y-label
                if j % 3 == 0:
                    ax.set_ylabel("Prop. of Samples")
                else:
                    ax.set_ylabel("")
                ax.set_ylim(0, max(0.05, local_max) * 1.10)
                if legend_handles is None:
                    legend_handles = ax.get_legend_handles_labels()
            for ax_unused in axes_flat[len(bins_show):]:
                ax_unused.axis("off")
            fig.legend(*legend_handles, loc="lower center", ncol=len(CATEGORIES),
                        bbox_to_anchor=(0.5, -0.02), **LEG_KW)
            plt.tight_layout(rect=[0, 0.05, 1, 0.97])
            _save_both(fig, out_dir / "06b_category_per_bin_fraction")
            plt.close()

    # ── Plot 07b: per-category, proportion of each bracket (within-bracket
    #     proportion, evolving across RL steps).  Same denominator as 06b. ────
    print("[plot] 07b_category_per_step_fraction.{png,pdf}")
    if len(cat_all):
        bins_show = [b for b in L.BRACKETS if b != "test_B0" and (cat_all["data_source"]==b).sum() > 0]
        steps = sorted(cat_all["rl_step"].unique())
        if bins_show and steps:
            ncats = len(CATEGORIES)
            fig, axes = plt.subplots(1, ncats, figsize=(5.4*ncats, 5.6),
                                      sharex=True, sharey=False)
            axes_flat = np.atleast_1d(axes).flatten()
            cmap_steps = plt.cm.viridis(np.linspace(0.15, 0.95, len(steps)))
            for j, (ax, c) in enumerate(zip(axes_flat, CATEGORIES)):
                local_max = 0.0
                for i, s in enumerate(steps):
                    fracs = []
                    for ds in bins_show:
                        n_total = ((cat_all["rl_step"] == s) & (cat_all["data_source"] == ds)).sum()
                        n_cat   = ((cat_all["rl_step"] == s) & (cat_all["data_source"] == ds)
                                    & (cat_all["category"] == c)).sum()
                        fracs.append(n_cat / n_total if n_total > 0 else 0.0)
                    ax.plot(range(len(bins_show)), fracs,
                             marker="o", ms=10, mec="black", mew=1.0, lw=3.4,
                             color=cmap_steps[i], label=f"step {s}")
                    if fracs: local_max = max(local_max, max(fracs))
                ax.set_xticks(range(len(bins_show)))
                ax.set_xticklabels([b.replace("test_", "") for b in bins_show])
                ax.set_xlabel("Puzzle Difficulty Bins")
                # Only leftmost panel gets y-label
                if j == 0:
                    ax.set_ylabel("Prop. of Samples")
                else:
                    ax.set_ylabel("")
                ax.set_title(cat_label(c, args.top_k, multiline=True),
                              color="black", pad=4, fontweight="bold")
                ax.set_ylim(0, max(0.05, local_max) * 1.10)
            handles, labels = axes_flat[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", ncol=len(steps),
                        bbox_to_anchor=(0.5, -0.02), **LEG_KW)
            plt.tight_layout(rect=[0, 0.07, 1, 0.97])
            _save_both(fig, out_dir / "07b_category_per_step_fraction")
            plt.close()

    # ── Per-category folders + examples (one figure per puzzle, all RL steps) ─
    if not args.skip_categories and len(cat_all):
        print("\n[categories] rendering example states per category (one figure per puzzle)")
        cat_root = out_dir / "categories"
        cat_root.mkdir(exist_ok=True)
        # alpha_star per RL step (KL fit)
        alpha_dict = dict(zip(
            sharp_df.loc[sharp_df["stage"].str.startswith("sft→rl"), "rl_step"].astype(int),
            sharp_df.loc[sharp_df["stage"].str.startswith("sft→rl"), "kl_alpha_global"].astype(float),
        ))
        for c in CATEGORIES:
            ex_dir = cat_root / c / "examples"
            ex_dir.mkdir(parents=True, exist_ok=True)
            # Clear stale per-step PNGs from previous (per-step) naming
            for stale in ex_dir.glob("step_*_puzzle_*.png"):
                stale.unlink()
            sub = cat_all[cat_all["category"] == c]
            if len(sub) == 0:
                print(f"  {c}: no examples")
                continue
            # pass@k per step manifest (unchanged)
            pk_rows = []
            for s in sorted(sub["rl_step"].unique()):
                sub_s = sub[sub["rl_step"] == s]
                pk = L.pass_at_k_for_states(sub_s, k_list=(1, 8, 16))
                pk_rows.append({"rl_step": s, "n": len(sub_s), **pk})
            pd.DataFrame(pk_rows).to_csv(cat_root / c / "pass_at_k_per_step.csv", index=False)
            # Pick top puzzles by max severity across all RL steps where they were tagged in c
            sub_pool = sub
            if args.example_bracket:
                sub_pool = sub[sub["data_source"] == args.example_bracket]
                if len(sub_pool) == 0:
                    print(f"  {c}: no examples in {args.example_bracket}, skipping")
                    continue
            puzzle_severity = sub_pool.groupby("puzzle_id")["severity"].max().sort_values(ascending=False)
            top_puzzles = puzzle_severity.head(args.n_examples).index.tolist()
            manifest_rows = []
            for pid in top_puzzles:
                pid_subs = sub[sub["puzzle_id"] == pid].sort_values("rl_step")
                if len(pid_subs) == 0: continue
                steps_in_cat = sorted(pid_subs["rl_step"].astype(int).tolist())
                ds = pid_subs.iloc[0]["data_source"]
                title = (f"{c}    puzzle={pid}    ds={ds}    "
                          f"steps in category: {steps_in_cat}    "
                          f"max |ΔM|={pid_subs['severity'].max():.2f}")
                out_png = cat_root / c / "examples" / f"puzzle_{pid}.png"
                L.plot_state_three_panel(pid, pre, sft, rl_states,
                                            top_n=8, title_suffix=title,
                                            out_path=out_png)
                # Per-step diagnostics for the manifest
                for _, row in pid_subs.iterrows():
                    manifest_rows.append(dict(
                        puzzle_id=pid, rl_step=int(row["rl_step"]), data_source=ds,
                        rank_sft=int(row["rank_sft"]), rank_rl=int(row["rank_rl"]),
                        p_sft_g=float(row["p_sft_g"]), p_rl_g=float(row["p_rl_g"]),
                        delta_p_g=float(row["delta_p_g"]), severity=float(row["severity"]),
                        png=str(out_png.relative_to(out_dir))))
            if manifest_rows:
                pd.DataFrame(manifest_rows).to_csv(cat_root / c / "manifest.csv", index=False)
            print(f"  {c}: {len(top_puzzles)} puzzle figures rendered")

    print(f"\n[done] all outputs in {out_dir}")


if __name__ == "__main__":
    main()
