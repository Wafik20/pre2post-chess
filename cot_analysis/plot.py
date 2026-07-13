"""Generate the 11-metric figure(s) from prompt_df.parquet.

Plots:
  fig_main.png            — all 11 metrics in a single 1×11 strip
                            (alternative: pass --grid 3 4 for a 3×4 grid)
  fig_move_quality.png    — candidate vs opponent quality overlaid
                            (candidate solid, opponent dashed)

Style matches the paper's other figures (serif font, larger labels,
viridis-by-size color when multiple runs are passed).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PLOT_STYLE = {
    "font.family":     "serif",
    "font.serif":      ["DejaVu Serif", "Liberation Serif", "Times", "serif"],
    "font.size":       18,
    "axes.titlesize":  17,
    "axes.labelsize":  18,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "legend.fontsize": 17,
    "axes.linewidth":  1.4,
    "grid.linewidth":  0.9,
    "lines.linewidth": 2.2,
}

# Display each (metric → (panel title, transform_to_apply)).
#
# Some norm-rank metrics are "lower is better"; we plot 1 - x so the
# y-axis follows the "up is better" convention.
INVERT_FOR_DISPLAY = {
    "mean_candidate_norm_rank", "mean_opponent_norm_rank",
    "best_candidate_norm_rank", "best_opponent_norm_rank",
    "selected_norm_rank_self",
}


def _transform(series: pd.Series, metric: str) -> pd.Series:
    if metric in INVERT_FOR_DISPLAY:
        return 1.0 - series
    return series


# Per-prompt aggregation → step-mean ± SEM
def _series(prompt_df: pd.DataFrame, metric: str) -> tuple[pd.Series, pd.Series]:
    g = prompt_df.groupby("step")[metric]
    return _transform(g.mean(), metric), g.sem()


# Color per run: viridis ordered by log10(params estimated from run_tag).
def _colors_for_runs(run_tags: list[str]) -> dict[str, tuple]:
    # rough param estimate from "<X>m" suffix in tag (e.g. "20m", "50m")
    def parse_n(tag: str) -> float:
        for part in tag.split("_"):
            if part.endswith("m") and part[:-1].isdigit():
                return float(part[:-1]) * 1e6
        return 1.0
    ns = np.log10([parse_n(t) for t in run_tags])
    if ns.max() == ns.min():
        return {t: plt.cm.tab10(i) for i, t in enumerate(run_tags)}
    norm = plt.Normalize(vmin=ns.min() - 0.2, vmax=ns.max() + 0.2)
    return {t: plt.cm.viridis(norm(n)) for t, n in zip(run_tags, ns)}


def _draw(ax, x, y, e, color, label=None, linestyle="-", marker="o"):
    order = np.argsort(x)
    x = np.asarray(x)[order]; y = np.asarray(y)[order]; e = np.asarray(e)[order]
    ax.plot(x, y, color=color, lw=2.4, alpha=0.85, linestyle=linestyle, zorder=2)
    ax.errorbar(x, y, yerr=e, fmt="none", ecolor=color, elinewidth=1.0,
                capsize=3, alpha=0.85, zorder=3)
    ax.scatter(x, y, s=110, color=color, marker=marker,
               edgecolors="white", linewidths=0.9, zorder=4, label=label)


# Each entry: (column_name, panel_title, panel_kind)
#   panel_kind = "single"   one curve per run
#   panel_kind = "quality"  two curves per run (candidate solid + opponent dashed)
PANELS: list[tuple[str, str, str]] = [
    ("candidate_count",       "Number of candidates",          "single"),
    ("width_depth_index",     r"Width-to-depth  $|L|/D$",      "single"),
    ("__move_quality__",      "Move quality",                  "quality"),
    ("target_in_any_depth",   "Target move in CoT",            "single"),
    ("dfs_consistency",       r"DFS consistency  $\tau$",      "single"),
    ("revisit_rate",          "Revisit rate",                  "single"),
    ("reasoning_char_count",  "Reasoning length (chars)",      "single"),
    ("n_ordered_visits",      "Move tokens written",           "single"),
    ("effective_branching",   "Avg branching factor",          "single"),
    ("max_depth",             r"Max depth  $D$",               "single"),
    ("commit_quality",        "Commit quality",                "single"),
]


def panel(ax, metric, title, prompt_dfs, colors):
    for tag, df in prompt_dfs.items():
        if metric not in df.columns:
            continue
        m, s = _series(df, metric)
        _draw(ax, m.index.values, m.values, s.values, colors[tag], label=tag)
    ax.set_title(title, fontweight="bold", pad=4)
    ax.set_xlabel("RL steps")
    ax.grid(True, which="major", linestyle="--", alpha=0.45)


def panel_quality(ax, prompt_dfs, colors):
    for tag, df in prompt_dfs.items():
        c = colors[tag]
        for metric, ls, marker, role in [
            ("mean_candidate_norm_rank", "-",  "o", "candidate"),
            ("mean_opponent_norm_rank",  "--", "s", "opponent"),
        ]:
            if metric not in df.columns:
                continue
            m, s = _series(df, metric)
            _draw(ax, m.index.values, m.values, s.values, c,
                  label=f"{tag} {role}", linestyle=ls, marker=marker)
    ax.set_title("Move quality", fontweight="bold", pad=4)
    ax.set_xlabel("RL steps")
    ax.set_ylabel(r"$1 - \mathrm{norm\ rank}$")
    ax.grid(True, which="major", linestyle="--", alpha=0.45)


def make_main_figure(prompt_dfs: dict[str, pd.DataFrame], out_path: Path,
                     grid: tuple[int, int] | None = None) -> None:
    n = len(PANELS)
    if grid is None:
        nrows, ncols = 1, n
    else:
        nrows, ncols = grid
    assert nrows * ncols >= n

    colors = _colors_for_runs(list(prompt_dfs.keys()))
    figsize = (3.4 * ncols + 0.5, 4.2 * nrows + 0.3)

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        for i, (metric, title, kind) in enumerate(PANELS):
            ax = axes[i // ncols, i % ncols]
            if kind == "quality":
                panel_quality(ax, prompt_dfs, colors)
            else:
                panel(ax, metric, title, prompt_dfs, colors)
        for j in range(n, nrows * ncols):
            axes[j // ncols, j % ncols].axis("off")

        # legend: model tags + role styles
        handles = [mlines.Line2D([0], [0], color=colors[t], marker="o",
                                  lw=2.6, ms=12, mec="white", mew=0.9, label=t)
                   for t in prompt_dfs]
        handles += [
            mlines.Line2D([0], [0], color="black", marker="o", lw=2.4, ms=11,
                          linestyle="-",  mec="white", mew=0.9, label="candidate"),
            mlines.Line2D([0], [0], color="black", marker="s", lw=2.4, ms=11,
                          linestyle="--", mec="white", mew=0.9, label="opponent"),
        ]
        fig.legend(handles=handles, loc="upper center",
                   bbox_to_anchor=(0.5, 1.04), ncol=len(handles),
                   frameon=True, framealpha=0.92, edgecolor="#cccccc",
                   columnspacing=1.4, handletextpad=0.5, handlelength=2.0)
        fig.tight_layout(rect=[0, 0, 1, 0.92], w_pad=0.4, h_pad=0.6)
        fig.savefig(out_path.with_suffix(".png"), dpi=170,
                    bbox_inches="tight", facecolor="white")
        fig.savefig(out_path.with_suffix(".pdf"),
                    bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"wrote {out_path.with_suffix('.png')}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, nargs="+", required=True,
                    help="One or more directories containing prompt_df.parquet. "
                         "Each is plotted as a separate series.")
    ap.add_argument("--tags", type=str, nargs="+", default=None,
                    help="Display tags (one per --data dir). "
                         "Default: basename of the data dir.")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output directory.")
    ap.add_argument("--grid", type=int, nargs=2, default=None,
                    help="Optional (rows, cols) grid override "
                         "(default: 1×N strip).")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tags = args.tags or [d.name for d in args.data]
    assert len(tags) == len(args.data), "tags and data must align"
    prompt_dfs = {tag: pd.read_parquet(d / "prompt_df.parquet")
                  for tag, d in zip(tags, args.data)}
    print(f"loaded runs: {list(prompt_dfs.keys())}")

    grid = tuple(args.grid) if args.grid else None
    make_main_figure(prompt_dfs, args.out / "fig_main", grid=grid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
