"""
policy_analysis_lib.py — shared helpers for run_policy_analysis.py.

Implements move-space sharpening (KL-optimal α*), trace sensitivity, trace
collapse, the five qualitative categories from the paper, and the plotting
primitives used by run_policy_analysis.py.

All functions consume "state" DataFrames produced by build_state(), which
turn raw_scores.parquet into one row per (puzzle, turn=0) carrying the
trace-marginal move distribution, per-trace distributions on a common
union vocab, and the standard scalar diagnostics.
"""

from __future__ import annotations

import math
import os
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.optimize import minimize_scalar

# ── Constants ────────────────────────────────────────────────────────────────

# Root of the per-(stage, step) raw_scores tree produced by Stage 1
# (measure_policy.py).  Layout:
#   RESULTS_ROOT/<cfg>/<stage>/<step>/raw_scores.parquet
# Override with the RESULTS_ROOT env var, or run_policy_analysis.py --results-root.
# Default: ./results next to this file.
RESULTS_ROOT = Path(os.environ.get(
    "RESULTS_ROOT", Path(__file__).resolve().parent / "results"))
BRACKETS = ["test_B0", "test_B1", "test_B2", "test_B3", "test_B4", "test_B5"]
EPS = 1e-12


# ── Distribution primitives ──────────────────────────────────────────────────

def softmax_from_log(log_vals) -> np.ndarray:
    x = np.asarray(log_vals, dtype=np.float64)
    return np.exp(x - logsumexp(x))


def kl(p, q, eps=EPS) -> float:
    p = np.asarray(p, dtype=np.float64); q = np.asarray(q, dtype=np.float64)
    p = p / max(p.sum(), eps); q = q / max(q.sum(), eps)
    mask = p > eps
    return float(np.sum(p[mask] * (np.log(p[mask]) - np.log(np.clip(q[mask], eps, 1.0)))))


def jsd(p, q, eps=EPS) -> float:
    p = np.asarray(p, dtype=np.float64); q = np.asarray(q, dtype=np.float64)
    p = p / max(p.sum(), eps); q = q / max(q.sum(), eps)
    m = 0.5 * (p + q)
    return 0.5 * kl(p, m, eps) + 0.5 * kl(q, m, eps)


def entropy(p, eps=EPS) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = p / max(p.sum(), eps)
    mask = p > eps
    return float(-np.sum(p[mask] * np.log(p[mask])))


def alpha_power(p, alpha, eps=EPS) -> np.ndarray:
    """π_α(a) ∝ p(a)^α; α=1 → identity, α>1 → sharper, α<1 → smoother."""
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    pa = p ** alpha
    return pa / pa.sum()


# ── Config / paths ───────────────────────────────────────────────────────────

def config_id(total_compute: str, modelsize: str, alpha: str, beta: str) -> str:
    return f"{total_compute}_{modelsize}_alpha{alpha}_beta{beta}"


def stage_step_path(cfg: str, stage: str, step: str) -> Path:
    return RESULTS_ROOT / cfg / stage / step / "raw_scores.parquet"


def discover_rl_steps(cfg: str) -> list[int]:
    rl_dir = RESULTS_ROOT / cfg / "rl"
    if not rl_dir.exists(): return []
    out = []
    for sd in rl_dir.iterdir():
        if sd.is_dir() and (sd / "raw_scores.parquet").exists() and (sd / "raw_scores.parquet").stat().st_size > 0:
            out.append(int(sd.name.replace("step_", "")))
    return sorted(out)


# ── Per-puzzle state ────────────────────────────────────────────────────────

_RAW_COLS = (
    "puzzle_id", "data_source", "turn", "gt_uci", "fen",
    "legal_moves_uci", "log_pi_tilde", "outcome",
)


def load_raw(path: Path) -> pd.DataFrame | None:
    """Load raw_scores.parquet, restrict to turn=0 and test_B* brackets."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return None
    df = pd.read_parquet(p, columns=list(_RAW_COLS))
    df = df[(df["turn"] == 0) & df["data_source"].str.startswith("test_B")]
    return df.reset_index(drop=True)


def build_state(raw: pd.DataFrame) -> pd.DataFrame:
    """Per-puzzle state: union vocab, marginal distribution, per-trace stats.

    Returns a DataFrame keyed by puzzle_id with columns:
      data_source, gt, all_u, marg, traces (list of np.array on union vocab),
      modal, pi_g, rank_g, margin, mean_per_trace_H, H_marg, TS_H,
      trace_modals, trace_pi_g, trace_rank_g, mean_outcome, max_outcome.
    """
    out = []
    for pid, sub in raw.groupby("puzzle_id"):
        gt = sub.iloc[0]["gt_uci"]
        ds = sub.iloc[0]["data_source"]
        fen = sub.iloc[0]["fen"] if "fen" in sub.columns else None
        traces_raw = []          # normalized per-trace π
        traces_unnorm_raw = []   # raw exp(log_pi_tilde) — does NOT sum to 1
        for _, r in sub.iterrows():
            u = list(r["legal_moves_uci"])
            log_pi = np.asarray(r["log_pi_tilde"])
            traces_raw.append((u, softmax_from_log(log_pi)))
            traces_unnorm_raw.append((u, np.exp(log_pi)))
        # Union vocab
        all_u = list(dict.fromkeys([u for u_, _ in traces_raw for u in u_]))
        idx = {u: i for i, u in enumerate(all_u)}
        # Embed per-trace normalized + unnormalized distributions on union vocab
        traces = []
        traces_unnorm = []
        for (u_, p_), (_, pu_) in zip(traces_raw, traces_unnorm_raw):
            v = np.zeros(len(all_u)); vu = np.zeros(len(all_u))
            for u, p, pu in zip(u_, p_, pu_):
                v[idx[u]]  = p
                vu[idx[u]] = pu
            traces.append(v)
            traces_unnorm.append(vu)
        marg = np.mean(traces, axis=0)
        marg_unnorm = np.mean(traces_unnorm, axis=0)
        modal = all_u[int(np.argmax(marg))]
        pi_g = float(marg[idx[gt]]) if gt in idx else 0.0
        if gt in idx and pi_g > 0:
            mask = np.ones_like(marg, dtype=bool); mask[idx[gt]] = False
            top_w = marg[mask].max() if mask.any() else EPS
            margin = float(np.log(max(pi_g, EPS)) - np.log(max(top_w, EPS)))
            rank_g = int(np.where(np.argsort(-marg) == idx[gt])[0][0]) + 1
        else:
            margin = np.nan; rank_g = 9999
        trace_modals = [all_u[int(np.argmax(t))] for t in traces]
        trace_pi_g   = [float(t[idx[gt]]) if gt in idx else 0.0 for t in traces]
        trace_rank_g = [int(np.where(np.argsort(-t) == idx[gt])[0][0]) + 1 if gt in idx else 9999 for t in traces]
        mean_per_trace_H = float(np.mean([entropy(t) for t in traces]))
        H_marg = entropy(marg)
        TS_H = max(0.0, H_marg - mean_per_trace_H)
        outcomes = sub["outcome"].astype(float).values
        out.append(dict(
            puzzle_id=pid, data_source=ds, gt=gt, fen=fen,
            all_u=all_u, marg=marg, traces=traces,
            marg_unnorm=marg_unnorm, traces_unnorm=traces_unnorm,
            modal=modal, pi_g=pi_g, rank_g=rank_g, margin=margin,
            mean_per_trace_H=mean_per_trace_H, H_marg=H_marg, TS_H=TS_H,
            trace_modals=trace_modals, trace_pi_g=trace_pi_g, trace_rank_g=trace_rank_g,
            mean_outcome=float(np.mean(outcomes)), max_outcome=float(np.max(outcomes)),
            n_correct=int(np.sum(outcomes > 0)), n_traces=int(len(outcomes)),
        ))
    return pd.DataFrame(out)


def cache_state(state: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(state, f)


def load_state_cache(path: Path) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def build_or_load_state(stage: str, step: str, cfg: str, cache_dir: Path) -> pd.DataFrame | None:
    """Per-(cfg, stage, step) per-puzzle state, with on-disk pickle cache."""
    cache = cache_dir / f"state_v3_{stage}_{step}.pkl"  # v3: also stores unnormalized exp(log_pi_tilde)
    cached = load_state_cache(cache)
    if cached is not None:
        return cached
    raw = load_raw(stage_step_path(cfg, stage, step))
    if raw is None or len(raw) == 0:
        return None
    st = build_state(raw)
    cache_state(st, cache)
    return st


# ── Move-space sharpening (KL fit) ──────────────────────────────────────────

def _join_states(rl: pd.DataFrame, sft: pd.DataFrame) -> pd.DataFrame:
    """Inner join on puzzle_id and align trace marginals to a shared union vocab.

    Returns DataFrame with columns: puzzle_id, data_source, gt,
      all_u, p_sft, p_rl  (all on the joint union of SFT∪RL vocabularies).
    """
    rl2 = rl.set_index("puzzle_id"); sft2 = sft.set_index("puzzle_id")
    common = rl2.index.intersection(sft2.index)
    out = []
    for pid in common:
        s = sft2.loc[pid]; r = rl2.loc[pid]
        all_u = list(dict.fromkeys(list(s["all_u"]) + list(r["all_u"])))
        idx = {u: i for i, u in enumerate(all_u)}
        def expand(src_u, src_v):
            out = np.zeros(len(all_u))
            for u, v in zip(src_u, src_v):
                out[idx[u]] = v
            return out
        p_sft = expand(s["all_u"], s["marg"])
        p_rl  = expand(r["all_u"], r["marg"])
        out.append(dict(puzzle_id=pid, data_source=s["data_source"], gt=s["gt"],
                         all_u=all_u, p_sft=p_sft, p_rl=p_rl,
                         sft_traces_idx=[expand(s["all_u"], t) for t in s["traces"]]))
    return pd.DataFrame(out)


def _fit_alpha_kl_one(p_src, p_tgt, alpha_max=5.0):
    """KL-optimal α for ONE (p_src, p_tgt) pair.  Returns (α*, residual_JSD, baseline_JSD)."""
    def loss(alpha):
        return kl(p_tgt, alpha_power(p_src, alpha))
    res = minimize_scalar(loss, bounds=(0.0, alpha_max), method="bounded",
                           options={"xatol": 1e-3})
    a_star = float(res.x)
    p_a = alpha_power(p_src, a_star)
    return a_star, float(jsd(p_tgt, p_a)), float(jsd(p_tgt, p_src))


def fit_alpha_kl_per_state(target: pd.DataFrame, source_col: str = "p_sft",
                              target_col: str = "p_rl", alpha_max: float = 5.0) -> pd.DataFrame:
    """Per-state KL-optimal α* (one fit per puzzle).  Returns DataFrame with
    columns: puzzle_id, alpha_star, residual_jsd, baseline_jsd, explained_sharp.
    """
    rows = []
    for _, r in target.iterrows():
        a_star, residual, baseline = _fit_alpha_kl_one(r[source_col], r[target_col], alpha_max)
        explained = (1.0 - residual / baseline) if baseline > EPS else np.nan
        rows.append({
            "puzzle_id": r["puzzle_id"], "data_source": r.get("data_source"),
            "alpha_star": a_star, "residual_jsd": residual,
            "baseline_jsd": baseline, "explained_sharp": float(explained),
        })
    return pd.DataFrame(rows)


def _centered_log(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Per-state centered log-probability:  log p(a) − mean_b log p(b).

    Under a power transform p_α(a) = p(a)^α / Z, this satisfies
    centered_log(p_α)(a) = α · centered_log(p)(a) exactly, so a linear
    regression of y = centered_log(q) on x = centered_log(p) returns
    slope β = α and R² = 1 for any pure power-sharpened distribution.
    """
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0)
    lp = np.log(p)
    return lp - lp.mean()


def _ols_with_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Returns (slope, intercept, R²) for y ≈ slope·x + intercept."""
    if len(x) < 2 or np.std(x) == 0:
        return float("nan"), float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), float(r2)


def fit_loglin_global(target: pd.DataFrame, source_col: str = "p_sft",
                         target_col: str = "p_rl") -> dict:
    """Pool every (state, action) pair and fit
       centered_log(p_target)(a) = β · centered_log(p_source)(a) + intercept.

    For a pure power-sharpened source, β recovers α* exactly with R² = 1.
    Returns {'slope', 'intercept', 'r2', 'n_points'}.
    """
    xs, ys = [], []
    for _, r in target.iterrows():
        xs.append(_centered_log(np.asarray(r[source_col])))
        ys.append(_centered_log(np.asarray(r[target_col])))
    if not xs:
        return {"slope": float("nan"), "intercept": float("nan"),
                "r2": float("nan"), "n_points": 0}
    x = np.concatenate(xs); y = np.concatenate(ys)
    slope, intercept, r2 = _ols_with_r2(x, y)
    return {"slope": slope, "intercept": intercept, "r2": r2, "n_points": int(len(x))}


def fit_loglin_per_state(target: pd.DataFrame, source_col: str = "p_sft",
                            target_col: str = "p_rl") -> pd.DataFrame:
    """Per-state centered-log-prob linear fit.  Returns DataFrame with
    puzzle_id, slope (β), intercept (~0 by construction), r2.
    """
    rows = []
    for _, r in target.iterrows():
        x = _centered_log(np.asarray(r[source_col]))
        y = _centered_log(np.asarray(r[target_col]))
        slope, intercept, r2 = _ols_with_r2(x, y)
        rows.append({"puzzle_id": r["puzzle_id"], "data_source": r.get("data_source"),
                     "slope": slope, "intercept": intercept, "r2": r2})
    return pd.DataFrame(rows)


# Back-compat aliases (kept so old call sites keep working — internals are now
# centered-log-prob, not binary logit).
fit_logit_linear_global    = fit_loglin_global
fit_logit_linear_per_state = fit_loglin_per_state


def fit_alpha_kl(target: pd.DataFrame, source_col: str = "p_sft",
                  target_col: str = "p_rl", alpha_max: float = 5.0) -> dict:
    """Single global α* minimizing E_s D_KL(π_target ‖ π_α(source)).

    Returns dict with α*, residual_jsd, explained_sharp, baseline_jsd,
    boundary flag, and per-state KL at α*.
    """
    sources = list(target[source_col].values)
    targets = list(target[target_col].values)

    def loss(alpha):
        L = 0.0
        for p_src, p_tgt in zip(sources, targets):
            p_alpha = alpha_power(p_src, alpha)
            L += kl(p_tgt, p_alpha)
        return L / max(1, len(sources))

    res = minimize_scalar(loss, bounds=(0.0, alpha_max), method="bounded",
                           options={"xatol": 1e-3})
    a_star = float(res.x); fit_loss = float(res.fun)
    boundary = (a_star <= 0.05) or (a_star >= alpha_max - 0.05)

    # Residual JSD at α*
    jsd_at_alpha = []
    jsd_baseline = []
    for p_src, p_tgt in zip(sources, targets):
        p_a = alpha_power(p_src, a_star)
        jsd_at_alpha.append(jsd(p_tgt, p_a))
        jsd_baseline.append(jsd(p_tgt, p_src))
    jsd_at_alpha = np.array(jsd_at_alpha); jsd_baseline = np.array(jsd_baseline)
    residual = float(jsd_at_alpha.mean())
    base = float(jsd_baseline.mean())
    explained = float(1.0 - residual / max(base, EPS)) if base > 0 else np.nan
    return dict(
        alpha_star=a_star, fit_loss=fit_loss, boundary=boundary,
        residual_jsd=residual, baseline_jsd=base, explained_sharp=explained,
        per_state_jsd_at_alpha=jsd_at_alpha,
        per_state_jsd_baseline=jsd_baseline,
    )


# ── Trace sensitivity ────────────────────────────────────────────────────────

def trace_sensitivity_per_state(row) -> float:
    """TS(s) = (1/N) Σ_i KL(π_i ‖ π̄)."""
    marg = row["marg"]; traces = row["traces"]
    return float(np.mean([kl(t, marg) for t in traces]))


def aggregate_TS(state: pd.DataFrame) -> dict:
    vals = state.apply(trace_sensitivity_per_state, axis=1)
    return dict(mean=float(vals.mean()), median=float(vals.median()),
                values=vals.values)


# ── Trace collapse ──────────────────────────────────────────────────────────

def collapse_per_state(rl_row, sft_row) -> dict:
    """Compute D_marg(s), D_trace(s), C_trace(s), and nearest-trace info."""
    all_u = list(dict.fromkeys(list(rl_row["all_u"]) + list(sft_row["all_u"])))
    idx = {u: i for i, u in enumerate(all_u)}
    def expand(src_u, src_v):
        out = np.zeros(len(all_u))
        for u, v in zip(src_u, src_v):
            out[idx[u]] = v
        return out
    p_rl  = expand(rl_row["all_u"], rl_row["marg"])
    p_sft = expand(sft_row["all_u"], sft_row["marg"])
    sft_traces = [expand(sft_row["all_u"], t) for t in sft_row["traces"]]
    d_marg = jsd(p_rl, p_sft)
    d_trace_list = [jsd(p_rl, t) for t in sft_traces]
    i_star = int(np.argmin(d_trace_list))
    d_trace = float(d_trace_list[i_star])
    nt_modal = all_u[int(np.argmax(sft_traces[i_star]))]
    return dict(d_marg=d_marg, d_trace=d_trace, c_trace=d_marg - d_trace,
                nearest_trace_idx=i_star, nearest_trace_modal=nt_modal,
                nearest_trace_modal_eq_g=(nt_modal == rl_row["gt"]))


def intra_sft_collapse(sft_state: pd.DataFrame) -> pd.DataFrame:
    """Self-comparison baseline: SFT vs SFT.  D_marg = 0 by definition;
    D_trace = min_i JSD(π̄_sft, π_i^sft) per state.  Returns DataFrame with
    same shape as aggregate_collapse() so plots can prepend it as rl_step=0.
    """
    rows = []
    for _, r in sft_state.iterrows():
        marg = r["marg"]; traces = r["traces"]
        d_traces = [jsd(marg, t) for t in traces]
        i_star = int(np.argmin(d_traces))
        nt_modal = r["all_u"][int(np.argmax(traces[i_star]))]
        rows.append({
            "puzzle_id": r["puzzle_id"], "data_source": r["data_source"], "gt": r["gt"],
            "d_marg": 0.0, "d_trace": float(d_traces[i_star]),
            "c_trace": -float(d_traces[i_star]),  # D_marg − D_trace, so negative
            "nearest_trace_idx": i_star, "nearest_trace_modal": nt_modal,
            "nearest_trace_modal_eq_g": (nt_modal == r["gt"]),
        })
    return pd.DataFrame(rows)


def aggregate_collapse(rl_state: pd.DataFrame, sft_state: pd.DataFrame) -> pd.DataFrame:
    sft_lookup = sft_state.set_index("puzzle_id")
    rows = []
    for _, r in rl_state.iterrows():
        if r["puzzle_id"] not in sft_lookup.index: continue
        s = sft_lookup.loc[r["puzzle_id"]]
        c = collapse_per_state(r, s)
        rows.append({"puzzle_id": r["puzzle_id"], "data_source": r["data_source"],
                     "gt": r["gt"], **c})
    return pd.DataFrame(rows)


# ── Five qualitative categories ──────────────────────────────────────────────

def margin_logit(p) -> float:
    """log(p / (1-p)) — the success-margin definition from the paper."""
    p = float(np.clip(p, EPS, 1 - EPS))
    return float(np.log(p / (1 - p)))


def classify_state(sft_row, rl_row, eps_tail: float = 0.05,
                     top_k: int = 1) -> dict:
    """k-generalized top-K transition categorization matching the paper spec.

    With top_k=1, this is exactly the strict spec (a* ∈ T_θ^1 ⟺ θ's modal = a*).
    With top_k>1, every "in T_θ^k" check becomes "rank ≤ top_k", so the relaxed
    version counts states where RL pushed GT into the top-K (not necessarily
    top-1) — i.e. "increase by at least a top-K rank" rather than "become top-1".

    Spec (k-generalized):
      1. gt_amplification           rank_sft ≤ k, rank_rl ≤ k, Δp(a*) > 0
      2. tail_discovery             rank_sft > k, rank_rl ≤ k, π_sft(a*) <  ε_tail
      3. topk_correction            rank_sft > k, rank_rl ≤ k, π_sft(a*) ≥ ε_tail
      4. gt_regression              rank_sft ≤ k, rank_rl > k
      5. wrong_mode_amplification   rank_sft > k, rank_rl > k, w_sft ∈ T_rl^k, Δp(w_sft) > 0
                                     (w_sft = sft_modal; "w_sft ∈ T_rl^k" = sft_modal's
                                      rank under π_rl ≤ k)
      6. other                      everything else
    """
    gt = sft_row["gt"]
    p_sft_g  = float(sft_row["pi_g"]); p_rl_g  = float(rl_row["pi_g"])
    rank_sft = int(sft_row["rank_g"]); rank_rl = int(rl_row["rank_g"])
    sft_top = sft_row["modal"]; rl_top = rl_row["modal"]
    sft_in_topK = (rank_sft <= top_k)
    rl_in_topK  = (rank_rl  <= top_k)
    delta_p_g = p_rl_g - p_sft_g

    delta_p_w = float("nan")
    cat = "other"
    if sft_in_topK and rl_in_topK:
        if delta_p_g > 0:
            cat = "gt_amplification"
    elif (not sft_in_topK) and rl_in_topK:
        cat = "tail_discovery" if p_sft_g < eps_tail else "topk_correction"
    elif sft_in_topK and (not rl_in_topK):
        cat = "gt_regression"
    elif (not sft_in_topK) and (not rl_in_topK):
        # w_sft ∈ T_rl^k ⟺ sft_top's rank under π_rl ≤ k.
        sft_u = list(sft_row["all_u"]); rl_u = list(rl_row["all_u"])
        if sft_top in rl_u:
            rl_marg = rl_row["marg"]
            sft_top_idx = rl_u.index(sft_top)
            p_rl_w = float(rl_marg[sft_top_idx])
            rank_sft_top_in_rl = int((rl_marg > p_rl_w).sum()) + 1
            if rank_sft_top_in_rl <= top_k and sft_top in sft_u:
                p_sft_w = float(sft_row["marg"][sft_u.index(sft_top)])
                delta_p_w = p_rl_w - p_sft_w
                if delta_p_w > 0:
                    cat = "wrong_mode_amplification"

    if cat == "wrong_mode_amplification":
        severity = abs(delta_p_w) if not math.isnan(delta_p_w) else 0.0
    elif cat in ("tail_discovery", "topk_correction"):
        severity = float(p_rl_g)
    else:
        severity = abs(delta_p_g)
    return dict(
        category=cat,
        p_sft_g=p_sft_g, p_rl_g=p_rl_g, delta_p_g=delta_p_g,
        sft_top=sft_top, rl_top=rl_top,
        sft_in_topK=sft_in_topK, rl_in_topK=rl_in_topK,
        rank_sft=rank_sft, rank_rl=rank_rl,
        delta_p_w=float(delta_p_w),
        severity=float(severity),
    )


def categorize_all(sft_state, rl_state, eps_tail: float = 0.05,
                     top_k: int = 1) -> pd.DataFrame:
    """Apply classify_state to every (puzzle) row; one row per state in rl_state."""
    sft_lookup = sft_state.set_index("puzzle_id")
    rows = []
    for _, r in rl_state.iterrows():
        pid = r["puzzle_id"]
        if pid not in sft_lookup.index: continue
        s = sft_lookup.loc[pid]
        c = classify_state(s, r, eps_tail=eps_tail, top_k=top_k)
        rows.append({
            "puzzle_id": pid, "data_source": r["data_source"], "gt": r["gt"],
            "max_outcome_rl": r["max_outcome"], "mean_outcome_rl": r["mean_outcome"],
            "n_correct_rl": r["n_correct"], "n_traces_rl": r["n_traces"],
            **c,
        })
    return pd.DataFrame(rows)


# ── Pass@k ───────────────────────────────────────────────────────────────────

def pass_at_k_unbiased(n_correct: int, n_traces: int, k: int) -> float:
    """Unbiased pass@k from Codex paper.  pass@k = 1 - C(n - n_correct, k) / C(n, k)."""
    if n_traces - n_correct < k:
        return 1.0
    if k > n_traces:
        k = n_traces
    # 1 - prod_{i=0..k-1} (n - c - i) / (n - i)
    p = 1.0
    for i in range(k):
        p *= (n_traces - n_correct - i) / max(1, (n_traces - i))
    return float(1.0 - p)


def pass_at_k_for_states(state_with_outcomes: pd.DataFrame, k_list=(1, 8, 16)) -> dict:
    """state_with_outcomes must have n_correct, n_traces columns."""
    out = {}
    for k in k_list:
        vals = [pass_at_k_unbiased(int(r["n_correct_rl"]), int(r["n_traces_rl"]), k)
                 for _, r in state_with_outcomes.iterrows()]
        out[f"pass@{k}"] = float(np.mean(vals)) if vals else np.nan
    return out


# ── Plotting ────────────────────────────────────────────────────────────────

def render_board_png_bytes(fen: str, gt_uci: str | None = None,
                              wrong_uci: str | None = None, other_uci: str | None = None,
                              size: int = 320) -> bytes:
    """Render a chess board with up to three move arrows as PNG bytes.

    GT in green, wrong move in red, other (e.g. SFT modal != GT) in gray.
    """
    import chess, chess.svg
    import cairosvg
    board = chess.Board(fen)
    arrows = []
    if gt_uci:
        try:
            m = chess.Move.from_uci(gt_uci)
            arrows.append(chess.svg.Arrow(m.from_square, m.to_square, color="#00aa00"))
        except Exception: pass
    if wrong_uci and wrong_uci != gt_uci:
        try:
            m = chess.Move.from_uci(wrong_uci)
            arrows.append(chess.svg.Arrow(m.from_square, m.to_square, color="#cc0000"))
        except Exception: pass
    if other_uci and other_uci not in (gt_uci, wrong_uci):
        try:
            m = chess.Move.from_uci(other_uci)
            arrows.append(chess.svg.Arrow(m.from_square, m.to_square, color="#888888"))
        except Exception: pass
    svg = chess.svg.board(board=board, arrows=arrows, size=size)
    return cairosvg.svg2png(bytestring=svg.encode("utf-8"),
                              output_width=size*2, output_height=size*2)


def plot_state_with_board(meta_row, sft_row, rl_row, alpha_star, top_n=8,
                            title="", out_path=None):
    """Two-panel figure: board (left) with arrows; per-move bar plot (right).

    GT is the green arrow on the board and the red-edged bar in the chart.
    The RL modal (if not GT) is drawn as a red arrow on the board.
    """
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import io
    from PIL import Image

    all_u = list(dict.fromkeys(list(sft_row["all_u"]) + list(rl_row["all_u"])))
    idx = {u: i for i, u in enumerate(all_u)}
    def expand(src_u, src_v):
        v = np.zeros(len(all_u))
        for u, p in zip(src_u, src_v):
            v[idx[u]] = p
        return v
    p_sft = expand(sft_row["all_u"], sft_row["marg"])
    p_rl  = expand(rl_row["all_u"],  rl_row["marg"])
    p_a   = alpha_power(p_sft, alpha_star)

    gt = sft_row["gt"]
    score = np.maximum(np.maximum(p_sft, p_a), p_rl)
    top_idx = np.argsort(-score)[:top_n]
    if gt in idx and idx[gt] not in top_idx:
        top_idx = np.append(top_idx, idx[gt])
    moves = [all_u[i] for i in top_idx]
    p_sft_t = p_sft[top_idx]; p_a_t = p_a[top_idx]; p_rl_t = p_rl[top_idx]

    rl_modal = rl_row["modal"]
    sft_modal = sft_row["modal"]
    wrong_arrow = rl_modal if rl_modal != gt else (sft_modal if sft_modal != gt else None)

    fen = sft_row.get("fen") or rl_row.get("fen")
    fig = plt.figure(figsize=(11, 4.5))
    if fen:
        ax_board = fig.add_axes([0.02, 0.05, 0.32, 0.85])
        try:
            png_bytes = render_board_png_bytes(fen, gt_uci=gt, wrong_uci=wrong_arrow, size=320)
            img = Image.open(io.BytesIO(png_bytes))
            ax_board.imshow(img); ax_board.axis("off")
            ax_board.set_title(f"GT={gt} (green)" + (f"  RL_modal={wrong_arrow} (red)" if wrong_arrow else ""),
                                fontsize=10)
        except Exception as e:
            ax_board.text(0.5, 0.5, f"(board render failed: {e})",
                           ha="center", va="center", transform=ax_board.transAxes)
            ax_board.axis("off")
        ax_bar = fig.add_axes([0.40, 0.18, 0.56, 0.70])
    else:
        ax_bar = fig.add_axes([0.10, 0.18, 0.85, 0.70])

    x = np.arange(len(moves)); w = 0.27
    bars1 = ax_bar.bar(x - w, p_sft_t, w, label=r"$\pi_{sft}$",
                        color="#4c72b0", edgecolor="black")
    bars2 = ax_bar.bar(x,     p_a_t,   w,
                        label=fr"$\pi_{{\alpha^\star}}\;(\alpha^\star={alpha_star:.2f})$",
                        color="#dd8452", edgecolor="black")
    bars3 = ax_bar.bar(x + w, p_rl_t,  w, label=r"$\pi_{rl}$",
                        color="#55a868", edgecolor="black")
    for bs in (bars1, bars2, bars3):
        for j, m in enumerate(moves):
            if m == gt:
                bs[j].set_edgecolor("red"); bs[j].set_linewidth(2.5)
    ax_bar.set_xticks(x); ax_bar.set_xticklabels(moves, rotation=45, ha="right", fontsize=9)
    ax_bar.set_ylabel("probability"); ax_bar.legend(fontsize=9, loc="upper right")
    if title:
        fig.suptitle(title, fontsize=10, y=0.99)
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
    return fig


# Back-compat alias
plot_distribution_triplet = plot_state_with_board


def plot_state_multi_step(pid: str, pre_state: pd.DataFrame, sft_state: pd.DataFrame,
                            rl_states_dict: dict, alpha_stars_dict: dict,
                            top_n: int = 8, title_suffix: str = "",
                            out_path: Path | None = None,
                            use_unnorm: bool = False):
    """One figure per puzzle: board + N_step bar-plot panels (π_pre, π_sft, π_α*, π_rl).

    If `use_unnorm=True`, plot the raw `exp(log_pi_tilde)` values instead of the
    legal-move-renormalized softmax — these do **not** sum to 1; they show the
    teacher-forced model probability of generating each move's token sequence.
    The π_α* bar is omitted in unnormalized mode (the α-power family is defined
    on a normalized distribution).

    Args
    ----
    pid : puzzle_id
    pre_state, sft_state : per-puzzle state DataFrames (from build_state)
    rl_states_dict       : {rl_step: state_df}, sorted ascending in the figure
    alpha_stars_dict     : {rl_step: α*}  (KL-optimal SFT→RL_step coefficient)
    top_n                : top moves to display (the GT is always added if missing)

    Each panel title shows the cumulative probability mass the displayed moves
    account for under each distribution.  GT bar gets a red edge across all panels.
    """
    import matplotlib.pyplot as plt
    import io
    from PIL import Image

    sft_lookup = sft_state.set_index("puzzle_id")
    pre_lookup = pre_state.set_index("puzzle_id") if pre_state is not None else None
    if pid not in sft_lookup.index: return None
    sft_row = sft_lookup.loc[pid]
    pre_row = pre_lookup.loc[pid] if (pre_lookup is not None and pid in pre_lookup.index) else None

    rl_rows = {}
    for s, st in sorted(rl_states_dict.items()):
        if pid in st["puzzle_id"].values:
            rl_rows[s] = st.set_index("puzzle_id").loc[pid]
    if not rl_rows: return None

    # Joint vocab (union of all distributions we'll plot)
    all_u = list(dict.fromkeys(list(sft_row["all_u"])
                                + (list(pre_row["all_u"]) if pre_row is not None else [])
                                + [u for r in rl_rows.values() for u in r["all_u"]]))
    idx_map = {u: i for i, u in enumerate(all_u)}

    def expand(src_u, src_v):
        v = np.zeros(len(all_u))
        for u, p in zip(src_u, src_v): v[idx_map[u]] = p
        return v

    marg_field = "marg_unnorm" if use_unnorm else "marg"
    p_pre = expand(pre_row["all_u"], pre_row[marg_field]) if pre_row is not None else None
    p_sft = expand(sft_row["all_u"], sft_row[marg_field])
    p_rls = {s: expand(r["all_u"], r[marg_field]) for s, r in rl_rows.items()}

    # Choose top-N moves by max across every distribution we'll show
    p_max = p_sft.copy()
    if p_pre is not None:
        p_max = np.maximum(p_max, p_pre)
    for v in p_rls.values():
        p_max = np.maximum(p_max, v)
    top_idx = list(np.argsort(-p_max)[:top_n])
    gt = sft_row["gt"]
    if gt in idx_map and idx_map[gt] not in top_idx:
        top_idx.append(idx_map[gt])
    moves = [all_u[i] for i in top_idx]
    top_idx = np.array(top_idx)

    # Coverage = sum of probabilities on the displayed moves (constant across steps for pre/sft)
    def cov(p): return float(p[top_idx].sum()) if p is not None else None
    cov_pre = cov(p_pre)
    cov_sft = cov(p_sft)

    fen = sft_row.get("fen") if hasattr(sft_row, "get") else sft_row["fen"]
    if not fen and pre_row is not None:
        fen = pre_row.get("fen") if hasattr(pre_row, "get") else pre_row["fen"]

    n_steps = len(rl_rows)
    fig = plt.figure(figsize=(4 + 4.2*n_steps, 4.4))
    gs = fig.add_gridspec(1, n_steps + 1, width_ratios=[1.0] + [1.2]*n_steps, wspace=0.32)

    # Board (left)
    ax_board = fig.add_subplot(gs[0, 0])
    rl_last = rl_rows[max(rl_rows.keys())]
    wrong_arrow = rl_last["modal"] if rl_last["modal"] != gt else None
    if fen:
        try:
            png_bytes = render_board_png_bytes(fen, gt_uci=gt, wrong_uci=wrong_arrow, size=320)
            img = Image.open(io.BytesIO(png_bytes))
            ax_board.imshow(img); ax_board.axis("off")
            ax_board.set_title(
                f"GT = {gt}  (green)" + (f"\nRL modal = {wrong_arrow}  (red)" if wrong_arrow else ""),
                fontsize=11, pad=6)
        except Exception as e:
            ax_board.text(0.5, 0.5, f"(board render failed: {e})",
                           ha="center", va="center", transform=ax_board.transAxes, fontsize=9)
            ax_board.axis("off")
    else:
        ax_board.text(0.5, 0.5, "(no FEN)", ha="center", va="center",
                       transform=ax_board.transAxes); ax_board.axis("off")

    # Per-step panels (right)
    x = np.arange(len(moves))
    if use_unnorm:
        w = 0.27
    else:
        w = 0.20
    # For unnormalized mode, find a global y-max across the panels to share scale
    y_max = 1.0
    if use_unnorm:
        cands = [p_sft[top_idx].max()]
        if p_pre is not None: cands.append(p_pre[top_idx].max())
        cands.extend([v[top_idx].max() for v in p_rls.values()])
        y_max = float(max(cands)) * 1.10 if cands else 1.0
    for j, s in enumerate(sorted(rl_rows.keys())):
        ax = fig.add_subplot(gs[0, j + 1])
        a_star = float(alpha_stars_dict.get(s, 1.0))
        p_rl = p_rls[s]

        if use_unnorm:
            # 3 bars: π_pre, π_sft, π_rl  (no α* — undefined for unnormalized)
            if p_pre is not None:
                bars0 = ax.bar(x - w, p_pre[top_idx], w, label=r"$\exp(\log\tilde\pi_{pre})$",
                                color="#7f7f7f", edgecolor="black")
            bars1 = ax.bar(x,     p_sft[top_idx], w, label=r"$\exp(\log\tilde\pi_{sft})$",
                            color="#4c72b0", edgecolor="black")
            bars3 = ax.bar(x + w, p_rl[top_idx],  w, label=r"$\exp(\log\tilde\pi_{rl})$",
                            color="#55a868", edgecolor="black")
            bars_all = ((bars0,) if p_pre is not None else ()) + (bars1, bars3)
        else:
            p_a = alpha_power(p_sft, a_star)
            if p_pre is not None:
                bars0 = ax.bar(x - 1.5*w, p_pre[top_idx], w, label=r"$\pi_{pre}$",
                                color="#7f7f7f", edgecolor="black")
            bars1 = ax.bar(x - 0.5*w, p_sft[top_idx], w, label=r"$\pi_{sft}$",
                            color="#4c72b0", edgecolor="black")
            bars2 = ax.bar(x + 0.5*w, p_a[top_idx],   w,
                            label=fr"$\pi_{{\alpha^\star}}$",
                            color="#dd8452", edgecolor="black")
            bars3 = ax.bar(x + 1.5*w, p_rl[top_idx],  w, label=r"$\pi_{rl}$",
                            color="#55a868", edgecolor="black")
            bars_all = ((bars0,) if p_pre is not None else ()) + (bars1, bars2, bars3)
        for bs in bars_all:
            for k, m in enumerate(moves):
                if m == gt:
                    bs[k].set_edgecolor("red"); bs[k].set_linewidth(2)

        ax.set_xticks(x); ax.set_xticklabels(moves, rotation=45, ha="right", fontsize=12)
        ax.tick_params(axis="y", labelsize=10)
        ax.set_ylim(0, 1.05 if not use_unnorm else y_max)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4, linewidth=0.6)
        cov_rl_s = float(p_rl[top_idx].sum())
        if use_unnorm:
            cov_str = ("Σ on shown moves: "
                        + (f"pre {cov_pre:.2f}  " if cov_pre is not None else "")
                        + f"sft {cov_sft:.2f}  rl {cov_rl_s:.2f}")
        else:
            cov_str = ("cov: "
                        + (f"pre {cov_pre:.0%}  " if cov_pre is not None else "")
                        + f"sft {cov_sft:.0%}  rl {cov_rl_s:.0%}")
        n_legal = len(all_u)
        title_alpha = "" if use_unnorm else f"   $\\alpha^\\star$ = {a_star:.2f}"
        ax.set_title(f"step {s}{title_alpha}    {len(moves)}/{n_legal} moves\n{cov_str}",
                      fontsize=9.5, pad=4)
        if j == 0:
            ax.set_ylabel("unnormalized prob" if use_unnorm else "probability",
                            fontsize=12)
            ax.legend(loc="upper right", fontsize=10, frameon=True,
                       framealpha=0.92, edgecolor="#cccccc")

    if title_suffix:
        fig.suptitle(title_suffix, fontsize=10, y=1.04)
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_state_three_panel(pid: str, pre_state: pd.DataFrame, sft_state: pd.DataFrame,
                              rl_states_dict: dict,
                              top_n: int = 8, title_suffix: str = "",
                              out_path: Path | None = None):
    """4-panel paper-style figure for one puzzle:
        [board] [π_pretrain] [π_sft] [π_rl-by-step]

    The pretrain and SFT panels each show one bar per top move.  The RL panel
    shows grouped bars per move, one bar per RL step, coloured by a viridis
    gradient (light = early, dark = late).  GT bar gets a red edge in all panels.
    """
    import matplotlib.pyplot as plt
    import io
    from PIL import Image

    sft_lookup = sft_state.set_index("puzzle_id")
    pre_lookup = pre_state.set_index("puzzle_id") if pre_state is not None else None
    if pid not in sft_lookup.index:
        return None
    sft_row = sft_lookup.loc[pid]
    pre_row = pre_lookup.loc[pid] if (pre_lookup is not None and pid in pre_lookup.index) else None
    rl_rows = {s: st.set_index("puzzle_id").loc[pid]
                for s, st in sorted(rl_states_dict.items())
                if pid in st["puzzle_id"].values}
    if not rl_rows:
        return None

    all_u = list(dict.fromkeys(list(sft_row["all_u"])
                                + (list(pre_row["all_u"]) if pre_row is not None else [])
                                + [u for r in rl_rows.values() for u in r["all_u"]]))
    idx_map = {u: i for i, u in enumerate(all_u)}

    def expand(src_u, src_v):
        v = np.zeros(len(all_u))
        for u, p in zip(src_u, src_v):
            v[idx_map[u]] = p
        return v

    p_pre = expand(pre_row["all_u"], pre_row["marg"]) if pre_row is not None else None
    p_sft = expand(sft_row["all_u"], sft_row["marg"])
    p_rls = {s: expand(r["all_u"], r["marg"]) for s, r in rl_rows.items()}

    p_max = p_sft.copy()
    if p_pre is not None:
        p_max = np.maximum(p_max, p_pre)
    for v in p_rls.values():
        p_max = np.maximum(p_max, v)
    top_idx = list(np.argsort(-p_max)[:top_n])
    gt = sft_row["gt"]
    if gt in idx_map and idx_map[gt] not in top_idx:
        top_idx.append(idx_map[gt])
    moves = [all_u[i] for i in top_idx]
    top_idx = np.array(top_idx)
    n = len(moves)

    fen = sft_row.get("fen") if hasattr(sft_row, "get") else sft_row["fen"]
    if not fen and pre_row is not None:
        fen = pre_row.get("fen") if hasattr(pre_row, "get") else pre_row["fen"]
    rl_last = rl_rows[max(rl_rows.keys())]
    wrong_arrow = rl_last["modal"] if rl_last["modal"] != gt else None

    fig = plt.figure(figsize=(16.5, 3.4))
    gs = fig.add_gridspec(1, 4, width_ratios=[0.85, 1.0, 1.0, 1.4], wspace=0.32)

    # Panel 1: board
    ax_board = fig.add_subplot(gs[0, 0])
    if fen:
        try:
            png_bytes = render_board_png_bytes(fen, gt_uci=gt, wrong_uci=wrong_arrow, size=320)
            img = Image.open(io.BytesIO(png_bytes))
            ax_board.imshow(img); ax_board.axis("off")
            ttl = f"GT = {gt}  (green)"
            if wrong_arrow:
                ttl += f"\nRL modal = {wrong_arrow}  (red)"
            ax_board.set_title(ttl, fontsize=12, pad=4)
        except Exception as e:
            ax_board.text(0.5, 0.5, f"(board render failed)", ha="center", va="center",
                           transform=ax_board.transAxes); ax_board.axis("off")
    else:
        ax_board.axis("off")

    x = np.arange(n)
    y_max = float(max(p_sft[top_idx].max() if n else 0,
                       p_pre[top_idx].max() if (p_pre is not None and n) else 0,
                       max((v[top_idx].max() for v in p_rls.values()), default=0))) * 1.10
    y_max = max(y_max, 0.05)

    def _hl_gt(bars):
        for k, m in enumerate(moves):
            if m == gt:
                bars[k].set_edgecolor("red"); bars[k].set_linewidth(2)

    # Panel 2: Pretrain
    ax_pre = fig.add_subplot(gs[0, 1])
    if p_pre is not None:
        bars_pre = ax_pre.bar(x, p_pre[top_idx], 0.7, color="#7f7f7f", edgecolor="black")
        _hl_gt(bars_pre)
    ax_pre.set_xticks(x); ax_pre.set_xticklabels(moves, rotation=45, ha="right", fontsize=12)
    ax_pre.set_ylim(0, y_max)
    ax_pre.set_title("Pretrain", fontsize=14, fontweight="bold", pad=4)
    ax_pre.set_ylabel("probability", fontsize=12)
    ax_pre.grid(True, axis="y", linestyle="--", alpha=0.4, linewidth=0.6)

    # Panel 3: SFT
    ax_sft = fig.add_subplot(gs[0, 2], sharey=ax_pre)
    bars_sft = ax_sft.bar(x, p_sft[top_idx], 0.7, color="#4c72b0", edgecolor="black")
    _hl_gt(bars_sft)
    ax_sft.set_xticks(x); ax_sft.set_xticklabels(moves, rotation=45, ha="right", fontsize=12)
    ax_sft.set_title("SFT", fontsize=14, fontweight="bold", pad=4)
    ax_sft.tick_params(axis="y", labelleft=False)
    ax_sft.grid(True, axis="y", linestyle="--", alpha=0.4, linewidth=0.6)

    # Panel 4: RL by step (grouped bars, viridis gradient)
    ax_rl = fig.add_subplot(gs[0, 3], sharey=ax_pre)
    steps_sorted = sorted(p_rls.keys())
    n_steps = len(steps_sorted)
    cmap_steps = plt.cm.viridis(np.linspace(0.18, 0.92, max(n_steps, 1)))
    bw = min(0.72 / n_steps, 0.18)
    for i, s in enumerate(steps_sorted):
        offset = (i - (n_steps - 1) / 2) * bw
        bars_rl = ax_rl.bar(x + offset, p_rls[s][top_idx], bw,
                              color=cmap_steps[i], edgecolor="black",
                              linewidth=0.5, label=f"step {s}")
        _hl_gt(bars_rl)
    ax_rl.set_xticks(x); ax_rl.set_xticklabels(moves, rotation=45, ha="right", fontsize=12)
    ax_rl.set_title("RL", fontsize=14, fontweight="bold", pad=4)
    ax_rl.tick_params(axis="y", labelleft=False)
    ax_rl.grid(True, axis="y", linestyle="--", alpha=0.4, linewidth=0.6)
    ax_rl.legend(loc="upper right", fontsize=10, frameon=True,
                  framealpha=0.92, edgecolor="#cccccc", ncol=1)

    if title_suffix:
        fig.suptitle(title_suffix, fontsize=11, y=1.04)
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=170, bbox_inches="tight")
        plt.close(fig)
    return fig


# ── Misc ─────────────────────────────────────────────────────────────────────

def parse_cfg(cfg: str) -> dict:
    parts = cfg.split("_")
    return dict(compute_total=parts[0], size_str=parts[1],
                alpha=parts[2].replace("alpha", ""),
                beta=parts[3].replace("beta", ""))
