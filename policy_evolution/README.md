# Policy-Evolution Analysis

How does the *move distribution* of a chess reasoning model change as it goes
**pretrain → SFT → RL**? This tool answers that by teacher-forcing each
checkpoint over a fixed set of puzzles, measuring the probability it assigns to
**every legal move** at the decision point, and then computing a set of
distributional metrics (sharpening, trace sensitivity, trace collapse, entropy
decomposition, and per-state transition categories) across RL training.

Every metric is computed in one short, documented function in
[`policy_analysis_lib.py`](policy_analysis_lib.py); the math below states exactly
what each one measures.

```
policy_evolution/
├── measure_policy.py        Stage 1: (model + rollouts) → raw_scores.parquet   (GPU)
├── policy_analysis_lib.py   All metric + plotting primitives (one fn per metric)
├── run_policy_analysis.py   Stage 2: raw_scores → figures + CSVs               (CPU)
├── collect.sh               One-command Stage 1 (reads manifest.tsv)
├── analyze.sh               One-command Stage 2 (figures + tables)
├── requirements.txt
└── README.md                (this file)
```

## Pipeline

```
                128-rollout eval          teacher-forcing            distributional
   checkpoint ──────────────────►  ┌──────────────────────┐  ──►  ┌──────────────────┐
   + rollouts    (generation        │ Stage 1              │       │ Stage 2          │
                  0.jsonl)          │ measure_policy.py    │       │ run_policy_      │
                                    │  → raw_scores.parquet│       │ analysis.py      │
                                    └──────────────────────┘       │  → figures + CSV │
                                                                   └──────────────────┘
```

- **Stage 1 — data collection (`collect.sh` → `measure_policy.py`).**
  For each checkpoint and each puzzle in its 128-rollout eval, re-score (teacher
  force) **all legal moves** at the position the model has to play, recording an
  unnormalized log-probability per legal move, per rollout. Output: one
  `raw_scores.parquet` per `(stage, step)`. Needs a GPU.

- **Stage 2 — metrics + figures (`analyze.sh` → `run_policy_analysis.py`).**
  Reads only the parquet files, builds a per-puzzle "state" object, and computes
  every metric below. CPU-only and cached, so re-runs are fast.

### Where the 128 rollouts come from (upstream, not in this folder)

Stage 1 assumes each checkpoint has already been evaluated with **128 sampled
rollouts per puzzle** (temperature 1) over the held-out puzzle brackets
`test_B1 … test_B5`. That eval is the standard generation step of the training
repo and produces a `generation/0.jsonl` file whose rows look like:

```jsonc
{
  "input":         "Pe2e4 Pe7e5 ... <T>",   // game prefix + start-of-thought tag
  "output":        "<T> ... </T> Bb5xc6",    // chain-of-thought + committed move
  "data_source":   "test_B1",                // puzzle difficulty bracket
  "ground_truth":  "['e1g1', 'c6d4', ...]",  // ground-truth move sequence (UCI)
  "outcome_score": 1.0                        // reward for this rollout
}
```

`measure_policy.py` re-reads these rollouts and re-scores them under the model;
it does **not** sample anything itself, so the analysis is fully deterministic.
To regenerate the rollouts, run your training repo's evaluation with
`n_samples = 128` for the pretrain final, SFT final, and each RL step you want.

## Quick start (reproduces the paper figure pack)

```bash
pip install -r requirements.txt

# ── Stage 1: produce raw_scores.parquet for every checkpoint ─────────────
# Edit manifest.tsv to point at your checkpoints + 128-rollout evals, then:
PARQUET_DIR=/path/to/eval_parquets bash collect.sh

# ── Stage 2: metrics + figures ───────────────────────────────────────────
RESULTS_ROOT=./results TOP_K=3 EXAMPLE_BRACKET=test_B5 \
RL_STEPS="50 100 250 500 750" bash analyze.sh
```

`manifest.tsv` lists one checkpoint per line — see
[`collect.sh`](collect.sh) for the format. Both scripts are driven entirely by
environment variables (documented at the top of each file); no paths are
hard-coded.

### Data layout

```
results/<cfg>/                          # cfg = <total>_<size>_alpha<α>_beta<β>[_tag]
├── pretrain/step_final/raw_scores.parquet
├── sft/step_final/raw_scores.parquet
├── rl/step_50/raw_scores.parquet
├── rl/step_100/raw_scores.parquet
│   ...
└── policy_analysis/                    # Stage-2 output (figures, CSVs, cache)
```

---

## The "state" object (foundation of every metric)

For one puzzle, the model produces 128 rollouts (traces). Each trace is a chain
of thought followed by a committed move. We turn this into a per-puzzle **state**
([`build_state`](policy_analysis_lib.py)) as follows.

- For each trace *i*, teacher-force the model and read its log-prob over each
  legal move, then softmax over the legal moves to get a per-trace move
  distribution **πᵢ** on the position's legal-move set.
- Take the union of legal moves across traces as a common vocabulary, and embed
  every πᵢ on it.
- The **trace marginal** is the average distribution
  **π̄ = (1/N) Σᵢ πᵢ** — the move distribution you'd get by sampling a random
  rollout, then sampling a move from it.
- `g` denotes the ground-truth move; `π(g)`, its rank, and the modal (top-1)
  move are recorded for both per-trace and marginal distributions.

Notation used below: **π_pretrain / π_sft / π_rl** are the trace marginals at
each stage; **πᵢ** are individual traces; `g` is the ground-truth move.

## Metrics

### 1. Move-space sharpening — how much RL just "sharpens" SFT

We ask: can the RL move distribution be explained as a **temperature
sharpening** of the SFT distribution? Define the power transform

```
π_α(a) ∝ π_sft(a)^α          α = 1 → identity,  α > 1 → sharper,  α < 1 → flatter
```

and find the exponent that best reproduces RL, in KL
([`fit_alpha_kl`](policy_analysis_lib.py)):

```
α*  =  argmin_α  E_states  D_KL( π_rl ‖ π_α )
```

| Quantity | Definition | Figure / column |
|---|---|---|
| `alpha_star` (α*) | KL-optimal sharpening exponent (global and per-state). α*>1 ⇒ RL sharpens SFT. | `01_sharpening_alpha.png` |
| `residual_jsd` | Mean `JSD(π_rl, π_α*)` — how far RL still is from the best sharpened SFT. | `01b_…png` |
| `baseline_jsd` | Mean `JSD(π_rl, π_sft)` — distance with no sharpening (α=1). | — |
| `explained_sharp` | `1 − residual_jsd / baseline_jsd` ∈ (−∞, 1]. Fraction of the SFT→RL move shift explained purely by sharpening. ≈1 ⇒ RL ≈ a temperature change; ≪1 ⇒ RL moves probability mass in ways sharpening cannot. | `01b_…png` |
| `logit_slope` | Slope β of `centeredlog(π_rl) = β·centeredlog(π_sft)`, where `centeredlog(p)(a) = log p(a) − mean_b log p(b)`. For a pure power-sharpened distribution β = α* exactly with R²=1, so β and its R² are a regression-based cross-check on α*. | `01c_…png` |

The same fit is also applied to the **pretrain→SFT** transition (reported as
`rl_step = 0`) so the SFT-induced sharpening can be compared to the RL-induced
one. `sharpening_fits.csv` has one row per transition with global + per-state
(median / IQR) values.

### 2. Trace sensitivity — does the model "think differently" per rollout?

A model can reach the same marginal in two ways: every rollout already agrees
(low sensitivity), or rollouts disagree and only their average looks decisive.
**Trace sensitivity** measures the latter
([`trace_sensitivity_per_state`](policy_analysis_lib.py)):

```
TS(s)  =  (1/N) Σᵢ  D_KL( πᵢ ‖ π̄ )
```

the mean KL of each trace from the marginal. TS≈0 ⇒ all rollouts induce the same
move distribution; large TS ⇒ the chain-of-thought genuinely steers the move,
trace by trace. Pretrain (no real CoT) sits near 0 and is used as an anchor;
`02_trace_sensitivity.png` plots `TS_rl` across RL steps against the SFT and
pretrain anchors.

### 3. Trace collapse — is RL converging onto one reasoning path?

Does RL's marginal match the **average** SFT rollout, or does it look like a
**single** SFT rollout that RL has locked onto?
([`collapse_per_state`](policy_analysis_lib.py)):

```
D_marg(s)   =  JSD( π_rl , π̄_sft )                    distance to the SFT average
D_trace(s)  =  min_i  JSD( π_rl , πᵢ_sft )             distance to the nearest single SFT trace
C_trace(s)  =  D_marg(s) − D_trace(s)  ≥ 0             collapse gap
```

`C_trace > 0` means RL is **closer to one particular SFT rollout** than to the
SFT average — evidence that RL collapsed the rollout ensemble onto a single
mode rather than uniformly sharpening it. We also record whether that nearest
SFT trace's modal move equals the ground truth (`nearest_trace_modal_eq_g`),
i.e. whether the path RL collapsed onto was a *correct* one.
`03_rl_sft_trace_collapse.png` plots all three across RL steps.

### 4. Entropy: pretrain vs SFT (and the within-trace decomposition)

For each state we record the marginal entropy `H_marg = H(π̄)` and the mean
per-trace entropy `H̄_trace = (1/N) Σᵢ H(πᵢ)`. Their gap

```
TS_H(s)  =  H_marg(s) − H̄_trace(s)  ≥ 0
```

is the entropy-form of trace sensitivity: how much of the marginal's spread comes
from *disagreement between rollouts* (between-trace) versus *uncertainty within a
rollout* (within-trace). `04_entropy_pretrain_vs_sft.png` contrasts the pretrain
and SFT entropy distributions and this decomposition.

### 5. Per-state transition categories — *what kind* of update RL made

For each puzzle we classify the SFT→RL change of the ground-truth move into one
of the following ([`classify_state`](policy_analysis_lib.py)). Let `rank_θ(g)` be
the rank of the ground-truth move under stage θ's marginal, and let
`T_θ^k = {moves with rank ≤ k}` be the **top-K** set (k set by `--top-k`):

| Category | Condition | Meaning |
|---|---|---|
| `gt_amplification` | g ∈ T_sft^k and g ∈ T_rl^k, Δπ(g) > 0 | g was already top-K; RL pushed more mass onto it. |
| `tail_discovery` | g ∉ T_sft^k, g ∈ T_rl^k, **π_sft(g) < ε_tail** | RL promoted g into the top-K from the deep tail (SFT barely considered it). |
| `topk_correction` | g ∉ T_sft^k, g ∈ T_rl^k, **π_sft(g) ≥ ε_tail** | RL promoted g into the top-K, but SFT already gave it non-trivial mass. |
| `gt_regression` | g ∈ T_sft^k, g ∉ T_rl^k | RL demoted g out of the top-K (a regression). |
| `wrong_mode_amplification` | g ∉ T_sft^k, g ∉ T_rl^k, SFT's modal move w stays top-K under RL and Δπ(w) > 0 | g never made it; RL instead doubled down on a wrong move. |
| `other` | none of the above | — |

`--top-k 1` is the strict spec ("become the top-1 move"); `--top-k k` relaxes
"become top-1" to "reach the top-k". `ε_tail` (`--eps-tail`) is the SFT
probability below which a promotion counts as discovered from the *tail* rather
than merely *corrected*. Outputs:

- `per_state_categories.parquet` — one row per (puzzle, RL step) with category +
  the diagnostics that produced it (`π_sft(g)`, `π_rl(g)`, ranks, Δπ, severity).
- `05_category_summary.png`, `06*/07*_category_per_*.png` — category counts per
  RL step, split by puzzle bracket, and as fractions.
- `categories/<cat>/examples/…png` — board renders of the most extreme states per
  category (skip with `SKIP_CATEGORIES=true` / `--skip-categories`).

### 6. pass@k (per category)

Using the recorded per-rollout `outcome_score`, we report the unbiased
[`pass_at_k_unbiased`](policy_analysis_lib.py) estimator from the Codex paper:

```
pass@k  =  1 − C(N − c, k) / C(N, k)
```

for `N = 128` rollouts and `c` correct ones — i.e. the probability that at least
one of `k` sampled rollouts solves the puzzle. This lets the category panels show
not just *how many* states fall in each category but how *solvable* they are.

## Distributional primitives

All of the above are built from a handful of standard functions in
`policy_analysis_lib.py`, each restricted to the legal-move simplex:

- `kl(p, q)` — Kullback–Leibler divergence `Σ p log(p/q)`.
- `jsd(p, q)` — Jensen–Shannon divergence `½KL(p‖m) + ½KL(q‖m)`, `m = ½(p+q)`
  (symmetric, bounded — used wherever we need a distance).
- `entropy(p)` — Shannon entropy.
- `alpha_power(p, α)` — the power/temperature transform `p^α / Z`.

## Output figure pack

| File | Content |
|---|---|
| `01_sharpening_alpha.png` | KL-optimal α* across stages (pretrain→SFT→RL steps). |
| `01b_sharpening_metrics.png` | explained-sharpening and residual JSD. |
| `01c_sharpening_distributions.png` | per-state α* / slope distributions. |
| `02_trace_sensitivity.png` | TS_rl vs RL step, with SFT / pretrain anchors. |
| `03_rl_sft_trace_collapse.png` | D_marg, D_trace, C_trace, fraction-correct. |
| `04_entropy_pretrain_vs_sft.png` | pretrain ↔ SFT entropy + decomposition. |
| `05_category_summary.png` | per-state transition categories (3-panel summary). |
| `06_/06b_category_per_bin.png` | category counts / fractions per puzzle bracket. |
| `07_/07b_category_per_step.png` | category counts / fractions per RL step. |
| `sharpening_fits.csv` | one row per (stage, RL step). |
| `per_step_summary.csv` | one row per RL step (all summary scalars). |
| `per_state_categories.parquet` | one row per (puzzle, RL step). |

## Performance notes

- **Stage 1** is the expensive part (one GPU forward pass per legal move per
  rollout). Legal moves are scored in a single batched KV-cache pass per
  position, and rollouts that share a board state share the context pass.
- **Stage 2** is CPU-only. The per-puzzle state objects are pickled under
  `<out_dir>/cache/state_v3_<stage>_<step>.pkl`, so re-running with a different
  `--top-k` / `--eps-tail` reuses them and takes seconds instead of re-reading
  the multi-GB parquets.

## Dependencies

```
python>=3.10
torch                 # Stage 1 only
transformers          # Stage 1 only (loads the HF checkpoint via trust_remote_code)
python-chess
numpy
pandas
pyarrow
scipy
matplotlib
seaborn
```
