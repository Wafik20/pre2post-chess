# CoT Search-Tree Analysis

A small, self-contained tool that parses chain-of-thought rollouts from a
chess reasoning model into a **prefix tree of moves** and computes a set of
search-shape, search-behavior, and Stockfish-graded quality metrics over RL
training. Designed for transparency: every metric is computed in one short
function whose docstring states exactly what it measures.

```
cot_analysis/
├── tree.py        Prefix-tree parser (≈100 LOC)
├── metrics.py     Shape, behavior, and length metrics (≈140 LOC)
├── quality.py     Stockfish-based candidate / opponent / commit quality (≈260 LOC)
├── analyze.py     Pipeline: JSONL → metrics → parquet
├── plot.py        Parquet → figure
├── run.sh         One-command entry point
└── README.md      (this file)
```

## What a "rollout" looks like

The model emits a chain-of-thought between `<T>` and `</T>` tags, where
candidate move lines are separated by `<sep>`. A toy example:

```
Input:  Pe2e4 Pe7e5 Ng1f3 Nb8c6 Bf1b5 Pa7a6 <T>
Output: <T> Bb5xc6 Pb7xc6 <sep> Bb5xc6 Pd7xc6 <sep> Bb5a4 </T> Bb5xc6
```

Parsing produces the following prefix tree (root = current position):

```
            root
          /      \
       Bb5xc6    Bb5a4          ← depth-1 candidates
      /     \
   Pb7xc6  Pd7xc6                ← depth-2 imagined opponent replies
```

The committed move is `Bb5xc6`. Two candidate lines share the prefix
`Bb5xc6` and revisit that node when written.

## Metrics

All metrics are computed per rollout, then averaged across the 16 rollouts
per prompt, then averaged across prompts at each RL step.

### Search shape (purely structural — no engine)

| Metric | Definition |
|---|---|
| `candidate_count` | Number of distinct depth-1 root candidates. |
| `num_nodes` | Distinct move-prefixes in the tree (excluding the root). |
| `max_depth` (D) | Depth of the deepest line. |
| `effective_branching` | Geometric-mean fanout `num_nodes^(1/D)`. |
| `width_depth_index` | `#leaves / D` — high = wide-shallow, low = narrow-deep. |

### Search behavior (over the temporal write-order)

| Metric | Definition |
|---|---|
| `revisit_rate` | Fraction of move-writes that re-enter a node the model has already written. Counts shared prefixes between candidate lines. A node = a unique move-prefix from root, *not* the bare move-string. |
| `dfs_consistency` (τ) | Kendall's τ between the model's first-visit order and the canonical depth-first preorder of the same tree. τ=1 means each subtree is finished before the next is opened (pure DFS); τ=0 means maximally interleaved. |

### Reasoning length

| Metric | Definition |
|---|---|
| `reasoning_char_count` | Number of characters between `<T>` and `</T>`. |
| `n_ordered_visits` | Total move tokens written into the tree (sum of all candidate-line lengths). |

### Move quality (Stockfish-graded)

For each board position, Stockfish ranks **every legal move** in a multipv
search at fixed depth. We then normalize:

```
norm_rank(move) = (rank − 1) / (n_legal − 1)   ∈ [0, 1]
```

where rank 1 = Stockfish's top move and rank `n_legal` = worst. **Lower
norm_rank = better.** Illegal moves are excluded from the means below (they
are reported separately as illegal-move counts).

| Metric | Definition |
|---|---|
| `mean_candidate_norm_rank` | Mean rank over legal depth-1 (player) moves. |
| `best_candidate_norm_rank` | Min rank over legal depth-1 moves — the best alternative the model considered. |
| `mean_opponent_norm_rank` | Mean rank over legal depth-2 (imagined opponent) moves. Lower = realistic opponent modeling. |
| `best_opponent_norm_rank` | Min over legal depth-2 moves. |
| `candidate_illegal_rate` | Fraction of depth-1 moves that aren't legal at the root. |
| `opponent_illegal_rate` | Same at depth 2. |
| `selected_norm_rank_self` | Stockfish rank of the move the model committed to. |
| `best_seen_norm_rank_self` | Best legal root candidate the model considered. |
| `selection_rank_gap_self` | `selected − best_seen`. >0 means the model considered a better move but didn't pick it. |
| `commit_quality` | Composite z-score: `−z(selected) − z(rank_gap)`. Higher = better committed-move selection relative to the run-wide distribution. |

### Target presence (string-match against ground truth)

| Metric | Definition |
|---|---|
| `target_in_root_candidates` | The ground-truth first move appears as one of the depth-1 candidates. |
| `target_in_any_depth` | The ground-truth UCI appears as a move label anywhere in the tree (loose — does not require correct board state). |

## Usage

### 1. Layout

Place your eval JSONLs under `checkpoints/` like this:

```
checkpoints/
├── sft/<run_tag>/eval_results/step_*/generations/0.jsonl       (= step 0)
└── rl/<run_tag>/eval_results_2k_t1_n16/step_K_*/generations/0.jsonl
```

Where `<run_tag>` is any string (e.g. `C6p5e18_20m_alpha0.200_beta0.008`)
and `K` is the RL training step. Multiple RL step folders are picked up
automatically.

Each JSONL row is one rollout with at least the following fields:

```jsonc
{
  "input":           "Pe2e4 Pe7e5 ... <T>",      // game prefix + delimiter
  "output":          "<T> Bb5xc6 ... </T> Bb5xc6",  // CoT + chosen move
  "data_source":     "test_B1",                   // or test_B2..B5
  "extracted_moves": ["Bb5xc6"],                  // committed move(s)
  "target_moves":    "Bb5xc6,Pb7xc6",             // ground-truth continuation
  "score":           1.0,                         // reward
  "outcome_score":   1.0                          // outcome-only reward
}
```

Only rows with `data_source ∈ {test_B1, …, test_B5}` are kept.

### 2. Run

```bash
# Required: a Stockfish binary somewhere on disk.
export STOCKFISH=/path/to/stockfish

# Optional knobs:
export STOCKFISH_DEPTH=8
export N_WORKERS=8

# Optional layout overrides:
export CHECKPOINTS=./checkpoints
export RUN_TAG=C6p5e18_20m_alpha0.200_beta0.008
export OUT=./results/$RUN_TAG

bash run.sh
```

This produces:

```
results/<run_tag>/
├── rollout_df.parquet  one row per rollout, all raw metrics
├── prompt_df.parquet   one row per (step, prompt), means + commit_quality z-score
├── fig_main.png/.pdf   the 11-metric figure
```

### 3. Compare runs

`plot.py` accepts multiple `--data` directories to overlay multiple runs:

```bash
python plot.py \
    --data ../results/run_20m  ../results/run_50m \
    --tags 20m 50m \
    --out  ../results/compare \
    --grid 3 4
```

Colors are auto-assigned along the viridis colormap by the numeric model
size parsed from the run tag.

## Validation

`verify.py` cross-checks an output `prompt_df.parquet` against a reference
one (per-step mean diffs + per-prompt agreement on the (step, input) join):

```bash
python verify.py --new <our_prompt_df.parquet> --reference <ref_prompt_df.parquet>
```

On the canonical `C6p5e18_20m_alpha0.200_beta0.008` run (SFT step + RL steps
100, 500, 1000):

| Metric family | Result |
|---|---|
| Tree shape and behavior (`num_nodes`, `max_depth`, `width_depth_index`, `revisit_rate`, `dfs_consistency`, target presence) | **exact match** for RL steps; ≤ 0.001 diff at step 0 |
| Stockfish quality means (`mean_candidate_norm_rank`, `selected_norm_rank_self`, etc.) | per-prompt correlation 0.75–0.92, mean abs diff ≤ 0.06 |
| Illegal-move rates | **systematically lower** in this tool — by design, see below |

**Why illegal rates differ:** the reference implementation subsamples the
Stockfish FEN cache (only ~10 % of unique FENs get ranked) to save
compute, then silently *drops* "legal but cache-missing" moves from the
denominator. This tool ranks **every** relevant FEN, so it surfaces more
legal moves and reports a more accurate (lower) illegal rate. The two
implementations agree on absolute illegal counts; they disagree on what
fraction of legal candidates the cache covered.

## Performance notes

- Stockfish is the only expensive step. We collect every unique FEN
  needed for candidate (depth-0 root) and opponent (depth-1 child) ranking
  across all rollouts, then run multipv-at-fixed-depth once per FEN in
  parallel processes.
- With `N_WORKERS=16` and `STOCKFISH_DEPTH=8`, the canonical 20m run
  (≈4 steps × 1500 prompts × 16 rollouts) takes ~10–20 min.
- All non-Stockfish metrics are deterministic and trivially fast.

## Extending

- **Add a new metric**: write a function in `metrics.py` (or `quality.py`
  if it needs a position cache), then add it to
  `compute_metrics_for_row` in `analyze.py` and the `PANELS` list in
  `plot.py`.
- **Add a new figure**: drop a function into `plot.py` that consumes
  `prompt_df.parquet`; the format is small and self-describing.
- **Different ground truth**: only `compute_target_presence` and
  `compute_commit_quality_components` depend on dataset-specific fields
  (`target_moves`, `extracted_moves`). Both are localized.

## License & dependencies

Pure Python. Dependencies:

```
python>=3.10
python-chess
pandas
numpy
pyarrow
matplotlib
```

Plus a Stockfish binary at `STOCKFISH` (any recent version).
