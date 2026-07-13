# Chess SFT: CoT generation + supervised fine-tuning

Self-contained pipeline for (1) generating chain-of-thought (CoT) training data
from chess puzzles via tree search, and (2) running multi-turn supervised
fine-tuning (SFT) on a pretrained checkpoint with that data.

The two stages are independent: you can plug in pre-generated CoT data and skip
straight to SFT, or generate fresh data first.

## Requirements

- Python 3.10+ with: `torch`, `transformers`, `accelerate`, `omegaconf`,
  `python-chess`, `numpy`, `pandas`, `tqdm`, `wandb` (optional, for logging).
- A [Stockfish](https://stockfishchess.org/) binary (used to label search trees).
- 1+ GPU for SFT and for model-policy CoT generation. Random / Stockfish-policy
  CoT generation is CPU-only.

### `${REPO_ROOT}` placeholder

Paths in this anonymized release use `${REPO_ROOT}` in place of an absolute home.
The shell scripts auto-detect `REPO_ROOT` (the `sft/` directory) and expand it.
A few **default** path strings inside the Python files (e.g. eval-data and
checkpoint defaults in `evaluation/` and `training/sft_trainer.py`) also contain
the literal `${REPO_ROOT}` — these are placeholders that are not used in the
normal workflow because real paths are supplied via the config YAML and CLI
flags. Override them there rather than editing the defaults.

---

## Step 1 — Generate CoT data

`cot_generation/scripts/generate_cot.sh` shards a puzzle CSV and runs
`puzzle_sequence_generator.py` on each shard, writing CoT examples to an output
directory. Three sampling modes are supported:

| Mode        | Policy                | Hardware  | Notes                                  |
|-------------|-----------------------|-----------|----------------------------------------|
| `stockfish` | Stockfish (default)   | CPU       | Difficulty-adaptive search budget      |
| `random`    | Random moves          | CPU       | Cheap baseline data                    |
| `model`     | HF policy checkpoint  | GPU       | Needs `MODEL_PATH` + `CONFIG_PATH`     |

```bash
cd cot_generation/scripts

# Stockfish-policy (default), pointing at your inputs:
DATA_PATH=/path/to/puzzles.csv \
OUTPUT_DIR=/path/to/cot_out \
STOCKFISH_PATH=/path/to/stockfish \
NUM_SF_WORKERS=12 \
bash generate_cot.sh stockfish

# Model-policy (uses a pretrained checkpoint as the move policy):
MODEL_PATH=/path/to/checkpoint/hf_model \
CONFIG_PATH=/path/to/checkpoint/config_snapshot/config.yaml \
DATA_PATH=/path/to/games.csv \
OUTPUT_DIR=/path/to/cot_out \
bash generate_cot.sh model
```

Sharding is controlled by `TOTAL_SAMPLES`, `SAMPLES_PER_JOB`, `START_OFFSET`
(env vars). Per-shard logs land in `cot_generation/logs/`. The contents of
`OUTPUT_DIR` become the `train_files` directory for SFT in Step 2.

---

## Step 2 — Run SFT

`run_sft.sh` loads a pretrained checkpoint and fine-tunes it on the generated
CoT data via `accelerate launch scripts/train/run_sft.py`.

```bash
PRETRAIN_ROOT=/path/to/checkpoints/C_6p5e19 \
TRAIN_DATA_DIR=/path/to/cot_out \
NUM_GPUS=2 \
WANDB_ENTITY=my-team \
bash run_sft.sh
```

What it does:

- Resolves the pretrain checkpoint from the spec
  `total_compute|modelsize|alpha|beta` (default `6p5e19|680m|0.750|0.030`) at
  `${PRETRAIN_ROOT}/{total_compute}_{modelsize}_alpha{alpha}/final/`.
- Looks up the cap on training files for that spec in
  `sft_max_train_files_map.json` (falls back to `default_max_train_files`).
- Trains with the config at
  `config/configs/qwen_multiturn_sft/sft_config_multi_path_2048.yaml`. Edit that
  file for batch size, epochs, optimizer/scheduler, block size, tokenizer, and
  the W&B project. Set the `entity:` field (or pass `WANDB_ENTITY`) to your own
  W&B account.

To train a different spec, edit the `pretrain_specs` array in `run_sft.sh` and
add a matching entry to `sft_max_train_files_map.json`.

### Useful `run_sft.py` flags

`run_sft.sh` wires these for you, but they can be passed directly:

- `--pretrained-model` / `--pretrained-weights` — checkpoint to fine-tune
- `--train-files` / `--data-name` / `--max-train-files` — training data
- `--cot-field` / `--cot-type` — which CoT field of the data to train on
- `--block-size`, `--lr` — sequence length / learning-rate overrides
- `--eval-only [--eval-pass-at-k]` — evaluate an existing checkpoint
