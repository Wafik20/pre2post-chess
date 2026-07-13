#!/usr/bin/env python3
"""Tokenize a cleaned parquet into .npy token shards for pretraining.

Two steps:
  1. stream the parquet's text column into size-bounded .txt shards, then
  2. tokenize each shard with the chess tokenizer and save uint32 .npy arrays.

The pretraining data loader (`pretraining/`) reads exactly this directory of
.npy shards (it holds out the last few as validation).

Usage:
    python tokenize_dataset.py \
        --parquet   /path/to/clean_games            # file or directory
        --out-dir   /path/to/pretrain_tokenized \
        --column    ctx \
        --tokenizer LanTokenizer
"""
import argparse
import pathlib
import sys

import numpy as np

# Make the shared llm_tokens/ package (at the repo root) importable.
repo_root = pathlib.Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from llm_tokens.chess import parquet_to_txt_shards
from llm_tokens.chess.tokenizer_factory import init_tokenizer


def tokenize_shards(tok, text_shards, out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for i, shard_path in enumerate(text_shards):
        out_path = out_dir / (pathlib.Path(shard_path).stem + ".npy")
        print(f"[tokenize] shard {i + 1}/{len(text_shards)}: {shard_path}")

        ids = []
        with open(shard_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    ids.extend(tok.encode(line))

        if ids:
            np.save(out_path, np.array(ids, dtype=np.uint32))
            total += len(ids)
            print(f"[tokenize] saved {len(ids):,} tokens → {out_path.name} | total {total:,}")

    print(f"[tokenize] done. {total:,} tokens ({total / 1e6:.1f}M) in {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", required=True, help="Cleaned parquet file or directory")
    ap.add_argument("--out-dir", required=True, help="Output directory for .npy token shards")
    ap.add_argument("--column", default="ctx", help="Text column to tokenize (default: ctx)")
    ap.add_argument("--tokenizer", default="LanTokenizer",
                    help="Tokenizer name registered in tokenizer_factory")
    ap.add_argument("--max-shard-mb", type=int, default=2,
                    help="Max size (MB) of each intermediate .txt shard")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    txt_dir = out_dir / "_txt_shards"

    # 1) parquet → .txt shards
    shard_names = parquet_to_txt_shards(
        args.parquet, txt_dir,
        column=args.column, prefix="ctx",
        max_bytes=args.max_shard_mb * 1024 * 1024,
    )
    shards = sorted(str(txt_dir / pathlib.Path(n).name) for n in shard_names)

    # 2) tokenize each shard → .npy
    tok = init_tokenizer(name=args.tokenizer, config={
        "special_tokens": ["<bos>", "<eos>", "<pad>"],
        "include_move_numbers": False,
    })
    tokenize_shards(tok, shards, out_dir)


if __name__ == "__main__":
    main()
