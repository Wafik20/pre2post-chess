#!/usr/bin/env python3
"""FEN-based decontamination (single training parquet).

Removes training games that ever reach any *puzzle test position*, so the
pretraining set cannot contain the exact positions used for evaluation.

For every puzzle FEN (board placement + side to move), each training game is
replayed move-by-move; if any intermediate position matches, the game is dropped.

Usage:
    python decontaminate.py \
        --train-parquet /path/to/train_shard.parquet \
        --puzzle-files  /path/to/puzzles_test.csv /path/to/puzzles_grandmaster.csv \
        --out           /path/to/clean_shard.parquet \
        --removed       /path/to/removed_shard.parquet   # optional

Training rows must have a `moves_uci` column (list/sequence of UCI moves).
"""
import argparse
import glob
from pathlib import Path

import chess
import pandas as pd
from tqdm import tqdm


def collect_puzzle_fens(puzzle_paths) -> set[str]:
    """Unique normalized FENs (placement + side to move) from puzzle files."""
    fens: set[str] = set()
    for pattern in puzzle_paths:
        for path in sorted(glob.glob(pattern)):
            p = Path(path)
            if p.suffix == ".csv":
                df = pd.read_csv(path)
            elif p.suffix == ".parquet":
                df = pd.read_parquet(path)
            else:
                continue
            if "FEN" not in df.columns:
                print(f"[decon] no FEN column in {path}, skipping")
                continue
            for fen in df["FEN"].dropna():
                parts = fen.strip().split()
                fens.add(parts[0] + " " + parts[1] if len(parts) >= 2 else parts[0])
    return fens


def _norm(board: chess.Board) -> str:
    parts = board.fen().split()
    return parts[0] + " " + parts[1]


def game_touches_any_fen(moves_uci, test_fens: set[str]) -> bool:
    """True if any position in the replayed game matches a test FEN."""
    board = chess.Board()
    if _norm(board) in test_fens:
        return True
    for uci in moves_uci:
        try:
            board.push_uci(str(uci))
        except (ValueError, chess.InvalidMoveError):
            break
        if _norm(board) in test_fens:
            return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-parquet", required=True, help="One training parquet to clean")
    ap.add_argument("--puzzle-files", required=True, nargs="+",
                    help="Puzzle test file(s)/glob(s) with a FEN column")
    ap.add_argument("--out", required=True, help="Output parquet for clean games")
    ap.add_argument("--removed", default=None, help="Optional parquet for removed games")
    args = ap.parse_args()

    test_fens = collect_puzzle_fens(args.puzzle_files)
    print(f"[decon] {len(test_fens):,} unique puzzle positions")
    if not test_fens:
        raise SystemExit("[decon] no puzzle FENs found — nothing to do")

    df = pd.read_parquet(args.train_parquet)
    if "moves_uci" not in df.columns:
        raise SystemExit("[decon] training parquet has no moves_uci column")

    contaminated = [
        game_touches_any_fen(moves, test_fens)
        for moves in tqdm(df["moves_uci"].tolist(), desc="scanning games")
    ]
    mask = pd.Series(contaminated, index=df.index)

    clean = df[~mask]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(args.out, index=False)

    if args.removed and mask.any():
        Path(args.removed).parent.mkdir(parents=True, exist_ok=True)
        df[mask].to_parquet(args.removed, index=False)

    n = len(df)
    print(f"[decon] {n:,} → {len(clean):,} "
          f"(removed {int(mask.sum()):,}, {100 * mask.sum() / max(n, 1):.2f}%) → {args.out}")


if __name__ == "__main__":
    main()
