#!/usr/bin/env python3
"""Preprocess crawled puzzle games (single CSV in → single CSV out).

Takes the CSV produced by `download_puzzles.py` (one row per puzzle, with a `PGN`
column of the source game and a `FEN` of the puzzle position) and:
  1. deduplicates by the normalized move sequence of the source game, and
  2. extracts `ctx`: the moves (SAN, with move numbers) leading up to the puzzle
     position, i.e. the game replayed until the board matches `FEN`.

Writes the deduplicated rows plus a new `ctx` column.

Usage:
    python preprocess_puzzles.py \
        --input  filtered_puzzles_with_pgn.csv \
        --output puzzles_processed.csv
"""
import argparse
import hashlib
import io
import re

import chess
import chess.pgn
import pandas as pd
from tqdm import tqdm


def normalize_pgn(pgn_text: str) -> str:
    """Canonical SAN move string for a PGN (used for dedup)."""
    if pd.isna(pgn_text) or pgn_text == "":
        return ""
    try:
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if game is None:
            return ""
        board = game.board()
        moves = []
        for move in game.mainline_moves():
            moves.append(board.san(move))
            board.push(move)
        return " ".join(moves).strip()
    except Exception:
        # Fallback: strip comments/variations and collapse whitespace
        text = re.sub(r'\{[^}]*\}', '', pgn_text)
        text = re.sub(r'\([^)]*\)', '', text)
        return ' '.join(text.split()).strip()


def pgn_hash(normalized_pgn: str) -> str:
    return hashlib.sha256(normalized_pgn.encode("utf-8")).hexdigest() if normalized_pgn else ""


def extract_ctx_until_fen(pgn_text: str, target_fen: str) -> str:
    """Replay the game in SAN until the board matches `target_fen`'s position.

    Returns the context moves formatted with move numbers, e.g. "1. e4 e5 2. Nf3".
    """
    try:
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if game is None:
            return ""
        board = game.board()
        target = target_fen.split()[0]   # piece placement only
        moves = []
        for move in game.mainline_moves():
            if board.fen().split()[0] == target:
                break
            moves.append(board.san(move))
            board.push(move)
            if board.fen().split()[0] == target:
                break
        formatted = []
        for i, mv in enumerate(moves):
            if i % 2 == 0:
                formatted.append(f"{i // 2 + 1}. {mv}")
            else:
                formatted.append(mv)
        return " ".join(formatted)
    except Exception as e:
        print(f"[puzzles] ctx error: {e}")
        return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="CSV from download_puzzles.py (needs PGN, FEN)")
    ap.add_argument("--output", default="puzzles_processed.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if "PGN" not in df.columns:
        raise SystemExit("[puzzles] input has no PGN column")

    seen = set()
    keep_rows, ctx_values = [], []
    n_dup = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="dedup+ctx"):
        pgn = str(row.get("PGN", ""))
        h = pgn_hash(normalize_pgn(pgn))
        if h and h in seen:        # duplicate source game
            n_dup += 1
            continue
        if h:
            seen.add(h)
        ctx = "" if (pd.isna(pgn) or pgn == "") else extract_ctx_until_fen(pgn, str(row.get("FEN", "")))
        keep_rows.append(row)
        ctx_values.append(ctx)

    out = pd.DataFrame(keep_rows).reset_index(drop=True)
    out["ctx"] = ctx_values
    out.to_csv(args.output, index=False)
    print(f"[puzzles] {len(df):,} → {len(out):,} unique (removed {n_dup:,} dups) → {args.output}")


if __name__ == "__main__":
    main()
