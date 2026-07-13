#!/usr/bin/env python3
"""Preprocess raw Lichess human games (single input → single output).

Reads a parquet file (or a directory of parquet files) of raw Lichess games and:
  1. drops bullet games,
  2. computes avg_elo = (white_elo + black_elo) / 2,
  3. deduplicates by game `id`,
  4. (optionally) filters to an Elo range,
and writes the cleaned games to one output parquet directory.

Expected input columns include at least: `id`, `white_elo`, `black_elo`, `event`.
(The released dataset already contains the text column used downstream; see README.)

Usage:
    python preprocess_human_games.py \
        --input  /path/to/raw_lichess_parquet_or_dir \
        --output /path/to/clean_games \
        --min-elo 800 --max-elo 3000
"""
import argparse
from pathlib import Path

import duckdb


def preprocess_human_games(input_path: str, output_dir: str,
                           min_elo: int = 800, max_elo: int = 3000) -> None:
    in_p = Path(input_path)
    glob = str(in_p / "**/*.parquet") if in_p.is_dir() else str(in_p)
    out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(":memory:")
    con.execute("SET threads = 8")

    # 1) load + drop bullet + compute avg_elo
    con.execute(f"""
        CREATE TABLE games AS
        SELECT *,
               CAST((white_elo + black_elo) / 2.0 AS INTEGER) AS avg_elo
        FROM read_parquet('{glob}', hive_partitioning=false)
        WHERE event NOT ILIKE '%bullet%'
    """)
    loaded = con.execute("SELECT COUNT(*) FROM games").fetchone()[0]
    print(f"[human] loaded (non-bullet): {loaded:,}")

    # 2) dedup by game id (keep one row per id)
    con.execute("""
        CREATE TABLE games_dedup AS
        SELECT * EXCLUDE (rn) FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY id) AS rn FROM games
        ) WHERE rn = 1
    """)
    deduped = con.execute("SELECT COUNT(*) FROM games_dedup").fetchone()[0]
    print(f"[human] after dedup by id: {deduped:,}  (removed {loaded - deduped:,})")

    # 3) Elo filter + export to parquet shards
    con.execute(f"""
        COPY (
            SELECT * FROM games_dedup
            WHERE avg_elo >= {min_elo} AND avg_elo < {max_elo}
        ) TO '{out_dir}/part-{{0:05d}}.parquet'
        (FORMAT PARQUET, ROW_GROUP_SIZE 100000, COMPRESSION 'ZSTD')
    """)
    kept = con.execute(f"""
        SELECT COUNT(*) FROM games_dedup
        WHERE avg_elo >= {min_elo} AND avg_elo < {max_elo}
    """).fetchone()[0]
    con.close()
    print(f"[human] kept (elo {min_elo}-{max_elo}): {kept:,}  →  {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Raw Lichess parquet file or directory")
    ap.add_argument("--output", required=True, help="Output directory for cleaned parquet")
    ap.add_argument("--min-elo", type=int, default=800)
    ap.add_argument("--max-elo", type=int, default=3000)
    args = ap.parse_args()
    preprocess_human_games(args.input, args.output, args.min_elo, args.max_elo)


if __name__ == "__main__":
    main()
