#!/usr/bin/env python3
"""Download the source-game PGNs for Lichess puzzles (single-pass).

Starts from the public Lichess puzzle database CSV
(https://database.lichess.org/#puzzles), filters/quality-balances puzzles by
rating, then fetches each puzzle's *source game* PGN via the Lichess games-export
API and writes a CSV with a `PGN` column for downstream preprocessing.

Usage:
    python download_puzzles.py \
        --puzzle-csv /path/to/lichess_db_puzzle.csv \
        --output     filtered_puzzles_with_pgn.csv
"""
import argparse
import random
import re
import time

import pandas as pd
import requests
from tqdm import tqdm

# ── Quality filters on the raw puzzle DB ──
RD_MAX = 100          # max RatingDeviation (lower = more reliable rating)
NBPLAYS_MIN = 50      # min times the puzzle has been played
POP_MIN = 0           # min popularity

# ── Rating bins + per-bin cap (keeps the set roughly balanced across strength) ──
BIN_EDGES = [0, 1600, 2000, 2400, 4000]
BIN_LABELS = ["1200-1600", "1600-2000", "2000-2400", "2400+"]
MAX_PER_BIN = 5000

OUTPUT_COLS = ["PuzzleId", "FEN", "Moves", "Rating", "RatingDeviation", "NbPlays",
               "Popularity", "Themes", "GameUrl", "RatingBin", "PGN"]

# ── Lichess games-export API (batched) ──
EXPORT_URL = "https://lichess.org/games/export/_ids"
BATCH_SIZE = 200
TIMEOUT = 30
SLEEP = 2              # politeness delay between batches (seconds)
MAX_RETRIES = 2
BASE_BACKOFF = 5
MAX_BACKOFF = 120

GID_PAT = re.compile(r'(?:https?://)?(?:www\.)?lichess\.org/([A-Za-z0-9]{8})(?=[/?#]|$)')
SESSION = requests.Session()


def fetch_batch(game_ids):
    """Fetch a comma-separated batch of game ids as concatenated PGN text."""
    params = {"clocks": "false", "evals": "false", "opening": "true", "pgnInJson": "false"}
    headers = {"Accept": "application/x-chess-pgn",
               "Content-Type": "text/plain; charset=utf-8",
               "User-Agent": "puzzle-source-fetch/0.1"}
    data = ",".join(game_ids).encode("utf-8")

    attempt = 0
    while True:
        try:
            r = SESSION.post(EXPORT_URL, params=params, data=data, headers=headers, timeout=TIMEOUT)
            if r.status_code == 200:
                return r.text
            if r.status_code == 429:           # rate limited
                ra = r.headers.get("Retry-After")
                wait = int(ra) if ra and ra.isdigit() else 60
            elif 500 <= r.status_code < 600:   # server error → backoff
                wait = min(MAX_BACKOFF, BASE_BACKOFF * (2 ** attempt)) + random.uniform(0, 1.5)
            else:
                r.raise_for_status()
                return r.text
        except requests.exceptions.RequestException:
            wait = min(MAX_BACKOFF, BASE_BACKOFF * (2 ** attempt)) + random.uniform(0, 1.5)

        attempt += 1
        if attempt > MAX_RETRIES:
            raise RuntimeError(f"Failed after {MAX_RETRIES} retries")
        time.sleep(wait)


def split_pgns(blob):
    return re.split(r'\n(?=\[Event )', blob.strip()) if blob.strip() else []


def pgn_game_id(pgn):
    m = re.search(r'\[Site\s+"https?://lichess\.org/([A-Za-z0-9]{8})', pgn)
    if m:
        return m.group(1)
    m = re.search(r'\[Link\s+"https?://lichess\.org/([A-Za-z0-9]{8})', pgn)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--puzzle-csv", required=True, help="lichess_db_puzzle.csv")
    ap.add_argument("--output", default="filtered_puzzles_with_pgn.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.puzzle_csv)
    df = df[(df["RatingDeviation"] <= RD_MAX)
            & (df["NbPlays"] >= NBPLAYS_MIN)
            & (df["Popularity"] >= POP_MIN)].copy()

    # Rating-bin balanced subsample (cap per bin)
    df["RatingBin"] = pd.cut(df["Rating"], bins=BIN_EDGES, labels=BIN_LABELS,
                             include_lowest=True, right=False)
    df = (df.sort_values(["RatingBin", "NbPlays", "Rating"], ascending=[True, False, False])
            .groupby("RatingBin", group_keys=False).head(MAX_PER_BIN))

    # Resolve the source game id from the puzzle's GameUrl
    df["GameId"] = df["GameUrl"].astype(str).str.extract(GID_PAT)
    df = df[~df["GameId"].isna()].copy()
    df["GameId"] = df["GameId"].astype(str)
    ids = df["GameId"].tolist()
    print(f"[puzzles] fetching PGNs for {len(ids):,} games")

    # Batch-fetch PGNs from the Lichess export API
    pgn_map = {}
    for i in tqdm(range(0, len(ids), BATCH_SIZE), desc="fetch-pgn"):
        chunk = ids[i:i + BATCH_SIZE]
        try:
            text = fetch_batch(chunk)
        except Exception:
            print(f"[puzzles] failed batch at {i}; backing off")
            time.sleep(60)
            continue
        for p in split_pgns(text):
            gid = pgn_game_id(p)
            if gid:
                pgn_map[gid] = p
        time.sleep(SLEEP)

    df["PGN"] = df["GameId"].map(pgn_map)
    df = df[df["PGN"].notna()].copy()
    out = df[[c for c in OUTPUT_COLS if c in df.columns]].reset_index(drop=True)
    out.to_csv(args.output, index=False)
    print(f"[puzzles] done: {len(out):,} puzzles with PGN → {args.output}")


if __name__ == "__main__":
    main()
