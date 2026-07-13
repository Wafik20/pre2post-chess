# utils.py — parquet → text helpers used by the data-preprocessing tokenizer step.
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import json


def parquet_to_raw_txt(parquet_path: str, out_path: str, column: str = "ctx") -> str:
    """Dump a single text column of a parquet to a flat .txt (one game per line)."""
    out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)
    s = (pd.read_parquet(parquet_path, columns=[column])[column]
         .dropna().astype(str).map(str.strip).replace("", pd.NA).dropna())
    s.to_csv(out, index=False, header=False)
    return str(out)


def parquet_to_txt_shards(parquet_path, out_dir, column="ctx", prefix="ctx",
                          max_bytes=512 * 1024 * 1024, encoding="utf-8",
                          keep_unicode=True, start_shard=0):
    """Stream a parquet (file or directory) into size-bounded .txt shards.

    Reads `column` row-group by row-group and writes one line per game,
    rotating to a new shard once `max_bytes` is exceeded. Returns the list of
    shard paths written.
    """
    esc = lambda s: json.dumps(s, ensure_ascii=not keep_unicode)[1:-1]
    p = Path(parquet_path)
    if p.is_dir():
        parquet_paths = sorted(str(x) for x in p.glob("*.parquet"))
        if not parquet_paths:
            raise ValueError(f"No parquet files found in directory: {parquet_path}")
    else:
        parquet_paths = [str(parquet_path)]

    all_row_groups = []
    for path in parquet_paths:
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            all_row_groups.append((pf, rg))

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Find first non-existing shard index >= start_shard
    shard = start_shard
    while (out_dir / f"{prefix}.{shard:04d}.txt").exists():
        shard += 1

    written, f = 0, (out_dir / f"{prefix}.{shard:04d}.txt").open("w", encoding=encoding, newline="")
    outs = [str(f.name)]

    def rotate():
        nonlocal shard, written, f
        f.close(); written = 0
        shard += 1
        while (out_dir / f"{prefix}.{shard:04d}.txt").exists():
            shard += 1
        f = (out_dir / f"{prefix}.{shard:04d}.txt").open("w", encoding=encoding, newline="")
        outs.append(str(f.name))

    for pf, rg in all_row_groups:
        col = pf.read_row_group(rg, columns=[column]).column(0).to_pandas()
        col = col.dropna().astype(str).map(str.strip).replace("", pd.NA).dropna()
        for v in col:
            line = esc(v) + "\n"
            if len(line) < 50:        # skip too-short lines
                continue
            b = len(line.encode(encoding))
            if written and written + b > max_bytes:
                rotate()
            f.write(line); written += b
    f.close()
    return outs
