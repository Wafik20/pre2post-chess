# data.py
from pathlib import Path
from typing import Iterable
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


def _collect_sharded_npy_files(data_dir: Path) -> list[Path]:
    """Collect .npy files from shard_* subdirectories, sorted by shard folder then filename."""
    files = []
    for shard_dir in sorted(data_dir.glob("shard_*")):
        files.extend(sorted(shard_dir.glob("*.npy")))
    return files


def _is_tokenized_file(file_path: Path) -> bool:
    """Check if file is a pre-tokenized numpy file."""
    return file_path.suffix == '.npy'


def _load_tokens_from_file(file_path: Path, tokenizer=None, max_tokens: int = None) -> list[int]:
    """Load tokens from either text file (with tokenizer) or pre-tokenized .npy file."""
    file_path = Path(file_path)
    
    if _is_tokenized_file(file_path):
        # Load pre-tokenized numpy array
        token_array = np.load(file_path)
        if max_tokens is not None and max_tokens > 0:
            token_array = token_array[:max_tokens]
        return token_array.tolist()
    else:
        # Load and tokenize text file
        if tokenizer is None:
            raise ValueError("Tokenizer required for text files")
        ids = []
        with file_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ids.extend(tokenizer.encode(line))
                if max_tokens is not None and max_tokens > 0 and len(ids) >= max_tokens:
                    break
        return ids[:max_tokens] if (max_tokens and max_tokens > 0) else ids


class PackedTextDataset(Dataset):
    def __init__(self, txt_files: list[str], tokenizer=None, seq_len: int = 512):
        """
        Dataset that packs all tokens into memory.
        
        Args:
            txt_files: List of file paths (either .txt or .npy)
            tokenizer: Tokenizer instance (required for .txt files, optional for .npy)
            seq_len: Sequence length for training
        """
        self.seq_len = seq_len
        ids: list[int] = []
        
        for p in txt_files:
            file_ids = _load_tokens_from_file(Path(p), tokenizer=tokenizer)
            ids.extend(file_ids)
            
        if len(ids) < seq_len + 1:
            raise ValueError("Not enough tokens to form a single sequence.")
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.nseq = (len(self.ids) - 1) // seq_len

    def __len__(self): 
        return self.nseq

    def __getitem__(self, i: int):
        s = i * self.seq_len
        x = self.ids[s : s + self.seq_len]
        y = self.ids[s + 1 : s + self.seq_len + 1]
        return x, y


class ShardedPackedTextDataset(Dataset):
    def __init__(self, txt_files: list[str], tokenizer=None, seq_len: int = 512, cache_size: int = 1_000_000, num_shards: int = None, shuffle: bool = False, seed: int = None):
        """
        Dataset that loads shards on-demand to save memory.

        Args:
            txt_files: List of file paths or directory (supports .txt or .npy files)
            tokenizer: Tokenizer instance (required for .txt files, optional for .npy)
            seq_len: Sequence length for training
            cache_size: Maximum tokens to load per shard
            num_shards: Limit number of shards to use
            shuffle: Whether to shuffle the order of shards
            seed: Seed for deterministic shard shuffling (required if shuffle=True for reproducibility)
        """
        self.tok = tokenizer
        self.seq_len = seq_len

        # print(f"[dataset] Loading dataset from: {txt_files}")
        if len(txt_files) == 1:
            txt_files = txt_files[0]
            if isinstance(txt_files, (str, Path)):
                p = Path(txt_files)
                if p.is_dir():
                    # Nested shard_* subdirectory structure takes priority
                    if any(p.glob("shard_*")):
                        all_shards = _collect_sharded_npy_files(p)
                    else:
                        txt_shards = sorted(p.glob("*.txt"))
                        npy_shards = sorted(p.glob("*.npy"))
                        all_shards = txt_shards + npy_shards
                    
                    if num_shards is not None:
                        self.txt_files = all_shards[:num_shards]
                    else:
                        self.txt_files = all_shards
                elif p.is_file():
                    self.txt_files = [p]
                else:
                    raise ValueError(f"Invalid txt_files path: {txt_files}")
        else:
            self.txt_files = [Path(p) for p in txt_files]
            if num_shards is not None:
                self.txt_files = self.txt_files[:num_shards]
        
        if shuffle:
            import random
            rng = random.Random(seed) if seed is not None else random.Random()
            rng.shuffle(self.txt_files)

        # Determine cache_size: if <= 0, auto-detect from first shard
        if cache_size is not None and cache_size > 0:
            self.cache_size = cache_size
        else:
            # Load first shard to determine actual token count per shard
            sample_tokens = len(_load_tokens_from_file(self.txt_files[0], tokenizer=tokenizer))
            self.cache_size = sample_tokens + 1024  # small buffer
            print(f"[dataset] Auto-detected shard size: ~{sample_tokens} tokens, cache_size={self.cache_size}")

        self._ids = torch.empty(0, dtype=torch.long)
        self._file_idx = 0  # current shard index

        self.steps_per_shard = max(1, (self.cache_size - 1) // self.seq_len)
        self.total_steps = self.steps_per_shard * len(self.txt_files)
        self._loaded_shard = -1

    def _load_next_shard(self, shard_idx: int | None = None):
        """Load a shard into cache (optionally the given index)."""
        if not self.txt_files:
            raise ValueError("No shards provided.")

        if shard_idx is None:
            shard_idx = self._file_idx
        p = self.txt_files[shard_idx]
        print(f"[dataset] Loading shard: {p.name}")

        # Use helper function to load tokens (works for both .txt and .npy)
        ids = _load_tokens_from_file(p, tokenizer=self.tok, max_tokens=self.cache_size)

        self._ids = torch.tensor(ids, dtype=torch.long)
        self.nseq = max(1, (len(self._ids) - 1) // self.seq_len)
        # cap to nominal steps-per-shard so __len__ contract holds
        self.nseq = min(self.nseq, self.steps_per_shard)

        self._loaded_shard = shard_idx
        # do NOT advance self._file_idx here anymore

    def __len__(self):
        return self.total_steps

    def __getitem__(self, idx):
        # decide shard and local position
        shard_idx  = (idx // self.steps_per_shard) % len(self.txt_files)
        local_idx  =  idx % self.steps_per_shard

        # load the right shard if needed
        if self._loaded_shard != shard_idx or len(self._ids) == 0:
            self._load_next_shard(shard_idx)

        # if this shard has fewer seqs than nominal capacity, wrap
        if self.nseq == 0:
            raise RuntimeError("Shard has zero sequences; increase cache_size or check data.")
        local_idx = local_idx % self.nseq

        s = local_idx * self.seq_len
        x = self._ids[s : s + self.seq_len]
        y = self._ids[s + 1 : s + self.seq_len + 1]
        return x, y



def create_dataloader(
    txt_files,
    tokenizer=None,
    batch_size=64,
    seq_len=512,
    num_workers=0,
    shuffle=True,
    cache_size=1_000_000,
    dataset_shuffle=False,  # New param for dataset-level shuffle
    num_shards=None,
    prefetch_factor=None,  # New: prefetch batches per worker
    persistent_workers=False,  # New: keep workers alive between epochs
    seed=None,  # Seed for deterministic shard shuffling
) -> DataLoader:
    """
    Create a DataLoader for training.
    
    Args:
        txt_files: List of file paths or directory path (supports .txt or .npy files)
        tokenizer: Tokenizer instance (required for .txt files, optional for .npy files)
        batch_size: Batch size for training
        seq_len: Sequence length
        num_workers: Number of DataLoader workers
        shuffle: Whether to shuffle batches
        cache_size: Maximum tokens to cache per shard
        dataset_shuffle: Whether to shuffle the order of shards
        num_shards: Limit number of shards to use
        prefetch_factor: Number of batches to prefetch per worker (default 2 if num_workers > 0)
        persistent_workers: Keep workers alive between epochs to avoid recreation overhead
    
    Returns:
        DataLoader instance
    """
    # ds = PackedTextDataset(list(txt_files), tokenizer, seq_len=seq_len)
    ds = ShardedPackedTextDataset(list(txt_files), tokenizer, seq_len=seq_len, cache_size=cache_size, shuffle=dataset_shuffle, num_shards=num_shards, seed=seed)
    
    # Build DataLoader kwargs
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    
    # Only add prefetch_factor and persistent_workers if num_workers > 0
    if num_workers > 0:
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor
        if persistent_workers:
            loader_kwargs["persistent_workers"] = persistent_workers
    
    return DataLoader(ds, **loader_kwargs)
