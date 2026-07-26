"""Compatibility shim for upstream pretraining checkpoint export.

The pinned upstream revision imports ``training.hf_tokenizer_utils`` while the
implementation is only present in the sibling SFT package. Keep a single
implementation and make the repository root importable before re-exporting it.
"""

from pathlib import Path
import sys


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sft.training.hf_tokenizer_utils import save_hf_tokenizer


__all__ = ["save_hf_tokenizer"]
