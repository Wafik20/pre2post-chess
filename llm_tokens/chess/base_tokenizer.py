# base_tokenizer.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

class BaseTokenizer(ABC):
    """Minimal interface for tokenizers used in pretraining."""

    # ---- required ----
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Convert text/PGN to token IDs."""
        raise NotImplementedError

    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        """Convert token IDs back to text/PGN."""
        raise NotImplementedError

    @abstractmethod
    def get_vocab(self) -> Dict[str, int]:
        """Return token -> id mapping (if available)."""
        raise NotImplementedError

    def bos_id(self) -> Optional[int]: return None
    def eos_id(self) -> Optional[int]: return None
    def pad_id(self) -> Optional[int]: return None
    def get_vocab_size(self) -> int: return len(self.get_vocab())

    def __call__(self, text: str) -> List[int]:
        """Alias for encode()."""
        return self.encode(text)
