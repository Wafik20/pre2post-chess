# lan_tokenizer_cot.py
"""
LAN Tokenizer with CoT support (trajectory tokens).

This extends the base LAN tokenizer with CoT-specific tokens:
- <T> for trajectory markers
- <best> for best move markers
"""
from typing import List, Dict, Optional
import io
import chess, chess.pgn
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from .base_tokenizer import BaseTokenizer

_RESULT = {"1-0", "0-1", "1/2-1/2", "*"}
FILES = "abcdefgh"
RANKS = "12345678"
SQUARES = [f+r for f in FILES for r in RANKS]
PROMOS = "QRBN"
DIGITS = set("0123456789")


def _vocab_with_cot(
    include_move_numbers: bool,
    keep_result: bool,
    bos: str,
    eos: str,
    unk: str,
    pad: str = None
) -> Dict[str, int]:
    """Create vocabulary including CoT special tokens."""
    base = [bos, eos, unk]
    if pad is not None:
        base.append(pad)
    ops = ["x", "=", "+", "#", "O-O", "O-O-O", ".", "..."]
    toks = base + list("KQRBNP") + SQUARES + list(PROMOS) + ops
    if include_move_numbers:
        toks += list("0123456789")
    if keep_result:
        toks += sorted(_RESULT)
    
    # Add CoT special tokens
    cot_tokens = [
        "<T>",      # Trajectory marker
        "<best>",   # Best move marker
    ]
    toks += cot_tokens
    
    return {t: i for i, t in enumerate(dict.fromkeys(toks))}


class LanTokenizerCoT(BaseTokenizer):
    """
    LAN Tokenizer with CoT trajectory support.
    
    This tokenizer extends the base LAN tokenizer with:
    - <T> token for marking trajectory boundaries
    - <best> token for marking best move selection points
    
    Example usage:
        text = "<T>Nc6 Bc4<best>Nc6 d3<best>d6<T>"
        encoded = tokenizer.encode(text)
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Args:
            config: Configuration dict with tokenizer settings
        """
        config = config or {}
        
        include_move_numbers = config.get("include_move_numbers", True)
        include_black_tripledots = config.get("include_black_tripledots", False)
        bos = config.get("bos", "<bos>")
        eos = config.get("eos", "<eos>")
        unk = config.get("unk", "<unk>")
        pad = config.get("pad", None)
        keep_result = config.get("keep_result", False)
        
        self._bos = bos
        self._eos = eos
        self._unk = unk
        self._pad = pad
        self._keep_res = keep_result
        self._include_nums = include_move_numbers
        self._include_black_ellipses = include_black_tripledots
        
        # Create vocabulary with CoT tokens
        tok2id = _vocab_with_cot(include_move_numbers, keep_result, bos, eos, unk, pad)
        self._tok2id = tok2id
        
        # Initialize tokenizer
        self.tk = Tokenizer(WordLevel(vocab=tok2id, unk_token=self._unk))
        self.tk.pre_tokenizer = WhitespaceSplit()
    
    def _pgn_to_tokens(self, text: str) -> List[str]:
        """Convert PGN text to tokens."""
        import os, contextlib
        with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
            g = chess.pgn.read_game(io.StringIO(text))
        if g is None:
            return text.split()
        
        b, out, n = g.board(), [], 1
        for mv in g.mainline_moves():
            if b.turn == chess.WHITE and self._include_nums:
                out += list(str(n)) + (
                    ["..."] if self._include_black_ellipses and b.fullmove_number < n else ["."]
                )
            
            if b.is_castling(mv):
                b.push(mv)
                suf = "#" if b.is_checkmate() else ("+" if b.is_check() else "")
                b.pop()
                out.append("O-O" if chess.square_file(mv.to_square) == 6 else "O-O-O")
                if suf:
                    out.append(suf)
                b.push(mv)
            else:
                piece = b.piece_at(mv.from_square).symbol().upper()
                frm = chess.square_name(mv.from_square)
                to = chess.square_name(mv.to_square)
                is_cap = b.is_capture(mv)
                promo = mv.promotion
                
                b.push(mv)
                suf = "#" if b.is_checkmate() else ("+" if b.is_check() else "")
                
                # Emit LAN tokens
                out.append(piece)
                out.append(frm)
                if is_cap:
                    out.append("x")
                out.append(to)
                if promo:
                    out += ["=", chess.piece_symbol(promo).upper()]
                if suf:
                    out.append(suf)
            
            if b.turn == chess.WHITE:
                n += 1
        
        res = g.headers.get("Result")
        if self._keep_res and res in _RESULT:
            out.append(res)
        
        return out
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.
        
        Handles both:
        - Regular PGN strings (parsed and tokenized)
        - CoT format strings with <T> and <best> tokens (split by whitespace)
        
        Args:
            text: Text to encode (PGN or CoT format)
        
        Returns:
            List of token IDs
        """
        # Check if this is CoT format (contains <T> or <best> tokens)
        is_cot_format = "<T>" in text or "<best>" in text
        
        if is_cot_format:
            # For CoT format, split by whitespace and encode directly
            # This preserves the special tokens
            tokens = [self._bos] + text.split() + [self._eos]
        else:
            # For regular PGN, use the PGN parser
            tokens = [self._bos] + self._pgn_to_tokens(text) + [self._eos]
        
        return self.tk.encode(" ".join(tokens)).ids
    
    def decode(self, ids: List[int]) -> str:
        """
        Decode token IDs to text.
        
        Args:
            ids: List of token IDs
        
        Returns:
            Decoded text
        """
        toks = [t for t in self.tk.decode(ids).split() if t not in {self._bos, self._eos}]
        
        # Check if this contains CoT format tokens
        is_cot_format = "<T>" in toks or "<best>" in toks
        
        if is_cot_format:
            # For CoT format, just join with spaces
            return " ".join(toks)
        
        # Otherwise, use LAN decoding logic
        out: List[str] = []
        i, n = 0, len(toks)
        
        while i < n:
            t = toks[i]
            
            if t and all(ch in DIGITS for ch in t):
                j = i
                num = []
                while j < n and all(ch in DIGITS for ch in toks[j]):
                    num.append(toks[j])
                    j += 1
                dots = ""
                if j < n and toks[j] in {".", "..."}:
                    dots = toks[j]
                    j += 1
                out.append("".join(num) + dots)
                i = j
                continue
            
            if t in {"O-O", "O-O-O"}:
                j = i + 1
                suf = toks[j] if j < n and toks[j] in {"+", "#"} else ""
                if suf:
                    j += 1
                out.append(t + suf)
                i = j
                continue
            
            if t in set("KQRBNP"):
                piece = t
                j = i + 1
                frm = toks[j] if j < n else ""
                j += 1
                cap = ""
                if j < n and toks[j] == "x":
                    cap = "x"
                    j += 1
                to = toks[j] if j < n else ""
                j += 1
                promo = ""
                if j + 1 <= n - 1 and toks[j] == "=" and toks[j + 1] in set(PROMOS):
                    promo = "=" + toks[j + 1]
                    j += 2
                suf = ""
                if j < n and toks[j] in {"+", "#"}:
                    suf = toks[j]
                    j += 1
                lan = f"{piece}{frm}{cap}{to}{promo}{suf}"
                out.append(lan)
                i = j
                continue
            
            out.append(t)
            i += 1
        
        return " ".join(out)
    
    def get_vocab(self) -> Dict[str, int]:
        """Get token-to-id vocabulary mapping."""
        return self._tok2id
    
    def bos_id(self) -> Optional[int]:
        """Get BOS token ID."""
        return self._tok2id[self._bos]
    
    def eos_id(self) -> Optional[int]:
        """Get EOS token ID."""
        return self._tok2id[self._eos]
    
    def pad_id(self) -> Optional[int]:
        """Get PAD token ID."""
        return self._tok2id.get(self._pad) if self._pad else None
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self._tok2id)

