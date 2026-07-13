"""
training/hf_tokenizer_utils.py

Utility to save a self-contained HuggingFace-compatible tokenizer for chess
LAN tokenizers (LanTokenizer, LanTokenizerCoT, LanTokenizerSFT).

The generated ``tokenizer.py`` has **no** external imports from ``llm_tokens``
and **no** hardcoded absolute paths, so it works with
``AutoTokenizer.from_pretrained(path, trust_remote_code=True)`` on any machine.
"""
from __future__ import annotations

import inspect
import json
import re
from pathlib import Path
from typing import Optional, Union


# Path to base_tokenizer.py in the shared llm_tokens/ package at the repo root
# (sft/training/hf_tokenizer_utils.py -> parents[2] == repo root).
_BASE_TOKENIZER_PATH = (
    Path(__file__).resolve().parents[2] / "llm_tokens" / "chess" / "base_tokenizer.py"
)

# ---------------------------------------------------------------------------
# HFTokenizerWrapper template that will be written into the generated file.
# This code is self-bootstrapping: it reads vocab.json and
# tokenizer_config.json from its own directory.
# ---------------------------------------------------------------------------
_HF_WRAPPER_TEMPLATE = r'''
# ============================================================
# HuggingFace-compatible wrapper (auto-generated)
# ============================================================
import json as _json
from pathlib import Path as _Path
from transformers import PreTrainedTokenizer
import torch
from transformers.tokenization_utils_base import BatchEncoding

from huggingface_hub import hf_hub_download

class HFTokenizerWrapper(PreTrainedTokenizer):
    def __init__(self, model_max_length=2048, **kwargs):
        # These are usually provided by from_pretrained
        repo_id = kwargs.get("name_or_path") or kwargs.get("_name_or_path")
        revision = kwargs.get("revision", None)

        if not repo_id or "/" not in str(repo_id):
            # Fallback: user may pass repo_id explicitly
            repo_id = kwargs.get("repo_id", None)
        if not repo_id:
            raise ValueError("Cannot infer repo_id; pass repo_id=... or ensure name_or_path is set.")

        import os
        if os.path.isdir(repo_id):
            vocab_path = os.path.join(repo_id, "vocab.json")
            cfg_path   = os.path.join(repo_id, "tokenizer_config.json")
        else:
            vocab_path = hf_hub_download(repo_id=repo_id, filename="vocab.json", revision=revision)
            cfg_path   = hf_hub_download(repo_id=repo_id, filename="tokenizer_config.json", revision=revision)

        with open(vocab_path, "r", encoding="utf-8") as _f:
            saved_vocab = _json.load(_f)
        with open(cfg_path, "r", encoding="utf-8") as _f:
            _tok_cfg = _json.load(_f)

        lan_config = _tok_cfg.get("lan_config", {})
        lan_class_name = _tok_cfg.get("lan_tokenizer_class", "LanTokenizerSFT")

        _cls = globals()[lan_class_name]
        custom_tokenizer = _cls(config=lan_config)

        # Override vocab with the saved vocab
        custom_tokenizer._tok2id = saved_vocab
        from tokenizers import Tokenizer as _TkTokenizer
        from tokenizers.models import WordLevel as _WordLevel
        from tokenizers.pre_tokenizers import WhitespaceSplit as _WhitespaceSplit
        custom_tokenizer.tk = _TkTokenizer(_WordLevel(vocab=saved_vocab, unk_token=custom_tokenizer._unk))
        custom_tokenizer.tk.pre_tokenizer = _WhitespaceSplit()

        self.custom_tokenizer = custom_tokenizer
        self._vocab = dict(saved_vocab)
        self._id_to_token = {i: t for t, i in self._vocab.items()}

        bos_token = _tok_cfg.get("bos_token")
        eos_token = _tok_cfg.get("eos_token")
        pad_token = _tok_cfg.get("pad_token")
        unk_token = _tok_cfg.get("unk_token")
        env_token = _tok_cfg.get("env_token")
        if "env_id" in _tok_cfg:
            env_token = self._id_to_token[_tok_cfg.get("env_id")]
        else:
            env_token = _tok_cfg.get("env_token")
        self.env_token = env_token

        for _key in ("bos_token","eos_token","pad_token","unk_token","env_token",
                     "model_max_length","name_or_path","lan_config",
                     "lan_tokenizer_class","tokenizer_class","auto_map","use_fast",
                     "revision","repo_id"):
            kwargs.pop(_key, None)

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            unk_token=unk_token,
            model_max_length=model_max_length,
            **kwargs,
        )

    # ---- PreTrainedTokenizer interface ----

    @property
    def vocab_size(self):
        return len(self._vocab)

    def get_vocab(self):
        return dict(self._vocab)

    def _tokenize(self, text):
        return []  # we override encode/decode directly

    def _convert_token_to_id(self, token):
        return self._vocab.get(token, self._vocab.get(self.unk_token, 0))

    def _convert_id_to_token(self, index):
        return self._id_to_token.get(index, self.unk_token or "")

    def convert_tokens_to_string(self, tokens):
        ids = [self._convert_token_to_id(t) for t in tokens]
        return self.custom_tokenizer.decode(ids)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            return token_ids_0
        return token_ids_0 + token_ids_1

    def encode(self, text, add_special_tokens=True, **kwargs):
        ids = self.custom_tokenizer.encode(text)
        if add_special_tokens:
            return ids[:-1]  # strip trailing EOS; vLLM adds its own
        if (len(ids) >= 2
                and self.bos_token_id is not None
                and self.eos_token_id is not None
                and ids[0] == self.bos_token_id
                and ids[-1] == self.eos_token_id):
            return ids[1:-1]
        return ids

    def decode(self, token_ids, skip_special_tokens=True, **kwargs):
        import numpy as np
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()
        elif isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()
        return self.custom_tokenizer.decode(token_ids)

    def save_vocabulary(self, save_directory, filename_prefix=None):
        save_directory = _Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        vocab_file = save_directory / (
            (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        )
        with open(vocab_file, "w", encoding="utf-8") as f:
            _json.dump(self._vocab, f, ensure_ascii=False, indent=2)
        return (str(vocab_file),)

    def __call__(
        self,
        text,
        text_pair=None,
        add_special_tokens=True,
        truncation=False,
        max_length=None,
        padding=False,
        return_tensors=None,
        **kwargs,
    ):
        if text_pair is not None:
            raise ValueError("text_pair not supported for this tokenizer.")

        # Normalize to batch
        is_batched = isinstance(text, (list, tuple))
        texts = list(text) if is_batched else [text]

        input_ids = [self.encode(t, add_special_tokens=add_special_tokens) for t in texts]

        # Truncation
        if truncation and max_length is not None:
            if self.truncation_side == "left":
                input_ids = [ids[-max_length:] for ids in input_ids]
            else:
                input_ids = [ids[:max_length] for ids in input_ids]

        # Attention masks (pre-padding)
        attention_mask = [[1] * len(ids) for ids in input_ids]

        # Padding
        if padding:
            if padding == "max_length":
                if max_length is None:
                    raise ValueError("padding='max_length' requires max_length.")
                pad_to = max_length
            else:
                pad_to = max(len(ids) for ids in input_ids) if input_ids else 0

            pad_id = self.pad_token_id
            if pad_id is None:
                pad_id = self.bos_token_id if self.bos_token_id is not None else 0

            for i, ids in enumerate(input_ids):
                pad_len = pad_to - len(ids)
                if pad_len > 0:
                    input_ids[i] = ids + [pad_id] * pad_len
                    attention_mask[i] = attention_mask[i] + [0] * pad_len

        data = {"input_ids": input_ids, "attention_mask": attention_mask}

        # Unbatch if single example and no tensor return
        if not is_batched and return_tensors is None:
            data = {"input_ids": data["input_ids"][0], "attention_mask": data["attention_mask"][0]}

        # Tensors
        if return_tensors == "pt":
            data = {k: torch.tensor(v, dtype=torch.long) for k, v in data.items()}

        return BatchEncoding(data, tensor_type=None)


__all__ = ["HFTokenizerWrapper"]
'''


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_and_clean_source(path: Path) -> str:
    """Read a Python source file, strip relative imports of BaseTokenizer."""
    source = path.read_text(encoding="utf-8")
    # Remove  ``from .base_tokenizer import BaseTokenizer``
    source = re.sub(
        r'^\s*from\s+\.base_tokenizer\s+import\s+BaseTokenizer\s*\n?',
        '',
        source,
        flags=re.MULTILINE,
    )
    return source


def _strip_future_annotations(source: str) -> str:
    return re.sub(
        r'^\s*from\s+__future__\s+import\s+annotations\s*\n?',
        '',
        source,
        flags=re.MULTILINE,
    )


def _build_tokenizer_py(base_source: str, variant_source: str) -> str:
    """Assemble the self-contained ``tokenizer.py`` content."""
    base_source = _strip_future_annotations(base_source)
    variant_source = _strip_future_annotations(variant_source)

    header = (
        '"""\n'
        'Auto-generated self-contained HF tokenizer.\n'
        'Do NOT edit manually -- regenerate via '
        'training.hf_tokenizer_utils.save_hf_tokenizer().\n'
        '"""\n'
    )
    future = 'from __future__ import annotations\n\n'

    return (
        header
        + future
        + "# --- BaseTokenizer (inlined) ---\n"
        + base_source.strip() + "\n\n"
        + "# --- Concrete tokenizer (inlined) ---\n"
        + variant_source.strip() + "\n\n"
        + _HF_WRAPPER_TEMPLATE.strip() + "\n"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_hf_tokenizer(
    tokenizer,
    tokcfg,
    save_directory: Union[str, Path],
    model_max_length: int = 2048,
    env_id: Optional[int] = None,
) -> None:
    """Save a self-contained HF-compatible tokenizer alongside model weights.

    Parameters
    ----------
    tokenizer : LanTokenizer | LanTokenizerCoT | LanTokenizerSFT
        The custom tokenizer instance.
    tokcfg : dict or OmegaConf DictConfig
        The tokenizer section from the YAML config.
    save_directory : str or Path
        Directory to write tokenizer files into (typically ``hf_model/``).
    model_max_length : int
        Maximum sequence length.
    """
    save_directory = Path(save_directory)
    save_directory.mkdir(parents=True, exist_ok=True)

    # 1. Normalise config to plain dict
    try:
        from omegaconf import DictConfig, OmegaConf
        if isinstance(tokcfg, DictConfig):
            tokcfg = OmegaConf.to_container(tokcfg, resolve=True)
    except ImportError:
        pass
    if not isinstance(tokcfg, dict):
        tokcfg = dict(tokcfg)

    # 2. Resolve source files
    variant_src_path = Path(inspect.getsourcefile(type(tokenizer)))
    base_source = _BASE_TOKENIZER_PATH.read_text(encoding="utf-8")
    variant_source = _read_and_clean_source(variant_src_path)

    # 3. Build and write tokenizer.py
    tokenizer_py = _build_tokenizer_py(base_source, variant_source)
    (save_directory / "tokenizer.py").write_text(tokenizer_py, encoding="utf-8")

    # 4. Save vocab.json
    vocab = tokenizer.get_vocab()
    with open(save_directory / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    # 5. Resolve special tokens
    id2tok = {i: t for t, i in vocab.items()}
    bos_token = id2tok.get(tokenizer.bos_id())
    eos_token = id2tok.get(tokenizer.eos_id())
    pad_id = tokenizer.pad_id()
    pad_token = id2tok.get(pad_id) if pad_id is not None else None
    unk_token = tokcfg.get("unk", "<unk>")

    class_name = type(tokenizer).__name__

    # 6. tokenizer_config.json
    tokenizer_config = {
        "tokenizer_class": "HFTokenizerWrapper",
        "auto_map": {
            "AutoTokenizer": ["tokenizer.HFTokenizerWrapper", None],
        },
        "model_max_length": model_max_length,
        "bos_token": bos_token,
        "eos_token": eos_token,
        "pad_token": pad_token,
        "unk_token": unk_token,
        "env_token": id2tok.get(env_id) if env_id else None,
        "use_fast": False,
        "lan_config": tokcfg,
        "lan_tokenizer_class": class_name,
    }
    if env_id is not None:
        tokenizer_config["env_id"] = env_id
    with open(save_directory / "tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)

    # 7. special_tokens_map.json
    special_tokens_map = {
        "bos_token": bos_token,
        "eos_token": eos_token,
        "pad_token": pad_token,
        "unk_token": unk_token,
        "env_token": id2tok.get(env_id) if env_id else None,
    }
    with open(save_directory / "special_tokens_map.json", "w", encoding="utf-8") as f:
        json.dump(special_tokens_map, f, ensure_ascii=False, indent=2)

