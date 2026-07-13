# llm_tokens/chess/tokenizer_factory.py
from typing import Optional
from .lan_tokenizer import LanTokenizer

_LAZY_REGISTRY = {
    "LanTokenizer": ("llm_tokens.chess.lan_tokenizer", "LanTokenizer"),
    "LanTokenizerSFT": ("llm_tokens.chess.lan_tokenizer_sft", "LanTokenizerSFT"),
    "LanTokenizerCoT": ("llm_tokens.chess.lan_tokenizer_cot", "LanTokenizerCoT"),
}


def init_tokenizer(name: str, config: Optional[dict] = None):
    if name == "LanTokenizer":
        return LanTokenizer(config=config)
    if name in _LAZY_REGISTRY:
        mod_path, cls_name = _LAZY_REGISTRY[name]
        import importlib
        mod = importlib.import_module(mod_path)
        cls = getattr(mod, cls_name)
        return cls(config=config)
    raise ValueError(f"Unknown tokenizer: {name}. Available: {list(_LAZY_REGISTRY)}")