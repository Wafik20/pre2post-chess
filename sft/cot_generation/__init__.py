"""
Chain-of-Thought data generation for chess using tree search methods.

This package provides tools for generating CoT training data from chess positions
by sampling trajectories and labelling them via subtree search.
"""

# Core generator components
from .generator import (
    TreeNode,
    board_from_pgn,
    CoTDataGenerator,
)

# Sampling policies for tree expansion
from .policy import (
    random_sampling_policy,
    stockfish_sampling_policy,
    model_sampling_policy,
    stockfish_eval,
)

__all__ = [
    # Core classes
    "TreeNode",
    "CoTDataGenerator",

    # Utilities
    "board_from_pgn",

    # Sampling policies
    "random_sampling_policy",
    "stockfish_sampling_policy",
    "model_sampling_policy",

    # Evaluation functions
    "stockfish_eval",
]
