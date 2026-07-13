"""Evaluation framework for chess move prediction tasks."""

from .inference import generate_move, generate_moves_batch
from .metrics import calculate_legal_move_accuracy, calculate_move_matching_accuracy, evaluate_predictions
from .base_evaluator import BaseEvaluator
from .example_evaluator import ExampleEvaluator, CustomEvaluator

__all__ = [
    "generate_move",
    "generate_moves_batch",
    "calculate_legal_move_accuracy",
    "calculate_move_matching_accuracy",
    "evaluate_predictions",
    "BaseEvaluator",
    "ExampleEvaluator",
    "CustomEvaluator",
]

