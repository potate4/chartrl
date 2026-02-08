"""Evaluation module for HCPC-RLVR."""

from .metrics import exact_match, relaxed_accuracy, compute_accuracy
from .diversity_metrics import (
    compute_table_consistency,
    compute_reasoning_diversity,
    compute_coherence,
    compute_diversity_metrics,
)
from .evaluator import Evaluator, evaluate_model

__all__ = [
    "exact_match",
    "relaxed_accuracy",
    "compute_accuracy",
    "compute_table_consistency",
    "compute_reasoning_diversity",
    "compute_coherence",
    "compute_diversity_metrics",
    "Evaluator",
    "evaluate_model",
]
