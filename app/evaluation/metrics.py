"""Accuracy metrics for evaluation."""

from typing import List, Tuple, Optional
import re

from utils.parsing import normalize_answer, try_parse_numeric


def exact_match(prediction: str, label: str) -> bool:
    """
    Check for exact string match after normalization.

    Args:
        prediction: Model prediction
        label: Ground truth label

    Returns:
        True if match, False otherwise
    """
    pred_norm = normalize_answer(prediction)
    label_norm = normalize_answer(label)
    return pred_norm == label_norm


def relaxed_accuracy(
    prediction: str,
    label: str,
    tolerance: float = 0.05,
) -> bool:
    """
    Check for match with numeric tolerance.

    For numeric answers: |pred - label| / |label| <= tolerance
    For text answers: exact string match

    Args:
        prediction: Model prediction
        label: Ground truth
        tolerance: Relative tolerance for numbers

    Returns:
        True if match, False otherwise
    """
    # Try numeric comparison first
    pred_num = try_parse_numeric(prediction)
    label_num = try_parse_numeric(label)

    if pred_num is not None and label_num is not None:
        if label_num != 0:
            rel_error = abs(pred_num - label_num) / abs(label_num)
            return rel_error <= tolerance
        else:
            return abs(pred_num - label_num) <= tolerance

    # Fall back to exact match
    return exact_match(prediction, label)


def compute_accuracy(
    predictions: List[str],
    labels: List[str],
    tolerance: float = 0.05,
) -> Tuple[float, List[bool]]:
    """
    Compute accuracy over a set of predictions.

    Args:
        predictions: List of predictions
        labels: List of ground truth labels
        tolerance: Numeric tolerance

    Returns:
        Tuple of (accuracy, per_sample_correct)
    """
    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels must have same length")

    if not predictions:
        return 0.0, []

    correct = [
        relaxed_accuracy(pred, label, tolerance)
        for pred, label in zip(predictions, labels)
    ]

    accuracy = sum(correct) / len(correct)

    return accuracy, correct


def extract_answer_from_output(output: str) -> str:
    """
    Extract answer from model output.

    Looks for <answer>...</answer> tags.

    Args:
        output: Full model output

    Returns:
        Extracted answer or empty string
    """
    match = re.search(r"<answer>(.*?)</answer>", output, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def compute_pass_at_k(
    results: List[List[bool]],
    k: int,
) -> float:
    """
    Compute Pass@k metric.

    Pass@k = fraction of problems with at least one correct answer in k attempts.

    Uses unbiased estimator from Chen et al. (2021).

    Args:
        results: List of per-sample correct flags, shape [num_samples, num_attempts]
        k: Number of attempts

    Returns:
        Pass@k score
    """
    if not results:
        return 0.0

    pass_at_k_scores = []

    for sample_results in results:
        n = len(sample_results)
        c = sum(sample_results)  # Number correct

        if n < k:
            # Not enough samples, use simple estimate
            score = 1.0 if c > 0 else 0.0
        else:
            # Unbiased estimator
            # Pass@k = 1 - C(n-c, k) / C(n, k)
            from math import comb
            if c == n:
                score = 1.0
            elif c == 0:
                score = 0.0
            else:
                score = 1.0 - comb(n - c, k) / comb(n, k)

        pass_at_k_scores.append(score)

    return sum(pass_at_k_scores) / len(pass_at_k_scores)


def compute_pass_at_k_spectrum(
    results: List[List[bool]],
    k_values: Optional[List[int]] = None,
) -> dict:
    """
    Compute Pass@k for multiple k values.

    Args:
        results: Per-sample results
        k_values: List of k values (default: [1, 2, 4, 8, 16, 32, 64])

    Returns:
        Dict mapping k to Pass@k score
    """
    if k_values is None:
        k_values = [1, 2, 4, 8, 16, 32, 64]

    return {k: compute_pass_at_k(results, k) for k in k_values}
