"""
Diversity metrics for evaluating HCPC-RLVR models.

These metrics measure:
- C_table: Table extraction consistency among correct rollouts
- D_reason: Reasoning diversity among correct rollouts
- Coherence: Cross-level coherence (CLC) scores
"""

from typing import List, Dict, Any, Tuple

from utils.parsing import parse_response
from utils.similarity import compute_pairwise_similarity, compute_table_similarity


def compute_table_consistency(
    rollouts: List[str],
    filter_correct: bool = True,
    ground_truth: Dict[str, Any] = None,
) -> float:
    """
    Compute table extraction consistency across rollouts.

    High consistency indicates reliable data extraction.

    Args:
        rollouts: List of model outputs
        filter_correct: If True, only consider correct rollouts
        ground_truth: Required if filter_correct is True

    Returns:
        Average pairwise table similarity (0-1)
    """
    parsed = [parse_response(r) for r in rollouts]

    if filter_correct and ground_truth:
        parsed = _filter_correct_rollouts(parsed, ground_truth)

    if len(parsed) < 2:
        return 1.0  # Single or no rollouts are trivially consistent

    tables = [p.get("table", {}) for p in parsed]

    # Compute pairwise similarities
    similarities = []
    for i in range(len(tables)):
        for j in range(i + 1, len(tables)):
            sim = compute_table_similarity(tables[i], tables[j])
            similarities.append(sim)

    return sum(similarities) / len(similarities) if similarities else 1.0


def compute_reasoning_diversity(
    rollouts: List[str],
    filter_correct: bool = True,
    ground_truth: Dict[str, Any] = None,
) -> float:
    """
    Compute reasoning diversity across rollouts.

    High diversity indicates multiple valid reasoning strategies.

    Args:
        rollouts: List of model outputs
        filter_correct: If True, only consider correct rollouts
        ground_truth: Required if filter_correct is True

    Returns:
        Diversity score (0-1), where 1 = maximally diverse
    """
    parsed = [parse_response(r) for r in rollouts]

    if filter_correct and ground_truth:
        parsed = _filter_correct_rollouts(parsed, ground_truth)

    if len(parsed) < 2:
        return 0.0  # Single rollout has no diversity

    reasonings = [p.get("reasoning", "") for p in parsed]
    reasonings = [r for r in reasonings if r]  # Remove empty

    if len(reasonings) < 2:
        return 0.0

    # Compute average pairwise similarity
    avg_sim, _ = compute_pairwise_similarity(reasonings)

    # Diversity = 1 - similarity
    return 1.0 - avg_sim


def compute_diversity_metrics(
    rollouts: List[str],
    ground_truth: Dict[str, Any],
    filter_correct: bool = True,
) -> Dict[str, float]:
    """
    Cross-rollout diversity metrics:
        C_table : mean pairwise table similarity (among correct rollouts)
        D_reason: 1 - mean pairwise reasoning similarity (among correct rollouts)
    """
    c_table = compute_table_consistency(
        rollouts, filter_correct=filter_correct, ground_truth=ground_truth
    )

    d_reason = compute_reasoning_diversity(
        rollouts, filter_correct=filter_correct, ground_truth=ground_truth
    )

    # Count correct rollouts
    parsed = [parse_response(r) for r in rollouts]
    correct = _filter_correct_rollouts(parsed, ground_truth)
    correct_rate = len(correct) / len(parsed) if parsed else 0.0

    return {
        "c_table": c_table,
        "d_reason": d_reason,
        "correct_rate": correct_rate,
        "num_correct": len(correct),
        "num_total": len(rollouts),
    }


def _filter_correct_rollouts(
    parsed_rollouts: List[Dict],
    ground_truth: Dict[str, Any],
    table_threshold: float = 0.6,
    answer_tolerance: float = 0.05,
) -> List[Dict]:
    """Filter to correct rollouts based on ground truth."""
    from utils.parsing import normalize_answer, try_parse_numeric

    gt_table = ground_truth.get("table", {})
    gt_answer = ground_truth.get("label", "")

    correct = []

    for rollout in parsed_rollouts:
        # Check table
        pred_table = rollout.get("table", {})
        if gt_table:
            sim = compute_table_similarity(pred_table, gt_table)
            if sim < table_threshold:
                continue

        # Check answer
        pred_answer = rollout.get("answer", "")

        # Numeric comparison
        pred_num = try_parse_numeric(pred_answer)
        gt_num = try_parse_numeric(gt_answer)

        if pred_num is not None and gt_num is not None:
            if gt_num != 0:
                if abs(pred_num - gt_num) / abs(gt_num) > answer_tolerance:
                    continue
            elif abs(pred_num - gt_num) > answer_tolerance:
                continue
        else:
            # String comparison
            if normalize_answer(pred_answer) != normalize_answer(gt_answer):
                continue

        correct.append(rollout)

    return correct


def compute_ood_gap(
    id_accuracy: float,
    ood_accuracy: float,
) -> float:
    """
    Compute OOD generalization gap.

    Lower gap is better (model generalizes well).

    Args:
        id_accuracy: In-distribution accuracy
        ood_accuracy: Out-of-distribution accuracy

    Returns:
        Gap (ID - OOD), as percentage points
    """
    return (id_accuracy - ood_accuracy) * 100
