"""Base reward functions from Chart-RVR."""

import re
import json
from typing import Dict, Any, Optional, List

from ..utils.parsing import (
    parse_response,
    normalize_answer,
    try_parse_numeric,
    split_reasoning_steps,
)
from ..utils.similarity import compute_similarity


def format_reward(completion: str) -> float:
    """
    Check if output follows the expected format.

    Expected format:
        <think>
        <type>...</type>
        <table>...</table>
        ...reasoning...
        </think>
        <answer>...</answer>

    Returns:
        2.0 if format is correct, 0.0 otherwise
    """
    # Check for required tags
    pattern = r"<think>.*?<type>.*?</type>.*?<table>.*?</table>.*?</think>.*?<answer>.*?</answer>"

    if re.search(pattern, completion, re.DOTALL):
        return 2.0
    return 0.0


def accuracy_reward(
    completion: str,
    label: str,
    tolerance: float = 0.05,
) -> float:
    """
    Check if answer matches ground truth.

    Handles both numeric (with tolerance) and text matching.

    Args:
        completion: Model output
        label: Ground truth answer
        tolerance: Relative tolerance for numeric comparison

    Returns:
        1.0 if correct, 0.0 otherwise
    """
    # Extract answer from completion
    parsed = parse_response(completion)
    pred_answer = parsed["answer"]

    if not pred_answer:
        return 0.0

    # Try numeric comparison first
    pred_num = try_parse_numeric(pred_answer)
    label_num = try_parse_numeric(label)

    if pred_num is not None and label_num is not None:
        # Numeric comparison with tolerance
        if label_num != 0:
            rel_error = abs(pred_num - label_num) / abs(label_num)
            return 1.0 if rel_error <= tolerance else 0.0
        else:
            return 1.0 if abs(pred_num - label_num) <= tolerance else 0.0

    # Fall back to string comparison
    pred_norm = normalize_answer(pred_answer)
    label_norm = normalize_answer(label)

    return 1.0 if pred_norm == label_norm else 0.0


def length_reward(
    completion: str,
    min_length: int = 70,
    max_length: int = 250,
) -> float:
    """
    Reward appropriate reasoning length.

    Encourages thorough but not excessive reasoning.

    Args:
        completion: Model output
        min_length: Minimum characters for full reward
        max_length: Maximum characters before penalty

    Returns:
        Reward value
    """
    parsed = parse_response(completion)
    reasoning = parsed["reasoning"]

    if not reasoning:
        return 0.0

    length = len(reasoning)

    # Base reward for meeting minimum
    if length >= min_length:
        reward = 1.0
    else:
        # Partial reward for shorter reasoning
        reward = length / min_length

    # Penalty for excessive length
    if length > max_length:
        reward -= 0.5

    # Bonus for step markers
    steps = split_reasoning_steps(reasoning)
    step_bonus = min(0.25 * len(steps), 1.0)
    reward += step_bonus

    return max(0.0, reward)


def token_count_reward(completion: str) -> float:
    """
    Check for proper tag structure.

    Ensures exactly one of each required tag.

    Returns:
        2.0 if all tags correct, 0.0 otherwise
    """
    required_tags = [
        ("<think>", "</think>"),
        ("<answer>", "</answer>"),
        ("<type>", "</type>"),
        ("<table>", "</table>"),
    ]

    for open_tag, close_tag in required_tags:
        if completion.count(open_tag) != 1:
            return 0.0
        if completion.count(close_tag) != 1:
            return 0.0

    # Check proper ordering
    if not re.search(r"<think>\s*\n?\s*<type>", completion):
        return 0.0
    if not re.search(r"</type>\s*\n?\s*<table>", completion):
        return 0.0

    return 2.0


def chart_type_reward(
    completion: str,
    chart_type: str,
) -> float:
    """
    Check if chart type prediction is correct.

    Args:
        completion: Model output
        chart_type: Ground truth chart type

    Returns:
        1.0 if correct, 0.0 otherwise
    """
    parsed = parse_response(completion)
    pred_type = parsed["type"]

    if not pred_type or not chart_type:
        return 0.0

    # Normalize and compare
    pred_type = pred_type.strip().lower()
    chart_type = chart_type.strip().lower()

    # Handle common variations
    type_aliases = {
        "bar chart": "bar",
        "bar graph": "bar",
        "line chart": "line",
        "line graph": "line",
        "pie chart": "pie",
        "scatter plot": "scatter",
    }

    pred_type = type_aliases.get(pred_type, pred_type)
    chart_type = type_aliases.get(chart_type, chart_type)

    return 1.0 if pred_type == chart_type else 0.0


def table_reward(
    completion: str,
    table: Dict[str, Any],
) -> float:
    """
    Reward for accurate table extraction.

    Checks:
    - JSON validity
    - Column accuracy
    - Row/value accuracy

    Args:
        completion: Model output
        table: Ground truth table

    Returns:
        Reward value (0-2 range)
    """
    parsed = parse_response(completion)
    pred_table = parsed["table"]

    if not pred_table:
        return 0.0

    reward = 0.0

    # Reward for valid JSON (already parsed)
    reward += 0.5

    if not table:
        return reward

    # Column comparison
    gt_cols = set(str(c).lower() for c in table.get("columns", []))
    pred_cols = set(str(c).lower() for c in pred_table.get("columns", []))

    if gt_cols and pred_cols:
        col_overlap = len(gt_cols & pred_cols) / max(len(gt_cols), 1)
        reward += col_overlap * 0.5

    # Row comparison
    gt_rows = table.get("rows", [])
    pred_rows = pred_table.get("rows", [])

    if gt_rows and pred_rows:
        row_score = _compare_table_rows(gt_rows, pred_rows)
        reward += row_score

    return min(reward, 2.0)


def _compare_table_rows(
    gt_rows: List,
    pred_rows: List,
    tolerance: float = 0.05,
) -> float:
    """Compare table rows with tolerance."""
    if not gt_rows or not pred_rows:
        return 0.0

    # Flatten to values
    def flatten(rows):
        values = []
        for row in rows:
            if isinstance(row, list):
                values.extend(row)
            elif isinstance(row, dict):
                values.extend(row.values())
            else:
                values.append(row)
        return values

    gt_vals = flatten(gt_rows)
    pred_vals = flatten(pred_rows)

    if not gt_vals:
        return 0.0

    # Count matches
    matches = 0
    pred_used = [False] * len(pred_vals)

    for gt_val in gt_vals:
        for i, pred_val in enumerate(pred_vals):
            if pred_used[i]:
                continue
            if _values_match(gt_val, pred_val, tolerance):
                matches += 1
                pred_used[i] = True
                break

    return matches / len(gt_vals)


def _values_match(v1, v2, tolerance: float = 0.05) -> bool:
    """Check if two values match."""
    # String match
    if str(v1).strip().lower() == str(v2).strip().lower():
        return True

    # Numeric match
    try:
        n1 = float(v1)
        n2 = float(v2)
        if n2 != 0:
            return abs(n1 - n2) / abs(n2) <= tolerance
        return abs(n1 - n2) <= tolerance
    except (ValueError, TypeError):
        return False


def process_reward(
    completion: str,
    reasoning: str,
) -> float:
    """
    Reward for reasoning process alignment.

    Uses semantic similarity between predicted and ground truth reasoning.

    Args:
        completion: Model output
        reasoning: Ground truth reasoning

    Returns:
        Similarity score (0-1)
    """
    parsed = parse_response(completion)
    pred_reasoning = parsed["reasoning"]

    if not pred_reasoning or not reasoning:
        return 0.0

    # Compute semantic similarity
    similarity = compute_similarity(pred_reasoning, reasoning)

    return similarity


def compute_base_rewards(
    completion: str,
    ground_truth: Dict[str, Any],
) -> Dict[str, float]:
    """
    Compute all base rewards for a completion.

    Args:
        completion: Model output
        ground_truth: Dict with label, table, chart_type, reasoning

    Returns:
        Dict mapping reward names to values
    """
    rewards = {}

    # Format reward
    rewards["format"] = format_reward(completion)

    # Accuracy reward
    rewards["accuracy"] = accuracy_reward(
        completion,
        ground_truth.get("label", ""),
    )

    # Length reward
    rewards["length"] = length_reward(completion)

    # Token count reward
    rewards["token_count"] = token_count_reward(completion)

    # Chart type reward
    if ground_truth.get("chart_type"):
        rewards["chart_type"] = chart_type_reward(
            completion,
            ground_truth["chart_type"],
        )
    else:
        rewards["chart_type"] = 0.0

    # Table reward
    if ground_truth.get("table"):
        rewards["table"] = table_reward(
            completion,
            ground_truth["table"],
        )
    else:
        rewards["table"] = 0.0

    # Process reward
    if ground_truth.get("reasoning"):
        rewards["process"] = process_reward(
            completion,
            ground_truth["reasoning"],
        )
    else:
        rewards["process"] = 0.0

    # Total
    rewards["total"] = sum(rewards.values())

    return rewards
