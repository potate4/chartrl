"""Base reward functions from Chart-RVR."""

import re
import json
from typing import Dict, Any, Optional, List

from utils.parsing import (
    parse_response,
    normalize_answer,
    try_parse_numeric,
    split_reasoning_steps,
)
from utils.similarity import compute_similarity


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
        1.0 if format is correct, 0.0 otherwise
    """
    # Check for required tags and order
    pattern = r"<think>.*?<type>.*?</type>.*?<table>.*?</table>.*?</think>.*?<answer>.*?</answer>"
    return 1.0 if re.search(pattern, completion, re.DOTALL) else 0.0


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
    min_tokens: int = 128,
    max_tokens: int = 768,
) -> float:
    """
    Reward appropriate output length (token-count proxy).

    Args:
        completion: Model output
        min_tokens: Minimum token count for reward
        max_tokens: Maximum token count for reward

    Returns:
        1.0 if within range, 0.0 otherwise
    """
    if not completion:
        return 0.0
    token_count = len(completion.split())
    return 1.0 if (min_tokens <= token_count <= max_tokens) else 0.0


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
        "scatter plot": "scatterplot",
        "scatter": "scatterplot",
        "stacked bar chart": "stacked bar",
        "stacked bar graph": "stacked bar",
        "stacked area chart": "stacked area",
        "area chart": "area",
    }

    pred_type = type_aliases.get(pred_type, pred_type)
    chart_type = type_aliases.get(chart_type, chart_type)

    return 1.0 if pred_type == chart_type else 0.0


def table_reward(
    completion: str,
    table: Dict[str, Any],
) -> float:
    """
    Reward for accurate table extraction (Chart-RVR).

    Formula:
    - Column header accuracy (exact match fraction)
    - Cell accuracy (exact positional match fraction)
    - +0.5 bonus if JSON is strictly parseable

    Args:
        completion: Model output
        table: Ground truth table

    Returns:
        Reward value
    """
    parsed = parse_response(completion)
    pred_table = parsed["table"]
    parseable_json = parsed.get("table_parse_success_strict", False)

    if not pred_table:
        return 0.0

    reward = 0.5 if parseable_json else 0.0

    if not table:
        return reward

    # Column header accuracy (exact match fraction)
    gt_cols = table.get("columns", [])
    pred_cols = pred_table.get("columns", [])
    if gt_cols:
        col_matches = 0
        for c in gt_cols:
            if str(c).strip().lower() in [str(pc).strip().lower() for pc in pred_cols]:
                col_matches += 1
        reward += col_matches / len(gt_cols)

    # Cell accuracy (exact positional match fraction)
    gt_rows = table.get("rows", [])
    pred_rows = pred_table.get("rows", [])
    if gt_rows:
        total_cells = 0
        matched_cells = 0
        for i, gt_row in enumerate(gt_rows):
            if not isinstance(gt_row, list):
                gt_row = [gt_row]
            total_cells += len(gt_row)
            if i >= len(pred_rows):
                continue
            pred_row = pred_rows[i]
            if not isinstance(pred_row, list):
                pred_row = [pred_row]
            for j, gt_cell in enumerate(gt_row):
                if j >= len(pred_row):
                    continue
                if _values_match(gt_cell, pred_row[j]):
                    matched_cells += 1
        if total_cells > 0:
            reward += matched_cells / total_cells

    return reward


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
    Reward for process conformity (Chart-RVR).

    Computes:
    - Reg: mean step-wise similarity for first m steps
    - Rrs: similarity for remaining steps

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

    pred_steps = split_reasoning_steps(pred_reasoning)
    gt_steps = split_reasoning_steps(reasoning)

    if not pred_steps or not gt_steps:
        return 0.0

    m = min(2, len(pred_steps), len(gt_steps))
    if m == 0:
        return 0.0

    # Step-wise conformity for first m steps
    step_sims = []
    for i in range(m):
        step_sims.append(compute_similarity(pred_steps[i], gt_steps[i]))
    reg = sum(step_sims) / len(step_sims) if step_sims else 0.0

    # Reasoning alignment for remaining steps
    pred_tail = " ".join(pred_steps[m:])
    gt_tail = " ".join(gt_steps[m:])
    rrs = compute_similarity(pred_tail, gt_tail) if pred_tail and gt_tail else 0.0

    return reg + rrs


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
