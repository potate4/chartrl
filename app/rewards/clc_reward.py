"""
Cross-Level Coherence (CLC) Reward.

This is a core contribution of the thesis. CLC verifies that the reasoning
actually references values from the extracted table, catching "hallucinated
reasoning" where models produce correct answers via fabricated intermediate steps.

Example of hallucinated reasoning:
    Extracted table: {2020: 100, 2021: 150}
    Reasoning: "The values are 80 and 170, so 80+170 = 250"
    Answer: 250 (correct, but reasoning uses fabricated values!)

CLC catches this by checking:
    C_coherence = |values_in_reasoning ∩ values_in_table| / |values_in_reasoning|
"""

from typing import Dict, Any, Set, List
from dataclasses import dataclass

from utils.parsing import (
    parse_response,
    extract_numbers,
    extract_table_values,
    try_parse_numeric,
)


@dataclass
class CLCResult:
    """Result of CLC computation."""
    reward: float
    coherence: float
    values_in_reasoning: Set[float]
    values_in_table: Set[float]
    overlap: Set[float]
    details: Dict[str, Any]


class CLCComputer:
    """
    Computes Cross-Level Coherence reward.

    Measures whether reasoning references values from the extracted table.
    """

    def __init__(
        self,
        w_clc: float = 1.0,
        tolerance: float = 0.05,
    ):
        """
        Initialize CLC computer.

        Args:
            w_clc: Weight for CLC reward
            tolerance: Tolerance for numeric matching
        """
        self.w_clc = w_clc
        self.tolerance = tolerance

    def compute(
        self,
        completion: str,
    ) -> CLCResult:
        """
        Compute CLC reward for a single completion.

        Args:
            completion: Model output

        Returns:
            CLCResult with coherence and details
        """
        parsed = parse_response(completion)

        # Extract values from reasoning
        reasoning = parsed.get("reasoning", "")
        reasoning_values = extract_numbers(reasoning)

        # Extract values from table
        table = parsed.get("table", {})
        table_values = extract_table_values(table)

        # If no values in reasoning, consider coherent
        if not reasoning_values:
            return CLCResult(
                reward=self.w_clc,  # Full reward for coherent
                coherence=1.0,
                values_in_reasoning=set(),
                values_in_table=table_values,
                overlap=set(),
                details={"reason": "no_values_in_reasoning"},
            )

        # Compute overlap (with tolerance)
        overlap = self._compute_overlap(reasoning_values, table_values)

        # Coherence = fraction of reasoning values found in table
        coherence = len(overlap) / len(reasoning_values)

        reward = self.w_clc * coherence

        return CLCResult(
            reward=reward,
            coherence=coherence,
            values_in_reasoning=reasoning_values,
            values_in_table=table_values,
            overlap=overlap,
            details={
                "num_reasoning_values": len(reasoning_values),
                "num_table_values": len(table_values),
                "num_overlap": len(overlap),
            },
        )

    def _compute_overlap(
        self,
        reasoning_values: Set[float],
        table_values: Set[float],
    ) -> Set[float]:
        """
        Compute overlap between value sets with tolerance.

        Args:
            reasoning_values: Values mentioned in reasoning
            table_values: Values in extracted table

        Returns:
            Set of reasoning values that appear in table
        """
        overlap = set()

        for rv in reasoning_values:
            for tv in table_values:
                if self._values_match(rv, tv):
                    overlap.add(rv)
                    break

        return overlap

    def _values_match(self, v1: float, v2: float) -> bool:
        """Check if two values match within tolerance."""
        if v2 != 0:
            return abs(v1 - v2) / abs(v2) <= self.tolerance
        return abs(v1 - v2) <= self.tolerance

    def compute_batch(
        self,
        completions: List[str],
    ) -> List[CLCResult]:
        """
        Compute CLC for a batch of completions.

        Args:
            completions: List of model outputs

        Returns:
            List of CLCResult
        """
        return [self.compute(c) for c in completions]


def compute_clc_reward(
    completion: str,
    w_clc: float = 1.0,
    tolerance: float = 0.05,
) -> float:
    """
    Convenience function to compute CLC reward.

    Args:
        completion: Model output
        w_clc: Weight for CLC reward
        tolerance: Tolerance for numeric matching

    Returns:
        CLC reward value
    """
    computer = CLCComputer(w_clc=w_clc, tolerance=tolerance)
    result = computer.compute(completion)
    return result.reward


def compute_clc_coherence(
    completion: str,
    tolerance: float = 0.05,
) -> float:
    """
    Compute just the coherence score (without weight).

    Args:
        completion: Model output
        tolerance: Tolerance for matching

    Returns:
        Coherence score in [0, 1]
    """
    computer = CLCComputer(w_clc=1.0, tolerance=tolerance)
    result = computer.compute(completion)
    return result.coherence


# ---------------------------------------------------------------------------
# GT-Anchored CLC — replaces process_reward without conflicting with d_reason
# ---------------------------------------------------------------------------

@dataclass
class GTCLCResult:
    """Result of GT-anchored CLC computation."""
    reward: float           # Final weighted score (w_gt_clc × recall), range [0, w_gt_clc]
    recall: float           # Raw recall in [0, 1]
    num_targets: int        # Number of weighted target values
    earned_weight: float    # Weight earned from matched targets
    total_weight: float     # Total possible weight
    details: Dict[str, Any]


def _num_match(v1: float, v2: float, tolerance: float = 0.05) -> bool:
    """Return True if v1 and v2 are within relative tolerance of each other."""
    if v2 != 0:
        return abs(v1 - v2) / abs(v2) <= tolerance
    return abs(v1 - v2) <= tolerance


class GTCLCComputer:
    """
    GT-Anchored Cross-Level Coherence reward.

    Checks whether the model's reasoning explicitly cites the key data values
    that appear in the GT reasoning chain.  Compatible with HCPC d_reason
    because it rewards WHAT values are used, not HOW they are expressed.

    Design:
      - Intermediate computed values (e.g. 14.2 - 8.7 = 5.5) are excluded from
        targets by intersecting GT-reasoning values with GT-table values.
        Only raw chart data points become targets, so rollout-specific rounding
        in intermediate steps never triggers a false penalty.
      - The final answer value is included as an additional target with higher
        weight (answer_weight) because explicitly deriving it in the reasoning
        chain is more important than citing any single source value.
    """

    def __init__(
        self,
        w_gt_clc: float = 1.0,
        value_weight: float = 1.0,
        answer_weight: float = 2.0,
        tolerance: float = 0.05,
    ):
        self.w_gt_clc = w_gt_clc
        self.value_weight = value_weight
        self.answer_weight = answer_weight
        self.tolerance = tolerance

    def compute(
        self,
        completion: str,
        gt_reasoning: str,
        gt_table: Dict[str, Any],
        gt_label: str,
    ) -> GTCLCResult:
        """
        Compute GT-anchored CLC for a single completion.

        Args:
            completion:  Model output string
            gt_reasoning: Ground-truth reasoning steps (from dataset)
            gt_table:     Ground-truth table dict (from dataset)
            gt_label:     Ground-truth answer string (from dataset)

        Returns:
            GTCLCResult with reward and diagnostic fields
        """
        # --- Build weighted target set ---

        # 1. Raw data values from the GT table (these are chart data points)
        gt_table_values = extract_table_values(gt_table) if gt_table else set()

        # 2. Values mentioned in GT reasoning, filtered to those that also
        #    appear in the GT table.  This excludes intermediate computed
        #    results (e.g. differences, sums) that may vary across rollouts.
        gt_key_values: Set[float] = set()
        if gt_table_values and gt_reasoning:
            for gv in extract_numbers(gt_reasoning):
                if any(_num_match(gv, tv, self.tolerance) for tv in gt_table_values):
                    gt_key_values.add(gv)

        # 3. Build targets dict: numeric_value -> weight
        targets: Dict[float, float] = {kv: self.value_weight for kv in gt_key_values}

        # 4. GT answer value gets answer_weight (the most critical target)
        gt_answer_num = try_parse_numeric(gt_label) if gt_label else None
        if gt_answer_num is not None:
            # If answer overlaps with an existing target, upgrade its weight
            upgraded = False
            for kv in list(targets.keys()):
                if _num_match(kv, gt_answer_num, self.tolerance):
                    targets[kv] = self.answer_weight
                    upgraded = True
                    break
            if not upgraded:
                targets[gt_answer_num] = self.answer_weight

        # Edge case: no numeric targets (categorical question, no GT table, etc.)
        if not targets:
            neutral = 0.5 * self.w_gt_clc
            return GTCLCResult(
                reward=neutral,
                recall=0.5,
                num_targets=0,
                earned_weight=0.0,
                total_weight=0.0,
                details={"reason": "no_numeric_targets"},
            )

        total_weight = sum(targets.values())

        # --- Extract values from model's reasoning steps ---
        parsed = parse_response(completion)
        model_reasoning = parsed.get("reasoning", "")
        model_values = extract_numbers(model_reasoning)

        if not model_values:
            return GTCLCResult(
                reward=0.0,
                recall=0.0,
                num_targets=len(targets),
                earned_weight=0.0,
                total_weight=total_weight,
                details={"reason": "no_values_in_model_reasoning"},
            )

        # --- Weighted recall ---
        earned_weight = 0.0
        matched_targets = []
        for target_val, weight in targets.items():
            for mv in model_values:
                if _num_match(mv, target_val, self.tolerance):
                    earned_weight += weight
                    matched_targets.append(target_val)
                    break

        recall = earned_weight / total_weight
        reward = self.w_gt_clc * recall

        return GTCLCResult(
            reward=reward,
            recall=recall,
            num_targets=len(targets),
            earned_weight=earned_weight,
            total_weight=total_weight,
            details={
                "num_matched": len(matched_targets),
                "answer_in_targets": gt_answer_num is not None,
                "gt_key_values": list(gt_key_values),
                "matched_targets": matched_targets,
            },
        )

    def compute_batch(
        self,
        completions: List[str],
        gt_reasoning: str,
        gt_table: Dict[str, Any],
        gt_label: str,
    ) -> List[GTCLCResult]:
        """Compute GT-CLC for a batch of completions sharing the same ground truth."""
        return [
            self.compute(c, gt_reasoning, gt_table, gt_label)
            for c in completions
        ]
