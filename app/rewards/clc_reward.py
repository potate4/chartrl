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

from ..utils.parsing import (
    parse_response,
    extract_numbers,
    extract_table_values,
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
