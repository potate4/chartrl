"""
Hierarchical Correct-Path Consistency (HCPC) Reward.

This is a core contribution of the thesis. HCPC measures cross-rollout
properties at each level with level-specific objectives:

- Chart Type: Should be CONSISTENT (all correct rollouts identify same type)
- Table: Should be CONSISTENT (all correct rollouts extract same data)
- Reasoning: Should be DIVERSE (multiple valid strategies to reach answer)

The key insight is that we filter to GROUND-TRUTH-CORRECT rollouts first,
then measure these properties only among rollouts that are actually correct.
"""

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from utils.parsing import parse_response, normalize_answer, try_parse_numeric
from utils.similarity import (
    compute_pairwise_similarity,
    compute_table_similarity,
)


@dataclass
class HCPCResult:
    """Result of HCPC computation."""
    reward: float                       # Group-level reward (for logging)
    per_rollout_rewards: List[float]    # Per-rollout: bonus for correct, 0 for wrong
    c_type: float                       # Type consistency
    c_table: float                      # Table consistency
    d_reason: float                     # Reasoning diversity
    correct_rate: float                 # Fraction of correct rollouts
    num_correct: int                    # Number of correct rollouts
    details: Dict[str, Any]


class HCPCComputer:
    """
    Computes Hierarchical Correct-Path Consistency reward.

    This measures cross-rollout properties among ground-truth-correct
    rollouts with level-specific objectives.
    """

    def __init__(
        self,
        w_type: float = 1.0,
        w_table: float = 2.0,
        w_reason: float = 1.5,
        table_sim_threshold: float = 0.6,
        answer_tolerance: float = 0.05,
    ):
        """
        Initialize HCPC computer.

        Args:
            w_type: Weight for type consistency
            w_table: Weight for table consistency
            w_reason: Weight for reasoning diversity
            table_sim_threshold: Threshold for considering table "correct"
            answer_tolerance: Tolerance for numeric answer matching
        """
        self.w_type = w_type
        self.w_table = w_table
        self.w_reason = w_reason
        self.table_sim_threshold = table_sim_threshold
        self.answer_tolerance = answer_tolerance

    def compute(
        self,
        rollouts: List[str],
        ground_truth: Dict[str, Any],
    ) -> HCPCResult:
        """
        Compute HCPC reward for a group of rollouts.

        Args:
            rollouts: List of K model outputs
            ground_truth: Dict with type, table, answer

        Returns:
            HCPCResult with reward and metrics
        """
        # Parse all rollouts
        parsed_rollouts = [parse_response(r) for r in rollouts]

        # Step 1: Filter to fully correct rollouts
        correct_rollouts, correct_indices = self._filter_correct(
            parsed_rollouts, ground_truth
        )

        num_correct = len(correct_rollouts)
        correct_rate = num_correct / len(rollouts) if rollouts else 0.0

        n_rollouts = len(rollouts)

        # Need at least 2 correct rollouts for cross-rollout metrics
        if num_correct < 2:
            return HCPCResult(
                reward=0.0,
                per_rollout_rewards=[0.0] * n_rollouts,
                c_type=0.0,
                c_table=0.0,
                d_reason=0.0,
                correct_rate=correct_rate,
                num_correct=num_correct,
                details={"reason": "insufficient_correct_rollouts"},
            )

        # Step 2: Compute type consistency (among correct rollouts)
        c_type = self._compute_type_consistency(correct_rollouts)

        # Step 3: Compute table consistency (among correct rollouts)
        c_table = self._compute_table_consistency(correct_rollouts)

        # Step 4: Compute reasoning diversity (among correct rollouts)
        d_reason = self._compute_reasoning_diversity(correct_rollouts)

        # Step 5: Combine into HCPC reward
        # R_HCPC = correct_rate × (w1·C_type + w2·C_table + w3·D_reason)
        weighted_sum = (
            self.w_type * c_type +
            self.w_table * c_table +
            self.w_reason * d_reason
        )
        reward = correct_rate * weighted_sum

        # Per-rollout: only correct rollouts receive the HCPC bonus
        correct_set = set(correct_indices)
        per_rollout_rewards = [
            reward if i in correct_set else 0.0
            for i in range(n_rollouts)
        ]

        return HCPCResult(
            reward=reward,
            per_rollout_rewards=per_rollout_rewards,
            c_type=c_type,
            c_table=c_table,
            d_reason=d_reason,
            correct_rate=correct_rate,
            num_correct=num_correct,
            details={
                "correct_indices": correct_indices,
                "weighted_sum": weighted_sum,
            },
        )

    def _filter_correct(
        self,
        parsed_rollouts: List[Dict],
        ground_truth: Dict[str, Any],
    ) -> Tuple[List[Dict], List[int]]:
        """
        Filter to fully correct rollouts.

        A rollout is fully correct if:
        1. Table similarity >= threshold
        2. Answer matches ground truth

        Args:
            parsed_rollouts: List of parsed rollout dicts
            ground_truth: Ground truth dict

        Returns:
            Tuple of (correct_rollouts, indices)
        """
        gt_table = ground_truth.get("table", {})
        gt_answer = ground_truth.get("label", "")

        correct_rollouts = []
        correct_indices = []

        for i, rollout in enumerate(parsed_rollouts):
            # Check table similarity
            pred_table = rollout.get("table", {})
            if gt_table:
                table_sim = compute_table_similarity(pred_table, gt_table)
                if table_sim < self.table_sim_threshold:
                    continue

            # Check answer
            pred_answer = rollout.get("answer", "")
            if not self._answers_match(pred_answer, gt_answer):
                continue

            # All checks passed
            correct_rollouts.append(rollout)
            correct_indices.append(i)

        return correct_rollouts, correct_indices

    def _answers_match(self, pred: str, label: str) -> bool:
        """Check if predicted answer matches label."""
        # Try numeric comparison
        pred_num = try_parse_numeric(pred)
        label_num = try_parse_numeric(label)

        if pred_num is not None and label_num is not None:
            if label_num != 0:
                return abs(pred_num - label_num) / abs(label_num) <= self.answer_tolerance
            return abs(pred_num - label_num) <= self.answer_tolerance

        # String comparison (allow containment for sentence-style answers)
        pred_norm = normalize_answer(pred)
        label_norm = normalize_answer(label)
        if not pred_norm or not label_norm:
            return False
        if pred_norm == label_norm:
            return True
        return pred_norm in label_norm or label_norm in pred_norm

    def _compute_type_consistency(self, rollouts: List[Dict]) -> float:
        """
        Compute type consistency: fraction matching modal type.

        High consistency is desired (all should identify same type).
        """
        if not rollouts:
            return 0.0

        types = [r.get("type", "").lower().strip() for r in rollouts]
        types = [t for t in types if t]  # Remove empty

        if not types:
            return 1.0  # All empty = consistent

        # Find modal type
        from collections import Counter
        type_counts = Counter(types)
        modal_type, modal_count = type_counts.most_common(1)[0]

        return modal_count / len(types)

    def _compute_table_consistency(self, rollouts: List[Dict]) -> float:
        """
        Compute table consistency: average pairwise similarity.

        High consistency is desired (all should extract same data).
        """
        if len(rollouts) < 2:
            return 1.0

        tables = [r.get("table", {}) for r in rollouts]

        # Compute pairwise similarities
        n = len(tables)
        similarities = []

        for i in range(n):
            for j in range(i + 1, n):
                sim = compute_table_similarity(tables[i], tables[j])
                similarities.append(sim)

        return sum(similarities) / len(similarities) if similarities else 1.0

    def _compute_reasoning_diversity(self, rollouts: List[Dict]) -> float:
        """
        Compute reasoning diversity: 1 - average pairwise similarity.

        High diversity is desired (multiple valid strategies).
        """
        if len(rollouts) < 2:
            return 0.0

        reasonings = [r.get("reasoning", "") for r in rollouts]
        reasonings = [r for r in reasonings if r]  # Remove empty

        if len(reasonings) < 2:
            return 0.0

        # Compute average pairwise similarity
        avg_sim, _ = compute_pairwise_similarity(reasonings)

        # Diversity = 1 - similarity
        return 1.0 - avg_sim


def compute_hcpc_reward(
    rollouts: List[str],
    ground_truth: Dict[str, Any],
    w_type: float = 1.0,
    w_table: float = 2.0,
    w_reason: float = 1.5,
    table_sim_threshold: float = 0.6,
) -> float:
    """
    Convenience function to compute HCPC reward.

    Args:
        rollouts: List of model outputs
        ground_truth: Ground truth dict
        w_type: Weight for type consistency
        w_table: Weight for table consistency
        w_reason: Weight for reasoning diversity
        table_sim_threshold: Threshold for table matching

    Returns:
        HCPC reward value
    """
    computer = HCPCComputer(
        w_type=w_type,
        w_table=w_table,
        w_reason=w_reason,
        table_sim_threshold=table_sim_threshold,
    )
    result = computer.compute(rollouts, ground_truth)
    return result.reward
