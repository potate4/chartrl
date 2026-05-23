"""Hierarchical Correct-Path Consistency (HCPC) reward.

HCPC is a cross-rollout bonus that conditions on ground-truth correctness
and, among the rollouts in a group that get the table and answer right,
rewards consistent table extraction together with diverse reasoning traces.
The bonus is applied per-rollout only to rollouts in the correct-path
subgroup G+.

This file matches the formulation reported in the paper:

    G+ = {o_i in G : y_hat_i = y* AND sim(T_hat_i, T*) >= tau}

    B_HCPC = (|G+|/K) * (w_table * C_table + w_reason * D_reason)

with tau = 0.8, w_table = 2.0, w_reason = 1.5. The per-rollout reward is
B_HCPC for rollouts in G+ and 0 otherwise.

C_type (chart-type consistency among correct rollouts) was considered in
an earlier prototype and removed because it is at ceiling for almost
every group where HCPC fires and therefore contributes a near-constant
offset rather than a differentiating signal.
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
    reward: float                       # Group-level bonus (for logging)
    per_rollout_rewards: List[float]    # Per-rollout: bonus for correct, 0 otherwise
    c_table: float                      # Table consistency among correct rollouts
    d_reason: float                     # Reasoning diversity among correct rollouts
    correct_rate: float                 # Fraction of correct rollouts (|G+|/K)
    num_correct: int                    # |G+|
    details: Dict[str, Any]


class HCPCComputer:
    """Computes the HCPC cross-rollout bonus."""

    def __init__(
        self,
        w_table: float = 2.0,
        w_reason: float = 1.5,
        table_sim_threshold: float = 0.8,
        answer_tolerance: float = 0.05,
    ):
        """Initialize HCPC computer.

        Args:
            w_table: Weight for table-consistency term.
            w_reason: Weight for reasoning-diversity term.
            table_sim_threshold: Threshold tau for considering the predicted
                table close enough to ground truth (default 0.8 to forbid
                "lucky" rollouts that produce a right answer from a wrong
                table).
            answer_tolerance: Relative tolerance for numeric answer matching.
        """
        self.w_table = w_table
        self.w_reason = w_reason
        self.table_sim_threshold = table_sim_threshold
        self.answer_tolerance = answer_tolerance

    def compute(
        self,
        rollouts: List[str],
        ground_truth: Dict[str, Any],
    ) -> HCPCResult:
        """Compute HCPC for a group of K rollouts on the same prompt.

        Args:
            rollouts: List of K raw model outputs.
            ground_truth: Dict containing at least 'table' and 'label'.

        Returns:
            HCPCResult with the group-level bonus value, the per-rollout
            reward vector, and the level-specific quantities for logging.
        """
        parsed_rollouts = [parse_response(r) for r in rollouts]

        # Step 1: filter to the correct-path subset G+.
        correct_rollouts, correct_indices = self._filter_correct(
            parsed_rollouts, ground_truth
        )

        num_correct = len(correct_rollouts)
        n_rollouts = len(rollouts)
        correct_rate = num_correct / n_rollouts if rollouts else 0.0

        # HCPC requires at least 2 correct rollouts to define pairwise terms.
        if num_correct < 2:
            return HCPCResult(
                reward=0.0,
                per_rollout_rewards=[0.0] * n_rollouts,
                c_table=0.0,
                d_reason=0.0,
                correct_rate=correct_rate,
                num_correct=num_correct,
                details={"reason": "insufficient_correct_rollouts"},
            )

        # Step 2: pairwise table consistency among correct rollouts.
        c_table = self._compute_table_consistency(correct_rollouts)

        # Step 3: pairwise reasoning diversity among correct rollouts.
        d_reason = self._compute_reasoning_diversity(correct_rollouts)

        # Step 4: combine into the group-level bonus.
        weighted_sum = self.w_table * c_table + self.w_reason * d_reason
        bonus = correct_rate * weighted_sum

        # Step 5: per-rollout application -- only G+ receives the bonus.
        correct_set = set(correct_indices)
        per_rollout_rewards = [
            bonus if i in correct_set else 0.0
            for i in range(n_rollouts)
        ]

        return HCPCResult(
            reward=bonus,
            per_rollout_rewards=per_rollout_rewards,
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
        """Return rollouts whose predicted table matches GT and whose
        predicted answer matches GT. The chart-type prediction is not
        used in the filter; it is supervised by the surrogate-task
        reward in R_base."""
        gt_table = ground_truth.get("table", {})
        gt_answer = ground_truth.get("label", "")

        correct_rollouts = []
        correct_indices = []

        for i, rollout in enumerate(parsed_rollouts):
            pred_table = rollout.get("table", {})
            if gt_table:
                table_sim = compute_table_similarity(pred_table, gt_table)
                if table_sim < self.table_sim_threshold:
                    continue

            pred_answer = rollout.get("answer", "")
            if not self._answers_match(pred_answer, gt_answer):
                continue

            correct_rollouts.append(rollout)
            correct_indices.append(i)

        return correct_rollouts, correct_indices

    def _answers_match(self, pred: str, label: str) -> bool:
        pred_num = try_parse_numeric(pred)
        label_num = try_parse_numeric(label)
        if pred_num is not None and label_num is not None:
            if label_num != 0:
                return abs(pred_num - label_num) / abs(label_num) <= self.answer_tolerance
            return abs(pred_num - label_num) <= self.answer_tolerance
        pred_norm = normalize_answer(pred)
        label_norm = normalize_answer(label)
        if not pred_norm or not label_norm:
            return False
        if pred_norm == label_norm:
            return True
        return pred_norm in label_norm or label_norm in pred_norm

    def _compute_table_consistency(self, rollouts: List[Dict]) -> float:
        """Mean pairwise table similarity among the correct-path rollouts."""
        if len(rollouts) < 2:
            return 1.0
        tables = [r.get("table", {}) for r in rollouts]
        n = len(tables)
        sims = []
        for i in range(n):
            for j in range(i + 1, n):
                sims.append(compute_table_similarity(tables[i], tables[j]))
        return sum(sims) / len(sims) if sims else 1.0

    def _compute_reasoning_diversity(self, rollouts: List[Dict]) -> float:
        """1 minus mean pairwise reasoning similarity among the correct-path
        rollouts. Reasoning embeddings use MiniLM-L6 (see utils.similarity)."""
        if len(rollouts) < 2:
            return 0.0
        reasonings = [r.get("reasoning", "") for r in rollouts]
        reasonings = [r for r in reasonings if r]
        if len(reasonings) < 2:
            return 0.0
        avg_sim, _ = compute_pairwise_similarity(reasonings)
        return 1.0 - avg_sim


def compute_hcpc_reward(
    rollouts: List[str],
    ground_truth: Dict[str, Any],
    w_table: float = 2.0,
    w_reason: float = 1.5,
    table_sim_threshold: float = 0.8,
) -> float:
    """Convenience function: return the scalar group-level HCPC bonus."""
    computer = HCPCComputer(
        w_table=w_table,
        w_reason=w_reason,
        table_sim_threshold=table_sim_threshold,
    )
    return computer.compute(rollouts, ground_truth).reward
