"""Hierarchical Correct-Path Consistency (HCPC) reward.

HCPC is a cross-rollout bonus that conditions on ground-truth correctness
and, among the rollouts in a group that get the table and answer right,
rewards consistent table extraction together with diverse reasoning traces.
The bonus is applied per-rollout only to rollouts in the correct-path
subgroup G+.

Formulation:

    G+ = {o_i in G : y_hat_i = y* AND sim(T_hat_i, T*) >= tau}

    B_cons = (|G+|/K) * (w_type * C_type + w_table * C_table)
    R_HCPC(o_i) = B_cons + w_reason * u_i   for o_i in G+
                = 0                         otherwise

with tau = 0.8, w_type = 1.0, w_table = 2.0, w_reason = 1.5. Here C_type
is the fraction of rollouts in G+ predicting the modal chart type, and
u_i = 1 - mean cosine sim of rollout i to all other reasonings in G+.
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
    c_type: float                       # Modal-type agreement among correct rollouts
    c_table: float                      # Table consistency among correct rollouts
    d_reason: float                     # Reasoning diversity among correct rollouts
    correct_rate: float                 # Fraction of correct rollouts (|G+|/K)
    num_correct: int                    # |G+|
    details: Dict[str, Any]


class HCPCComputer:
    """Computes the HCPC cross-rollout bonus.

    Set `use_d_reason=False` to run the ablation where the reasoning-
    diversity term w_reason*u_i is dropped. The shared consistency
    component B_cons = (|G+|/K) * (w_type*C_type + w_table*C_table)
    remains.
    """

    def __init__(
        self,
        w_type: float = 1.0,
        w_table: float = 2.0,
        w_reason: float = 1.5,
        table_sim_threshold: float = 0.8,
        answer_tolerance: float = 0.05,
        use_d_reason: bool = True,
    ):
        self.w_type = w_type
        self.w_table = w_table
        self.w_reason = w_reason
        self.table_sim_threshold = table_sim_threshold
        self.answer_tolerance = answer_tolerance
        self.use_d_reason = use_d_reason

    def compute(
        self,
        rollouts: List[str],
        ground_truth: Dict[str, Any],
    ) -> HCPCResult:
        parsed_rollouts = [parse_response(r) for r in rollouts]
        correct_rollouts, correct_indices = self._filter_correct(
            parsed_rollouts, ground_truth
        )

        num_correct = len(correct_rollouts)
        n_rollouts = len(rollouts)
        correct_rate = num_correct / n_rollouts if rollouts else 0.0

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

        c_type = self._compute_type_consistency(correct_rollouts)
        c_table = self._compute_table_consistency(correct_rollouts)

        if self.use_d_reason:
            d_reason, per_rollout_uniqueness = self._compute_reasoning_diversity(
                correct_rollouts
            )
        else:
            d_reason = 0.0
            per_rollout_uniqueness = [0.0] * num_correct

        b_cons = correct_rate * (self.w_type * c_type + self.w_table * c_table)

        per_rollout_rewards = [0.0] * n_rollouts
        for slot, i in enumerate(correct_indices):
            per_rollout_rewards[i] = b_cons + self.w_reason * per_rollout_uniqueness[slot]

        group_bonus = sum(per_rollout_rewards) / n_rollouts

        return HCPCResult(
            reward=group_bonus,
            per_rollout_rewards=per_rollout_rewards,
            c_type=c_type,
            c_table=c_table,
            d_reason=d_reason,
            correct_rate=correct_rate,
            num_correct=num_correct,
            details={
                "correct_indices": correct_indices,
                "b_cons": b_cons,
            },
        )

    def _filter_correct(
        self,
        parsed_rollouts: List[Dict],
        ground_truth: Dict[str, Any],
    ) -> Tuple[List[Dict], List[int]]:
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

    def _compute_type_consistency(self, rollouts: List[Dict]) -> float:
        types = [r.get("type", "") for r in rollouts]
        if not types:
            return 0.0
        from collections import Counter
        modal_count = Counter(types).most_common(1)[0][1]
        return modal_count / len(types)

    def _compute_table_consistency(self, rollouts: List[Dict]) -> float:
        if len(rollouts) < 2:
            return 1.0
        tables = [r.get("table", {}) for r in rollouts]
        n = len(tables)
        sims = []
        for i in range(n):
            for j in range(i + 1, n):
                sims.append(compute_table_similarity(tables[i], tables[j]))
        return sum(sims) / len(sims) if sims else 1.0

    def _compute_reasoning_diversity(
        self, rollouts: List[Dict]
    ) -> Tuple[float, List[float]]:
        n = len(rollouts)
        if n < 2:
            return 0.0, [0.0] * n
        reasonings = [r.get("reasoning", "") for r in rollouts]
        valid_idx = [k for k, r in enumerate(reasonings) if r and r.strip()]
        if len(valid_idx) < 2:
            return 0.0, [0.0] * n
        valid_texts = [reasonings[k] for k in valid_idx]
        avg_sim, sim_matrix = compute_pairwise_similarity(valid_texts)
        d_reason = 1.0 - avg_sim
        uniqueness = [0.0] * n
        m = len(valid_idx)
        for slot, k in enumerate(valid_idx):
            others = [sim_matrix[slot][j] for j in range(m) if j != slot]
            mean_sim_k = sum(others) / len(others) if others else 0.0
            uniqueness[k] = 1.0 - mean_sim_k
        return d_reason, uniqueness


def compute_hcpc_reward(
    rollouts: List[str],
    ground_truth: Dict[str, Any],
    w_type: float = 1.0,
    w_table: float = 2.0,
    w_reason: float = 1.5,
    table_sim_threshold: float = 0.8,
    use_d_reason: bool = True,
) -> float:
    computer = HCPCComputer(
        w_type=w_type,
        w_table=w_table,
        w_reason=w_reason,
        table_sim_threshold=table_sim_threshold,
        use_d_reason=use_d_reason,
    )
    return computer.compute(rollouts, ground_truth).reward
