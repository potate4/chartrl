"""
NSR + HCPC Trainer: Negative Sample Reinforcement with Hierarchical Consistency Penalty.

This combines NSR's conservative update strategy with HCPC's consistency insights,
but uses HCPC as a PENALTY rather than a reward.

Key insight: Instead of rewarding consistency among correct rollouts,
we PENALIZE inconsistency among wrong rollouts.

Why this makes sense:
1. NSR philosophy: Don't reinforce correct (preserves diversity), only penalize wrong
2. HCPC insight: Wrong answers that are also inconsistent are "doubly bad"
   - They got the wrong answer AND couldn't even agree on chart type/table
3. This gives stronger negative signal for confused/random outputs

Penalty formula for wrong rollouts:
    penalty = base_penalty × (1 + inconsistency_factor)

Where inconsistency_factor measures how inconsistent the wrong rollouts are
among themselves (inverted HCPC metrics).
"""

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from .base_trainer import BaseTrainer, PolicyMethodMixin
from rewards.hcpc_reward import HCPCComputer
from utils.parsing import parse_response, normalize_answer, try_parse_numeric


@dataclass
class NSRHCPCMetrics:
    """Metrics for NSR+HCPC computation."""
    num_correct: int
    num_wrong: int
    wrong_type_inconsistency: float
    wrong_table_inconsistency: float
    penalty_multiplier: float


class NSRHCPCTrainer(BaseTrainer, PolicyMethodMixin):
    """
    NSR trainer with HCPC-based inconsistency penalty.

    For correct samples: advantage = 0 (skip, like standard NSR)
    For wrong samples: advantage = -penalty × (1 + inconsistency_factor)

    The inconsistency_factor is computed from WRONG rollouts:
    - If wrong rollouts disagree on chart type → higher penalty
    - If wrong rollouts disagree on table extraction → higher penalty
    - If wrong rollouts have similar (bad) reasoning → penalty for groupthink

    This penalizes "confidently wrong" outputs more than "uncertainly wrong".
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_rollouts = None
        self._last_ground_truth = None
        self._hcpc_computer = HCPCComputer(
            w_type=self.config.rewards.w_type,
            w_table=self.config.rewards.w_table,
            w_reason=self.config.rewards.w_reason,
            table_sim_threshold=self.config.rewards.table_sim_threshold,
        )

    def compute_advantages(self, rewards: List[float]) -> List[float]:
        """
        Compute NSR+HCPC advantages.

        Args:
            rewards: Raw reward values for each rollout

        Returns:
            Advantages with HCPC inconsistency penalty applied to wrong samples
        """
        if not rewards:
            return rewards

        # Get threshold
        max_reward = self.get_max_reward(self.config)
        threshold = self.config.reward_threshold * max_reward

        # Classify rollouts
        correct_indices = [i for i, r in enumerate(rewards) if r >= threshold]
        wrong_indices = [i for i, r in enumerate(rewards) if r < threshold]

        # Compute inconsistency among WRONG rollouts
        inconsistency_factor = self._compute_wrong_inconsistency(wrong_indices)

        advantages = []
        for i, reward in enumerate(rewards):
            if reward >= threshold:
                # Correct: skip (no gradient) - standard NSR
                advantages.append(0.0)
            else:
                # Wrong: penalize with inconsistency multiplier
                base_penalty = threshold - reward

                # Apply inconsistency multiplier (1 + factor)
                # Higher inconsistency = stronger penalty
                penalty_multiplier = 1.0 + inconsistency_factor

                final_penalty = -base_penalty * penalty_multiplier
                advantages.append(final_penalty)

        return advantages

    def _compute_wrong_inconsistency(self, wrong_indices: List[int]) -> float:
        """
        Compute inconsistency factor among wrong rollouts.

        High inconsistency means wrong rollouts disagree with each other,
        suggesting the model is confused (not just wrong but uncertain).

        We want to penalize this more because:
        1. Confused outputs are less useful for learning
        2. It signals the model needs more guidance

        Returns:
            Inconsistency factor in [0, 1]
        """
        if len(wrong_indices) < 2:
            return 0.0

        if self._last_rollouts is None:
            return 0.0

        # Get wrong rollouts
        wrong_rollouts = [self._last_rollouts[i] for i in wrong_indices]
        parsed = [parse_response(r) for r in wrong_rollouts]

        # Compute type inconsistency (1 - consistency)
        type_inconsistency = 1.0 - self._compute_type_agreement(parsed)

        # Compute table inconsistency
        table_inconsistency = 1.0 - self._compute_table_agreement(parsed)

        # Compute reasoning similarity (high similarity in wrong = groupthink penalty)
        reasoning_similarity = self._compute_reasoning_similarity(parsed)

        # Combine: weight type and table inconsistency, add groupthink penalty
        # The idea:
        # - Inconsistent wrong answers → extra penalty (confused model)
        # - Very similar wrong reasoning → extra penalty (reinforced bad pattern)
        inconsistency = (
            0.4 * type_inconsistency +
            0.4 * table_inconsistency +
            0.2 * reasoning_similarity  # Penalize groupthink on wrong answers
        )

        return min(inconsistency, 1.0)

    def _compute_type_agreement(self, parsed_rollouts: List[Dict]) -> float:
        """Compute type agreement among rollouts."""
        types = [p.get("type", "").lower().strip() for p in parsed_rollouts]
        types = [t for t in types if t]

        if not types:
            return 0.0

        from collections import Counter
        counts = Counter(types)
        if not counts:
            return 0.0
        modal_count = counts.most_common(1)[0][1]
        return modal_count / len(types)

    def _compute_table_agreement(self, parsed_rollouts: List[Dict]) -> float:
        """Compute table agreement among rollouts."""
        from utils.similarity import compute_table_similarity

        tables = [p.get("table", {}) for p in parsed_rollouts]
        tables = [t for t in tables if t]

        if len(tables) < 2:
            return 0.0

        # Pairwise similarity
        sims = []
        for i in range(len(tables)):
            for j in range(i + 1, len(tables)):
                sims.append(compute_table_similarity(tables[i], tables[j]))

        return sum(sims) / len(sims) if sims else 0.0

    def _compute_reasoning_similarity(self, parsed_rollouts: List[Dict]) -> float:
        """Compute reasoning similarity (high = groupthink on wrong answer)."""
        from utils.similarity import compute_pairwise_similarity

        reasonings = [p.get("reasoning", "") for p in parsed_rollouts]
        reasonings = [r for r in reasonings if r]

        if len(reasonings) < 2:
            return 0.0

        avg_sim, _ = compute_pairwise_similarity(reasonings)
        return avg_sim

    def _create_reward_function(self):
        """Override to capture rollouts for HCPC computation."""
        base_reward_fn = super()._create_reward_function()

        def reward_fn_with_capture(completions, **kwargs):
            # Capture for inconsistency computation
            from trainers.base_trainer import BaseTrainer

            # Normalize completions
            normalized = []
            for c in completions:
                if isinstance(c, list):
                    # Extract from conversation format
                    for msg in c:
                        if isinstance(msg, dict) and msg.get("role") == "assistant":
                            content = msg.get("content", "")
                            if isinstance(content, list):
                                text_parts = [p.get("text", "") for p in content if p.get("type") == "text"]
                                normalized.append(" ".join(text_parts))
                            else:
                                normalized.append(str(content))
                            break
                    else:
                        normalized.append(str(c))
                else:
                    normalized.append(str(c))

            self._last_rollouts = normalized

            # Call base reward function
            return base_reward_fn(completions, **kwargs)

        return reward_fn_with_capture
