"""
Standard GRPO Trainer.

GRPO (Group Relative Policy Optimization) computes advantages as:
    advantage[i] = reward[i] - mean(rewards)

This reinforces above-average responses and penalizes below-average ones.

Issue: Strong positive reinforcement causes "reasoning collapse" - the model
converges to a single winning strategy, hurting OOD generalization.
"""

from typing import List

from .base_trainer import BaseTrainer, PolicyMethodMixin


class GRPOTrainer(BaseTrainer, PolicyMethodMixin):
    """
    Standard GRPO trainer.

    Uses relative advantages: advantage = reward - mean(rewards)

    This is the baseline method (Chart-RVR reproduction).
    """

    def compute_advantages(self, rewards: List[float]) -> List[float]:
        """
        Compute GRPO advantages.

        Standard formula: advantage[i] = reward[i] - mean(rewards)

        Args:
            rewards: Raw reward values for each rollout

        Returns:
            Relative advantages
        """
        if not rewards:
            return rewards

        # Compute mean
        mean_reward = sum(rewards) / len(rewards)

        # Compute relative advantages
        advantages = [r - mean_reward for r in rewards]

        # Optional: normalize by std for stability
        if len(rewards) > 1:
            var = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)
            std = max(var ** 0.5, 1.0)  # Use 1.0 as floor to avoid division by small number
            advantages = [a / std for a in advantages]

        return advantages
