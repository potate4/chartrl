"""
Negative Sample Reinforcement (NSR) Trainer.

From the NSR paper: "The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning"

NSR modifies GRPO by:
1. SKIPPING correct samples (no gradient)
2. Only PENALIZING wrong samples

Why this works:
- By not reinforcing correct answers, NSR doesn't push toward any single strategy
- It only pushes AWAY from wrong strategies
- Probability mass redistributes naturally across all valid approaches
- Preserves diversity and exploration capacity

Key finding from the paper:
- NSR achieves comparable Pass@1 to GRPO
- NSR significantly outperforms GRPO at Pass@k for large k
- NSR maintains base model's entropy (diversity)
"""

from typing import List

from .base_trainer import BaseTrainer, PolicyMethodMixin


class NSRTrainer(BaseTrainer, PolicyMethodMixin):
    """
    Negative Sample Reinforcement trainer.

    Only penalizes wrong samples, skips correct ones entirely.

    This preserves diversity by not reinforcing any particular correct strategy.
    """

    def compute_advantages(self, rewards: List[float]) -> List[float]:
        """
        Compute NSR advantages.

        For correct samples (reward >= threshold): advantage = 0 (skip)
        For wrong samples (reward < threshold): advantage = -(threshold - reward)

        Args:
            rewards: Raw reward values for each rollout

        Returns:
            NSR advantages
        """
        if not rewards:
            return rewards

        # Get threshold for correct/wrong classification
        # Use a fraction of max reward
        max_reward = self.get_max_reward(self.config)
        threshold = self.config.reward_threshold * max_reward

        advantages = []

        for reward in rewards:
            if reward >= threshold:
                # Correct sample: skip (no gradient)
                advantages.append(0.0)
            else:
                # Wrong sample: penalize
                # Penalty proportional to how wrong it is
                penalty = -(threshold - reward)
                advantages.append(penalty)

        return advantages
