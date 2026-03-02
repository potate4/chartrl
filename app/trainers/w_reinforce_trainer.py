"""
Weighted REINFORCE (W-REINFORCE) Trainer.

W-REINFORCE is a hybrid of PSR and NSR:
- Weak positive reinforcement for correct samples: λ × reward
- Strong negative reinforcement for wrong samples: -(threshold - reward)

The key insight is that λ is small (0.1), so:
- We still boost correct answers, maintaining accuracy
- But the boost is weak, preserving diversity
- Wrong answers are strongly penalized (full strength)

This gives the best of both worlds:
- Better Pass@1 than pure NSR (some positive signal)
- Better Pass@k than GRPO (weak positive preserves diversity)

Paper recommends λ = 0.1
"""

from typing import List

from .base_trainer import BaseTrainer, PolicyMethodMixin


class WREINFORCETrainer(BaseTrainer, PolicyMethodMixin):
    """
    Weighted REINFORCE trainer.

    Combines weak positive reinforcement with strong negative reinforcement.

    λ·PSR + NSR where λ = 0.1 (configurable via lambda_psr)
    """

    def compute_advantages(self, rewards: List[float]) -> List[float]:
        """
        Compute W-REINFORCE advantages.

        For correct samples: advantage = λ × reward (weak boost)
        For wrong samples: advantage = -(threshold - reward) (strong penalty)

        Args:
            rewards: Raw reward values for each rollout

        Returns:
            W-REINFORCE advantages
        """
        if not rewards:
            return rewards

        # Get parameters
        lambda_psr = self.config.lambda_psr  # Default 0.1
        max_reward = self.get_max_reward(self.config)
        threshold = self.config.reward_threshold * max_reward

        advantages = []

        for reward in rewards:
            if reward >= threshold:
                # Correct sample: weak positive reinforcement
                advantage = lambda_psr * reward
            else:
                # Wrong sample: strong negative reinforcement
                advantage = -(threshold - reward)

            advantages.append(advantage)

        return advantages
