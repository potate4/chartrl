"""
Weighted-REINFORCE (W-REINFORCE) Trainer

Combines PSR and NSR with weighting: λ·L_PSR + L_NSR

Based on: "Decomposed Reinforcement Learning from Verifiable Feedback (RLVR)"
Objective: L_W-REINFORCE(θ) = λ·L_PSR(θ) + L_NSR(θ)

Recommended: λ = 0.1 (from paper)

Expected behavior:
- Best Pass@1 (from 10% PSR weight)
- High Pass@k (from 100% NSR weight)
- Best overall performance across all k values
"""

import logging
from typing import List, Dict, Any
from datasets import Dataset


def weighted_reinforce_reward_filter(
    rewards: List[float],
    lambda_psr: float = 0.1,
    reward_threshold: float = 0.5
) -> List[float]:
    """
    Apply Weighted-REINFORCE reward weighting.

    Args:
        rewards: List of rewards for generated samples
        lambda_psr: Weight for PSR component (default: 0.1)
        reward_threshold: Threshold to separate positive/negative samples

    Returns:
        Weighted rewards: λ·r for positive, -r for negative
    """
    # W-REINFORCE: Weight positive and negative samples differently
    # Positive samples: λ·r (reduced weight)
    # Negative samples: -|r-threshold| (full weight, inverted)

    weighted_rewards = []
    for r in rewards:
        if r >= reward_threshold:
            # Positive sample: apply λ weight
            weighted_rewards.append(lambda_psr * r)
        else:
            # Negative sample: full weight, inverted
            weighted_rewards.append(-abs(r - reward_threshold))

    positive_count = sum(1 for r in rewards if r >= reward_threshold)
    negative_count = sum(1 for r in rewards if r < reward_threshold)
    total_count = len(rewards)

    logging.debug(f"W-REINFORCE: {positive_count} positive (λ={lambda_psr}), "
                 f"{negative_count} negative (λ=1.0) / {total_count} total")

    return weighted_rewards


def get_weighted_reinforce_config(
    base_config: Dict[str, Any],
    lambda_psr: float = 0.1,
    reward_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Get Weighted-REINFORCE training configuration.

    Args:
        base_config: Base GRPO configuration
        lambda_psr: Weight for PSR component (default: 0.1)
        reward_threshold: Threshold to separate positive/negative samples

    Returns:
        Updated configuration for W-REINFORCE training
    """
    w_reinforce_config = base_config.copy()

    # W-REINFORCE specific settings
    w_reinforce_config['training_mode'] = 'w-reinforce'
    w_reinforce_config['lambda_psr'] = lambda_psr
    w_reinforce_config['reward_threshold'] = reward_threshold

    # Validate lambda
    if not 0.0 <= lambda_psr <= 1.0:
        logging.warning(f"W-REINFORCE: lambda_psr={lambda_psr} outside [0, 1] range")
        logging.warning("  Recommended range: 0.05 to 0.2")

    # Log recommended settings
    if lambda_psr != 0.1:
        logging.info(f"W-REINFORCE: Using λ={lambda_psr} (paper recommends 0.1)")

    # Ensure sufficient generations to get both positive and negative samples
    if 'num_generations' in w_reinforce_config and w_reinforce_config['num_generations'] < 4:
        logging.warning(f"W-REINFORCE: num_generations={w_reinforce_config['num_generations']} may be too low.")
        logging.warning("  Recommended: >= 4 to get mix of positive and negative samples")

    logging.info("=" * 80)
    logging.info("WEIGHTED-REINFORCE TRAINING MODE ENABLED")
    logging.info(f"  PSR weight (λ): {lambda_psr}")
    logging.info(f"  NSR weight: 1.0")
    logging.info(f"  Objective: {lambda_psr}·L_PSR + L_NSR")
    logging.info(f"  Reward threshold: {reward_threshold}")
    logging.info("  Expected: Best Pass@1 and Pass@k across all k")
    logging.info("=" * 80)

    return w_reinforce_config


def suggest_lambda_values() -> List[float]:
    """
    Suggest lambda values for hyperparameter search.

    Returns:
        List of recommended lambda values to try
    """
    return [0.05, 0.1, 0.15, 0.2, 0.5]


# For integration with main.py
class WeightedREINFORCETrainingMixin:
    """
    Mixin to add W-REINFORCE training behavior to GRPO trainer.

    Usage:
        Apply weighted_reinforce_reward_filter() to rewards before computing loss.
    """

    def __init__(self, lambda_psr: float = 0.1, reward_threshold: float = 0.5):
        self.lambda_psr = lambda_psr
        self.reward_threshold = reward_threshold
        self.training_mode = 'w-reinforce'

    def filter_rewards(self, rewards: List[float]) -> List[float]:
        """Apply Weighted-REINFORCE reward weighting"""
        return weighted_reinforce_reward_filter(
            rewards,
            self.lambda_psr,
            self.reward_threshold
        )
