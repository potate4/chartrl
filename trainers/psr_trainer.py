"""
PSR (Positive Sample Reinforcement) Trainer

Trains only on correct samples (reward = +1 or positive rewards).
Maximizes likelihood of correct responses.

Based on: "Decomposed Reinforcement Learning from Verifiable Feedback (RLVR)"
Objective: L_PSR(θ) = -E[Σ_{y:r(x,y)=1} log π_θ(y|x)]

Expected behavior:
- High Pass@1 (best single response)
- Lower Pass@k at large k (reduced diversity)
"""

import logging
from typing import List, Dict, Any
from datasets import Dataset


def filter_positive_samples(dataset: Dataset, reward_threshold: float = 0.5) -> Dataset:
    """
    Filter dataset to keep only positive samples (correct responses).

    Args:
        dataset: Training dataset with reward information
        reward_threshold: Minimum reward to consider sample as positive (default: 0.5)

    Returns:
        Filtered dataset containing only positive samples
    """

    def is_positive(example):
        """Check if sample has positive reward"""
        # During GRPO generation, samples are scored with rewards
        # We want to keep only those with high reward (correct responses)

        # Note: In the GRPO dataset format, each example represents a problem
        # with multiple completions. We'll filter at the completion level during training.
        return True  # Return all examples, filtering happens in training loop

    # For PSR, we don't pre-filter the dataset because GRPO generates
    # samples during training. Instead, we'll filter in the reward function.
    logging.info("PSR Trainer: Will train only on positive samples (high reward)")
    logging.info(f"  Reward threshold: >= {reward_threshold}")

    return dataset


def psr_reward_filter(rewards: List[float], reward_threshold: float = 0.5) -> List[float]:
    """
    Filter rewards for PSR training - zero out negative samples.

    Args:
        rewards: List of rewards for generated samples
        reward_threshold: Minimum reward to consider as positive

    Returns:
        Filtered rewards (negative samples set to 0)
    """
    # PSR: Only learn from positive samples
    # Set negative rewards to 0 (no gradient update)
    filtered_rewards = [
        r if r >= reward_threshold else 0.0
        for r in rewards
    ]

    positive_count = sum(1 for r in filtered_rewards if r > 0)
    total_count = len(filtered_rewards)

    logging.debug(f"PSR: {positive_count}/{total_count} positive samples")

    return filtered_rewards


def get_psr_config(base_config: Dict[str, Any], reward_threshold: float = 0.5) -> Dict[str, Any]:
    """
    Get PSR-specific training configuration.

    Args:
        base_config: Base GRPO configuration
        reward_threshold: Reward threshold for positive samples

    Returns:
        Updated configuration for PSR training
    """
    psr_config = base_config.copy()

    # PSR-specific settings
    psr_config['training_mode'] = 'psr'
    psr_config['reward_threshold'] = reward_threshold

    # PSR may benefit from slightly lower learning rate
    # since we're only learning from correct samples (more stable)
    if 'learning_rate' in psr_config:
        psr_config['learning_rate'] = psr_config['learning_rate'] * 0.8
        logging.info(f"PSR: Reduced learning rate to {psr_config['learning_rate']}")

    logging.info("=" * 80)
    logging.info("PSR TRAINING MODE ENABLED")
    logging.info("  Training only on POSITIVE samples (correct responses)")
    logging.info(f"  Reward threshold: {reward_threshold}")
    logging.info("  Expected: High Pass@1, Lower Pass@k at large k")
    logging.info("=" * 80)

    return psr_config


# For integration with main.py
class PSRTrainingMixin:
    """
    Mixin to add PSR training behavior to GRPO trainer.

    Usage:
        Apply psr_reward_filter() to rewards before computing loss.
    """

    def __init__(self, reward_threshold: float = 0.5):
        self.reward_threshold = reward_threshold
        self.training_mode = 'psr'

    def filter_rewards(self, rewards: List[float]) -> List[float]:
        """Filter rewards for PSR (only positive samples)"""
        return psr_reward_filter(rewards, self.reward_threshold)
