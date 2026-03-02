"""
NSR (Negative Sample Reinforcement) Trainer

Trains only on incorrect samples (reward = -1 or negative rewards).
Minimizes likelihood of incorrect responses.

Based on: "Decomposed Reinforcement Learning from Verifiable Feedback (RLVR)"
Objective: L_NSR(θ) = E[Σ_{y:r(x,y)=-1} log π_θ(y|x)] (minimize likelihood of negatives)

Expected behavior:
- High Pass@k across all k (preserves diversity)
- Matches GRPO Pass@1 (surprising result from paper!)
- Redistributes probability mass away from known errors
"""

import logging
from typing import List, Dict, Any
from datasets import Dataset


def filter_negative_samples(dataset: Dataset, reward_threshold: float = 0.5) -> Dataset:
    """
    Filter dataset to keep only negative samples (incorrect responses).

    Args:
        dataset: Training dataset with reward information
        reward_threshold: Maximum reward to consider sample as negative (default: 0.5)

    Returns:
        Filtered dataset containing only negative samples
    """

    def is_negative(example):
        """Check if sample has negative reward"""
        # During GRPO generation, samples are scored with rewards
        # We want to keep only those with low reward (incorrect responses)

        # Note: In the GRPO dataset format, each example represents a problem
        # with multiple completions. We'll filter at the completion level during training.
        return True  # Return all examples, filtering happens in training loop

    # For NSR, we don't pre-filter the dataset because GRPO generates
    # samples during training. Instead, we'll filter in the reward function.
    logging.info("NSR Trainer: Will train only on negative samples (low reward)")
    logging.info(f"  Reward threshold: < {reward_threshold}")

    return dataset


def nsr_reward_filter(rewards: List[float], reward_threshold: float = 0.5) -> List[float]:
    """
    Filter rewards for NSR training - zero out positive samples, invert negative samples.

    Args:
        rewards: List of rewards for generated samples
        reward_threshold: Maximum reward to consider as negative

    Returns:
        Filtered rewards (positive samples set to 0, negative samples inverted)
    """
    # NSR: Only learn from negative samples
    # Invert the reward sign for negative samples (to minimize their likelihood)
    # Set positive rewards to 0 (no gradient update)

    filtered_rewards = []
    for r in rewards:
        if r < reward_threshold:
            # Negative sample: invert reward to minimize likelihood
            # The magnitude should be negative to push probability down
            filtered_rewards.append(-abs(r - reward_threshold))
        else:
            # Positive sample: ignore (no gradient)
            filtered_rewards.append(0.0)

    negative_count = sum(1 for r in filtered_rewards if r < 0)
    total_count = len(filtered_rewards)

    logging.debug(f"NSR: {negative_count}/{total_count} negative samples")

    return filtered_rewards


def get_nsr_config(base_config: Dict[str, Any], reward_threshold: float = 0.5) -> Dict[str, Any]:
    """
    Get NSR-specific training configuration.

    Args:
        base_config: Base GRPO configuration
        reward_threshold: Reward threshold for negative samples

    Returns:
        Updated configuration for NSR training
    """
    nsr_config = base_config.copy()

    # NSR-specific settings
    nsr_config['training_mode'] = 'nsr'
    nsr_config['reward_threshold'] = reward_threshold

    # NSR may benefit from slightly higher learning rate
    # since negative signals need stronger weight to redistribute probability mass
    if 'learning_rate' in nsr_config:
        nsr_config['learning_rate'] = nsr_config['learning_rate'] * 1.2
        logging.info(f"NSR: Increased learning rate to {nsr_config['learning_rate']}")

    # Increase num_generations to get more negative samples
    if 'num_generations' in nsr_config and nsr_config['num_generations'] < 4:
        logging.warning(f"NSR: num_generations={nsr_config['num_generations']} may be too low.")
        logging.warning("  Recommended: >= 4 to get sufficient negative samples")

    logging.info("=" * 80)
    logging.info("NSR TRAINING MODE ENABLED")
    logging.info("  Training only on NEGATIVE samples (incorrect responses)")
    logging.info(f"  Reward threshold: < {reward_threshold}")
    logging.info("  Expected: High Pass@k across all k, preserves diversity")
    logging.info("=" * 80)

    return nsr_config


# For integration with main.py
class NSRTrainingMixin:
    """
    Mixin to add NSR training behavior to GRPO trainer.

    Usage:
        Apply nsr_reward_filter() to rewards before computing loss.
    """

    def __init__(self, reward_threshold: float = 0.5):
        self.reward_threshold = reward_threshold
        self.training_mode = 'nsr'

    def filter_rewards(self, rewards: List[float]) -> List[float]:
        """Filter rewards for NSR (only negative samples, inverted)"""
        return nsr_reward_filter(rewards, self.reward_threshold)
