"""Trainer implementations for HCPC-RLVR."""

from .base_trainer import BaseTrainer
from .grpo_trainer import GRPOTrainer
from .nsr_trainer import NSRTrainer
from .w_reinforce_trainer import WREINFORCETrainer

__all__ = [
    "BaseTrainer",
    "GRPOTrainer",
    "NSRTrainer",
    "WREINFORCETrainer",
]


def get_trainer(policy_method: str):
    """
    Get trainer class by policy method name.

    Args:
        policy_method: One of 'grpo', 'nsr', 'w_reinforce'

    Returns:
        Trainer class
    """
    trainers = {
        "grpo": GRPOTrainer,
        "nsr": NSRTrainer,
        "w_reinforce": WREINFORCETrainer,
    }

    if policy_method not in trainers:
        raise ValueError(
            f"Unknown policy method: {policy_method}. "
            f"Available: {list(trainers.keys())}"
        )

    return trainers[policy_method]
