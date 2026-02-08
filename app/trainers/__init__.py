"""Trainer implementations for HCPC-RLVR."""

from .base_trainer import BaseTrainer
from .grpo_trainer import GRPOTrainer
from .nsr_trainer import NSRTrainer
from .w_reinforce_trainer import WREINFORCETrainer
from .nsr_hcpc_trainer import NSRHCPCTrainer

__all__ = [
    "BaseTrainer",
    "GRPOTrainer",
    "NSRTrainer",
    "WREINFORCETrainer",
    "NSRHCPCTrainer",
]


def get_trainer(policy_method: str, use_hcpc: bool = False):
    """
    Get trainer class by policy method name.

    Args:
        policy_method: One of 'grpo', 'nsr', 'w_reinforce'
        use_hcpc: Whether to use HCPC variant (for NSR)

    Returns:
        Trainer class
    """
    # Special case: NSR with HCPC uses penalty-based trainer
    if policy_method == "nsr" and use_hcpc:
        return NSRHCPCTrainer

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
