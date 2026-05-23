"""Trainer implementations for HCPC-RLVR."""

from .base_trainer import BaseTrainer
from .grpo_trainer import GRPOTrainer
from .nsr_masked_trainer import NSRMaskedTrainer

__all__ = [
    "BaseTrainer",
    "GRPOTrainer",
    "NSRMaskedTrainer",
]


def get_trainer(policy_method: str, use_hcpc: bool = False):
    """Return the trainer class for the requested advantage estimator.

    Args:
        policy_method: 'grpo' or 'nsr'.
        use_hcpc: kept for backward compatibility; routing is the same
            either way since HCPC is implemented as a reward bonus, not
            a trainer variant.
    """
    trainers = {
        "grpo": GRPOTrainer,
        "nsr": NSRMaskedTrainer,
    }
    if policy_method not in trainers:
        raise ValueError(
            f"Unknown policy method: {policy_method}. "
            f"Available: {list(trainers.keys())}"
        )
    return trainers[policy_method]
