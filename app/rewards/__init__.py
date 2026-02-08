"""Reward functions for HCPC-RLVR training."""

from .base_rewards import (
    format_reward,
    accuracy_reward,
    length_reward,
    token_count_reward,
    chart_type_reward,
    table_reward,
    process_reward,
    compute_base_rewards,
)
from .hcpc_reward import compute_hcpc_reward, HCPCComputer
from .clc_reward import compute_clc_reward, CLCComputer
from .reward_aggregator import RewardAggregator, compute_total_rewards

__all__ = [
    # Base rewards
    "format_reward",
    "accuracy_reward",
    "length_reward",
    "token_count_reward",
    "chart_type_reward",
    "table_reward",
    "process_reward",
    "compute_base_rewards",
    # HCPC
    "compute_hcpc_reward",
    "HCPCComputer",
    # CLC
    "compute_clc_reward",
    "CLCComputer",
    # Aggregator
    "RewardAggregator",
    "compute_total_rewards",
]
