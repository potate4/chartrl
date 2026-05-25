"""Reward aggregation for HCPC-RLVR.

Combines the per-rollout Chart-RVR base rewards with the cross-rollout HCPC
bonus into the total per-rollout reward used by the trainer:

    R_i = R_base(o_i) + B_HCPC * 1[o_i in G+]

where B_HCPC is the group-level HCPC bonus (zero unless |G+| >= 2) and
1[o_i in G+] is the indicator that rollout i is in the correct-path subgroup.
HCPC details are in hcpc_reward.py.
"""

from typing import List, Dict, Any
from dataclasses import dataclass

from .base_rewards import compute_base_rewards
from .hcpc_reward import HCPCComputer


@dataclass
class AggregatedReward:
    """Aggregated reward for a single rollout."""
    total: float
    base: float
    hcpc: float
    breakdown: Dict[str, float]


class RewardAggregator:
    """Aggregates Chart-RVR base rewards and the HCPC bonus."""

    def __init__(
        self,
        use_hcpc: bool = True,
        # HCPC weights
        w_type: float = 1.0,
        w_table: float = 2.0,
        w_reason: float = 1.5,
        table_sim_threshold: float = 0.8,
        use_d_reason: bool = True,
        # Base reward flags
        use_format_reward: bool = True,
        use_accuracy_reward: bool = True,
        use_length_reward: bool = True,
        use_token_count_reward: bool = True,
        use_chart_type_reward: bool = True,
        use_table_reward: bool = True,
        use_process_reward: bool = True,
    ):
        self.use_hcpc = use_hcpc

        # Base reward flags
        self.use_format_reward = use_format_reward
        self.use_accuracy_reward = use_accuracy_reward
        self.use_length_reward = use_length_reward
        self.use_token_count_reward = use_token_count_reward
        self.use_chart_type_reward = use_chart_type_reward
        self.use_table_reward = use_table_reward
        self.use_process_reward = use_process_reward

        self.hcpc_computer = HCPCComputer(
            w_type=w_type,
            w_table=w_table,
            w_reason=w_reason,
            table_sim_threshold=table_sim_threshold,
            use_d_reason=use_d_reason,
        )

    def compute(
        self,
        rollouts: List[str],
        ground_truth: Dict[str, Any],
    ) -> List[AggregatedReward]:
        """Compute aggregated reward for each rollout in a K-sized group."""
        # Per-rollout base rewards (Chart-RVR backbone)
        base_rewards = [
            compute_base_rewards(
                r,
                ground_truth,
                use_format=self.use_format_reward,
                use_accuracy=self.use_accuracy_reward,
                use_length=self.use_length_reward,
                use_token_count=self.use_token_count_reward,
                use_chart_type=self.use_chart_type_reward,
                use_table=self.use_table_reward,
                use_process=self.use_process_reward,
            )
            for r in rollouts
        ]

        # Cross-rollout HCPC bonus (zero on rollouts not in G+)
        hcpc_per_rollout = [0.0] * len(rollouts)
        hcpc_result = None
        if self.use_hcpc:
            hcpc_result = self.hcpc_computer.compute(rollouts, ground_truth)
            hcpc_per_rollout = hcpc_result.per_rollout_rewards

        results = []
        for i, base in enumerate(base_rewards):
            hcpc_i = hcpc_per_rollout[i]
            total = base["total"] + hcpc_i

            breakdown = {**{f"base_{k}": v for k, v in base.items()},
                         "hcpc": hcpc_i}

            if hcpc_result is not None:
                breakdown.update({
                    "hcpc_c_type": hcpc_result.c_type,
                    "hcpc_c_table": hcpc_result.c_table,
                    "hcpc_d_reason": hcpc_result.d_reason,
                    "hcpc_correct_rate": hcpc_result.correct_rate,
                    "hcpc_num_correct": float(hcpc_result.num_correct),
                    "hcpc_num_rollouts": float(len(rollouts)),
                })

            results.append(AggregatedReward(
                total=total,
                base=base["total"],
                hcpc=hcpc_i,
                breakdown=breakdown,
            ))

        return results

    def compute_rewards_tensor(
        self,
        rollouts: List[str],
        ground_truth: Dict[str, Any],
    ) -> List[float]:
        results = self.compute(rollouts, ground_truth)
        return [r.total for r in results]

    @classmethod
    def from_config(cls, config) -> "RewardAggregator":
        # When HCPC is on, process-conformity (GT-similarity) reward is
        # disabled because HCPC's D_reason rewards reasoning diversity, which
        # is in tension with the GT-similarity signal.
        use_process = config.rewards.use_process_reward
        if config.rewards.use_hcpc and use_process:
            use_process = False

        return cls(
            use_hcpc=config.rewards.use_hcpc,
            w_type=config.rewards.w_type,
            w_table=config.rewards.w_table,
            w_reason=config.rewards.w_reason,
            table_sim_threshold=config.rewards.table_sim_threshold,
            use_d_reason=config.rewards.use_d_reason,
            use_format_reward=config.rewards.use_format_reward,
            use_accuracy_reward=config.rewards.use_accuracy_reward,
            use_length_reward=config.rewards.use_length_reward,
            use_token_count_reward=config.rewards.use_token_count_reward,
            use_chart_type_reward=config.rewards.use_chart_type_reward,
            use_table_reward=config.rewards.use_table_reward,
            use_process_reward=use_process,
        )


def compute_total_rewards(
    rollouts: List[str],
    ground_truth: Dict[str, Any],
    use_hcpc: bool = True,
    **kwargs,
) -> List[float]:
    aggregator = RewardAggregator(use_hcpc=use_hcpc, **kwargs)
    return aggregator.compute_rewards_tensor(rollouts, ground_truth)


def create_reward_function(config):
    """Create the reward function for the trainer.

    Returns a callable that takes completions and ground-truth fields and
    returns the per-rollout reward list.
    """
    aggregator = RewardAggregator.from_config(config)

    def reward_fn(
        completions: List[str],
        labels: List[str],
        tables: List[Dict] = None,
        chart_types: List[str] = None,
        reasonings: List[str] = None,
        **kwargs,
    ) -> List[float]:
        ground_truth = {
            "label": labels[0] if labels else "",
            "table": tables[0] if tables else {},
            "chart_type": chart_types[0] if chart_types else "",
            "reasoning": reasonings[0] if reasonings else "",
        }
        return aggregator.compute_rewards_tensor(completions, ground_truth)

    return reward_fn
