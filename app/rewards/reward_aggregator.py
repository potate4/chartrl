"""
Reward aggregation for HCPC-RLVR.

Combines base rewards (from Chart-RVR), HCPC, and CLC into total rewards.
"""

from typing import List, Dict, Any
from dataclasses import dataclass

from .base_rewards import compute_base_rewards
from .hcpc_reward import HCPCComputer, HCPCResult
from .clc_reward import CLCComputer, CLCResult


@dataclass
class AggregatedReward:
    """Aggregated reward for a single rollout."""
    total: float
    base: float
    hcpc: float
    clc: float
    breakdown: Dict[str, float]


class RewardAggregator:
    """
    Aggregates all reward components for GRPO training.

    Total reward for each rollout:
        R_total[i] = R_base[i] + R_HCPC + R_CLC[i]

    Note: R_HCPC is a group-level reward (same for all rollouts).
          R_base and R_CLC are per-rollout.
    """

    def __init__(
        self,
        use_hcpc: bool = True,
        use_clc: bool = True,
        # HCPC weights
        w_type: float = 1.0,
        w_table: float = 2.0,
        w_reason: float = 1.5,
        table_sim_threshold: float = 0.8,
        # CLC weight
        w_clc: float = 1.0,
        # Base reward flags
        use_format_reward: bool = True,
        use_accuracy_reward: bool = True,
        use_length_reward: bool = True,
        use_token_count_reward: bool = True,
        use_chart_type_reward: bool = True,
        use_table_reward: bool = True,
        use_process_reward: bool = True,
    ):
        """
        Initialize reward aggregator.

        Args:
            use_hcpc: Whether to use HCPC reward
            use_clc: Whether to use CLC reward
            w_type: HCPC weight for type consistency
            w_table: HCPC weight for table consistency
            w_reason: HCPC weight for reasoning diversity
            table_sim_threshold: HCPC threshold for table matching
            w_clc: CLC reward weight
            use_process_reward: Whether to use process reward (GT reasoning similarity)
                                Set to False when using HCPC (promotes diversity instead)
        """
        self.use_hcpc = use_hcpc
        self.use_clc = use_clc

        # Base reward flags
        self.use_format_reward = use_format_reward
        self.use_accuracy_reward = use_accuracy_reward
        self.use_length_reward = use_length_reward
        self.use_token_count_reward = use_token_count_reward
        self.use_chart_type_reward = use_chart_type_reward
        self.use_table_reward = use_table_reward
        self.use_process_reward = use_process_reward

        # Initialize computers
        self.hcpc_computer = HCPCComputer(
            w_type=w_type,
            w_table=w_table,
            w_reason=w_reason,
            table_sim_threshold=table_sim_threshold,
        )
        self.clc_computer = CLCComputer(w_clc=w_clc)

    def compute(
        self,
        rollouts: List[str],
        ground_truth: Dict[str, Any],
    ) -> List[AggregatedReward]:
        """
        Compute aggregated rewards for all rollouts.

        Args:
            rollouts: List of K model outputs
            ground_truth: Dict with label, table, chart_type, reasoning

        Returns:
            List of AggregatedReward, one per rollout
        """
        results = []

        # Compute base rewards for each rollout
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

        # Compute HCPC (per-rollout: only correct rollouts get bonus)
        hcpc_per_rollout = [0.0] * len(rollouts)
        hcpc_result = None
        if self.use_hcpc:
            hcpc_result = self.hcpc_computer.compute(rollouts, ground_truth)
            hcpc_per_rollout = hcpc_result.per_rollout_rewards

        # Compute CLC for each rollout
        clc_results = []
        if self.use_clc:
            clc_results = self.clc_computer.compute_batch(rollouts)
        else:
            # Create dummy results with 0 reward
            clc_results = [CLCResult(0, 0, set(), set(), set(), {}) for _ in rollouts]

        # Aggregate
        for i, (base, clc) in enumerate(zip(base_rewards, clc_results)):
            hcpc_i = hcpc_per_rollout[i]
            total = base["total"] + hcpc_i + clc.reward

            breakdown = {
                **{f"base_{k}": v for k, v in base.items()},
                "hcpc": hcpc_i,
                "clc": clc.reward,
                "clc_coherence": clc.coherence,
            }

            if hcpc_result:
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
                clc=clc.reward,
                breakdown=breakdown,
            ))

        return results

    def compute_rewards_tensor(
        self,
        rollouts: List[str],
        ground_truth: Dict[str, Any],
    ) -> List[float]:
        """
        Compute rewards as a simple list of floats.

        Convenience method for GRPO training.

        Args:
            rollouts: List of model outputs
            ground_truth: Ground truth dict

        Returns:
            List of total reward values
        """
        results = self.compute(rollouts, ground_truth)
        return [r.total for r in results]

    @classmethod
    def from_config(cls, config) -> "RewardAggregator":
        """
        Create aggregator from config.

        Args:
            config: TrainingConfig instance

        Returns:
            RewardAggregator
        """
        # When using HCPC, disable process_reward by default
        # (HCPC promotes diversity, process_reward promotes GT similarity - contradictory)
        use_process = config.rewards.use_process_reward
        if config.rewards.use_hcpc and use_process:
            # Auto-disable if HCPC is on (unless explicitly set)
            # Check if it was explicitly set in config or just default
            use_process = False

        return cls(
            use_hcpc=config.rewards.use_hcpc,
            use_clc=config.rewards.use_clc,
            w_type=config.rewards.w_type,
            w_table=config.rewards.w_table,
            w_reason=config.rewards.w_reason,
            table_sim_threshold=config.rewards.table_sim_threshold,
            w_clc=config.rewards.w_clc,
            # Base reward flags
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
    use_clc: bool = True,
    **kwargs,
) -> List[float]:
    """
    Convenience function to compute total rewards.

    Args:
        rollouts: List of model outputs
        ground_truth: Ground truth dict
        use_hcpc: Whether to use HCPC
        use_clc: Whether to use CLC
        **kwargs: Additional arguments for RewardAggregator

    Returns:
        List of total reward values
    """
    aggregator = RewardAggregator(
        use_hcpc=use_hcpc,
        use_clc=use_clc,
        **kwargs,
    )
    return aggregator.compute_rewards_tensor(rollouts, ground_truth)


def create_reward_function(config):
    """
    Create reward function for GRPO trainer.

    Returns a callable that takes completions and returns rewards.

    Args:
        config: TrainingConfig

    Returns:
        Reward function
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
        """
        Compute rewards for a batch of completions.

        This is called by GRPO trainer for each group of rollouts.
        """
        # Build ground truth from inputs
        ground_truth = {
            "label": labels[0] if labels else "",
            "table": tables[0] if tables else {},
            "chart_type": chart_types[0] if chart_types else "",
            "reasoning": reasonings[0] if reasonings else "",
        }

        return aggregator.compute_rewards_tensor(completions, ground_truth)

    return reward_fn
