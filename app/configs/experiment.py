"""Predefined experiment configurations for the 6 experiments in the methodology."""

from .base import TrainingConfig, RewardConfig, CheckpointConfig
from typing import Dict


def _make_reward_config(use_hcpc: bool, use_d_reason: bool = True) -> RewardConfig:
    """Create reward config with HCPC settings.
       Pass use_d_reason=False for the table-only HCPC ablation."""
    return RewardConfig(
        use_hcpc=use_hcpc,
        use_clc=False,  # CLC disabled for now
        use_process_reward=not use_hcpc,
        # Keep other defaults
        w_type=1.0,
        w_table=2.0,
        w_reason=1.5,
        w_clc=1.0,
        table_sim_threshold=0.6,
        use_d_reason=use_d_reason,
    )


# Shared base config for all experiments — only policy_method, lambda_psr,
# and use_hcpc differ between experiments. This ensures fair comparison.
_SHARED = dict(
    seed=2026,
    learning_rate=1e-6,
    gradient_accumulation_steps=4,
    num_epochs=2,
    batch_size=1,
    warmup_ratio=None,
    weight_decay=None,
    kl_coef=None,
    temperature=0.8,
    top_p=1.0,
    remove_unused_columns=False,
    apply_advantages_in_reward_fn=False,
    image_min_pixels=320 * 28 * 28,
    image_max_pixels=320 * 28 * 28,
    image_resample="bicubic",
    lora_target_modules=["q_proj", "v_proj"],
    torch_dtype_auto=True,
    attn_implementation=None,
    use_flash_attention=False,
    use_python_list_dataset=True,
    wandb_project="chartrl-nsr",
    checkpoint=CheckpointConfig(
        save_every_n_steps=10,
        keep_last_n=3,
        keep_best=False,
    ),
)


# The 6 experiments from the methodology
EXPERIMENTS: Dict[str, TrainingConfig] = {
    # Experiment 1: GRPO baseline (Chart-RVR reproduction)
    "grpo_baseline": TrainingConfig(
        experiment_name="grpo_baseline",
        policy_method="grpo",
        **_SHARED,
        rewards=_make_reward_config(use_hcpc=False),
    ),

    # Experiment 2: GRPO + HCPC
    "grpo_hcpc": TrainingConfig(
        experiment_name="grpo_hcpc",
        policy_method="grpo",
        **_SHARED,
        rewards=_make_reward_config(use_hcpc=True),
    ),

    # Experiment 3: NSR baseline
    "nsr_baseline": TrainingConfig(
        experiment_name="nsr_baseline",
        policy_method="nsr",
        **_SHARED,
        rewards=_make_reward_config(use_hcpc=False),
    ),

    # Experiment 4: NSR + HCPC
    "nsr_hcpc": TrainingConfig(
        experiment_name="nsr_hcpc",
        policy_method="nsr",
        **_SHARED,
        rewards=_make_reward_config(use_hcpc=True),
    ),

    # Experiment 5: W-REINFORCE baseline
    "w_reinforce_baseline": TrainingConfig(
        experiment_name="w_reinforce_baseline",
        policy_method="w_reinforce",
        lambda_psr=0.1,
        **_SHARED,
        rewards=_make_reward_config(use_hcpc=False),
    ),

    # Experiment 6: W-REINFORCE + HCPC (Full HCPC-RLVR)
    "w_reinforce_hcpc": TrainingConfig(
        experiment_name="w_reinforce_hcpc",
        policy_method="w_reinforce",
        lambda_psr=0.1,
        **_SHARED,
        rewards=_make_reward_config(use_hcpc=True),
    ),

    # Ablation: HCPC = C_table only (drop the D_reason term).
    # Same training setup as grpo_hcpc; only the HCPC bonus formula differs.
    "grpo_hcpc_table_only": TrainingConfig(
        experiment_name="grpo_hcpc_table_only",
        policy_method="grpo",
        **_SHARED,
        rewards=_make_reward_config(use_hcpc=True, use_d_reason=False),
    ),
}


def get_experiment_config(name: str, **overrides) -> TrainingConfig:
    """
    Get experiment config by name with optional overrides.

    Args:
        name: Experiment name from EXPERIMENTS
        **overrides: Override any config field

    Returns:
        TrainingConfig with specified settings

    Example:
        config = get_experiment_config("nsr_hcpc", subset_size=1000)
    """
    if name not in EXPERIMENTS:
        available = list(EXPERIMENTS.keys())
        raise ValueError(f"Unknown experiment: {name}. Available: {available}")

    # Get base config
    base_config = EXPERIMENTS[name]

    # Apply overrides
    config_dict = base_config.to_dict()

    for key, value in overrides.items():
        if key == "rewards" and isinstance(value, dict):
            # Handle nested reward config
            for rk, rv in value.items():
                config_dict["rewards"][rk] = rv
        elif key == "checkpoint" and isinstance(value, dict):
            # Handle nested checkpoint config
            for ck, cv in value.items():
                config_dict["checkpoint"][ck] = cv
        else:
            config_dict[key] = value

    return TrainingConfig.from_dict(config_dict)


def list_experiments() -> None:
    """Print all available experiments."""
    print("Available experiments:")
    print("-" * 50)
    for name, config in EXPERIMENTS.items():
        hcpc_str = "HCPC" if config.rewards.use_hcpc else "    "
        print(f"  {name:25} | {config.policy_method:12} | {hcpc_str}")
    print("-" * 50)
