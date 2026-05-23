"""Predefined experiment configurations matching the paper's main results.

Five configurations cover the rows of Table 1:
    Base         : evaluation only, no training
    GRPO         : Chart-RVR backbone with GRPO advantage rule
    GRPO+HCPC    : GRPO with the HCPC cross-rollout bonus
    NSR          : Chart-RVR backbone with NSR advantage rule
    NSR+HCPC     : NSR with the HCPC cross-rollout bonus

All training runs share the hyperparameters listed in the paper's
Experimental Setup; only `policy_method` and `use_hcpc` differ.
"""

from .base import TrainingConfig, RewardConfig, CheckpointConfig
from typing import Dict


def _make_reward_config(use_hcpc: bool) -> RewardConfig:
    """RewardConfig matching the paper:
       HCPC: w_table=2.0, w_reason=1.5, tau=0.8.
       When HCPC is on, the Chart-RVR process-conformity reward is disabled
       (HCPC's D_reason rewards diversity, in tension with process-conformity)."""
    return RewardConfig(
        use_hcpc=use_hcpc,
        use_process_reward=not use_hcpc,
        w_table=2.0,
        w_reason=1.5,
        table_sim_threshold=0.8,
    )


# Hyperparameters shared by all training runs. Matches Section 4.1 of the paper.
_SHARED = dict(
    seed=2026,
    learning_rate=1e-5,          # paper value
    gradient_accumulation_steps=2,
    batch_size=2,                # effective batch size = 4
    num_epochs=2,                # ~2000 optimization steps on a 1K-sample subset
    warmup_ratio=None,
    weight_decay=None,
    kl_coef=None,                # no KL penalty (paper)
    beta=0.0,
    num_generations=4,           # K=4 rollouts per prompt
    temperature=1.0,             # training temperature
    top_p=1.0,
    reward_threshold=0.5,        # NSR threshold = 0.5 * R_max
    remove_unused_columns=False,
    apply_advantages_in_reward_fn=False,
    image_min_pixels=320 * 28 * 28,
    image_max_pixels=320 * 28 * 28,
    image_resample="bicubic",
    lora_target_modules=["q_proj", "v_proj"],  # LoRA on W_q, W_v
    lora_r=8,
    lora_alpha=16,
    torch_dtype_auto=True,
    attn_implementation=None,
    use_flash_attention=False,
    use_python_list_dataset=True,
    subset_size=1000,            # 1K training subset
    wandb_project="chartrl-cross-rollout",
    checkpoint=CheckpointConfig(
        save_every_n_steps=200,
        keep_last_n=3,
        keep_best=False,
    ),
)


EXPERIMENTS: Dict[str, TrainingConfig] = {
    "grpo_baseline": TrainingConfig(
        experiment_name="grpo_baseline",
        policy_method="grpo",
        **_SHARED,
        rewards=_make_reward_config(use_hcpc=False),
    ),
    "grpo_hcpc": TrainingConfig(
        experiment_name="grpo_hcpc",
        policy_method="grpo",
        **_SHARED,
        rewards=_make_reward_config(use_hcpc=True),
    ),
    "nsr_baseline": TrainingConfig(
        experiment_name="nsr_baseline",
        policy_method="nsr",
        **_SHARED,
        rewards=_make_reward_config(use_hcpc=False),
    ),
    "nsr_hcpc": TrainingConfig(
        experiment_name="nsr_hcpc",
        policy_method="nsr",
        **_SHARED,
        rewards=_make_reward_config(use_hcpc=True),
    ),
}


def get_experiment_config(name: str, **overrides) -> TrainingConfig:
    if name not in EXPERIMENTS:
        available = list(EXPERIMENTS.keys())
        raise ValueError(f"Unknown experiment: {name}. Available: {available}")

    base_config = EXPERIMENTS[name]
    config_dict = base_config.to_dict()

    for key, value in overrides.items():
        if key == "rewards" and isinstance(value, dict):
            for rk, rv in value.items():
                config_dict["rewards"][rk] = rv
        elif key == "checkpoint" and isinstance(value, dict):
            for ck, cv in value.items():
                config_dict["checkpoint"][ck] = cv
        else:
            config_dict[key] = value

    return TrainingConfig.from_dict(config_dict)


def list_experiments() -> None:
    print("Available experiments:")
    print("-" * 50)
    for name, config in EXPERIMENTS.items():
        hcpc_str = "HCPC" if config.rewards.use_hcpc else "    "
        print(f"  {name:25} | {config.policy_method:12} | {hcpc_str}")
    print("-" * 50)
