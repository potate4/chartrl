"""Base configuration dataclasses for HCPC-RLVR training."""

from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


@dataclass
class CheckpointConfig:
    """Checkpoint management settings."""

    # Save frequency
    save_every_n_steps: int = 50

    # Retention policy
    keep_last_n: int = 5
    keep_best: bool = True

    # Resume settings
    resume: bool = False
    resume_from: Optional[str] = None
    from_scratch: bool = False

    # Paths
    output_base: str = "./outputs"


@dataclass
class RewardConfig:
    """Reward function weights and settings."""

    # Base rewards (from Chart-RVR)
    use_format_reward: bool = True
    use_accuracy_reward: bool = True
    use_length_reward: bool = True
    use_token_count_reward: bool = True
    use_chart_type_reward: bool = True
    use_table_reward: bool = True
    use_process_reward: bool = True

    # HCPC reward settings
    use_hcpc: bool = True
    w_type: float = 1.0
    w_table: float = 2.0
    w_reason: float = 1.5
    table_sim_threshold: float = 0.8

    # CLC reward settings (disabled for now)
    use_clc: bool = False
    w_clc: float = 1.0


@dataclass
class TrainingConfig:
    """Main training configuration."""

    # Experiment identity
    experiment_name: str = "default"
    seed: int = 42

    # Model settings
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    use_flash_attention: bool = True
    attn_implementation: Optional[str] = None
    torch_dtype_auto: bool = False

    # LoRA settings
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "up_proj", "down_proj", "gate_proj"
    ])

    # Training hyperparameters
    num_epochs: int = 4
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-6
    warmup_ratio: Optional[float] = 0.03
    weight_decay: Optional[float] = 0.01
    max_grad_norm: float = 1.0

    # GRPO specific
    num_generations: int = 4  # K rollouts per sample
    max_prompt_length: int = 4096
    max_completion_length: int = 768
    temperature: float = 0.8
    top_p: float = 0.95

    # Policy update method
    policy_method: str = "grpo"  # grpo, nsr, w_reinforce
    lambda_psr: float = 0.1  # For W-REINFORCE
    reward_threshold: float = 0.5  # For NSR/W-REINFORCE correct/wrong split

    # KL penalty
    kl_coef: Optional[float] = 0.01
    beta: float = 0.0

    # Dataset
    dataset_name: str = "sanchit97/chart-rvr-grpo-train"
    eval_dataset_name: str = "lmms-lab/EvoChart"
    subset_size: Optional[int] = None  # For quick iteration

    # Precision
    bf16: bool = True
    gradient_checkpointing: bool = True

    # Logging
    logging_steps: int = 10
    logging_first_step: bool = True
    use_wandb: bool = True
    wandb_project: str = "hcpc-rlvr"
    log_completions: bool = True
    log_completions_every: int = 1
    log_completions_max: int = -1
    log_completions_max_chars: int = 0
    log_completions_to_file: bool = False
    log_rewards: bool = True
    log_rewards_every: int = 1
    log_rewards_to_file: bool = False
    log_metrics_every: int = 10
    log_metrics_window: int = 50

    # TRL / dataset compatibility
    remove_unused_columns: bool = False
    apply_advantages_in_reward_fn: bool = False
    use_python_list_dataset: bool = False
    save_total_limit: int = 3

    # Image preprocessing
    image_min_pixels: int = 4 * 28 * 28
    image_max_pixels: int = 512 * 28 * 28
    image_resample: str = "lanczos"

    # Paths
    output_dir: str = "./outputs"
    cache_dir: str = "./cache"

    # Nested configs
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self):
        """Validate configuration."""
        valid_methods = ["grpo", "nsr", "w_reinforce"]
        if self.policy_method not in valid_methods:
            raise ValueError(f"policy_method must be one of {valid_methods}")

        if self.lambda_psr < 0 or self.lambda_psr > 1:
            raise ValueError("lambda_psr must be in [0, 1]")

        # Ensure output dir exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "TrainingConfig":
        """Create config from dictionary."""
        # Handle nested configs
        if "checkpoint" in config_dict and isinstance(config_dict["checkpoint"], dict):
            config_dict["checkpoint"] = CheckpointConfig(**config_dict["checkpoint"])
        if "rewards" in config_dict and isinstance(config_dict["rewards"], dict):
            config_dict["rewards"] = RewardConfig(**config_dict["rewards"])
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        from dataclasses import asdict
        return asdict(self)

    def get_run_name(self) -> str:
        """Generate descriptive run name."""
        parts = [self.experiment_name, self.policy_method]
        if self.rewards.use_hcpc:
            parts.append("hcpc")
        return "_".join(parts)
