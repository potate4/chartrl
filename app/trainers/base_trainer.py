"""Base trainer class for HCPC-RLVR."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path

import torch
from trl import GRPOConfig, GRPOTrainer as TRLGRPOTrainer

from configs import TrainingConfig
from rewards import RewardAggregator
from utils.checkpointing import CheckpointManager
from utils.logging_utils import get_logger, WandBLogger, MetricsLogger


class BaseTrainer(ABC):
    """
    Base trainer class that wraps TRL's GRPOTrainer.

    Subclasses implement different advantage computation methods:
    - GRPO: Standard relative advantages
    - NSR: Only penalize wrong samples
    - W-REINFORCE: Weighted combination
    """

    def __init__(
        self,
        model,
        processor,
        train_dataset,
        config: TrainingConfig,
        eval_dataset=None,
    ):
        """
        Initialize trainer.

        Args:
            model: PEFT model
            processor: Model processor
            train_dataset: Training dataset
            config: Training configuration
            eval_dataset: Optional evaluation dataset
        """
        self.model = model
        self.processor = processor
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config

        self.logger = get_logger(config.experiment_name)

        # Initialize reward aggregator
        self.reward_aggregator = RewardAggregator.from_config(config)

        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            experiment_name=config.experiment_name,
            output_base=config.output_dir,
            save_every_n_steps=config.checkpoint.save_every_n_steps,
            keep_last_n=config.checkpoint.keep_last_n,
            keep_best=config.checkpoint.keep_best,
        )

        # Initialize metrics logger
        self.metrics_logger = None
        self.wandb_logger = None

        # Internal state
        self._trl_trainer = None
        self._current_step = 0
        self._resume_step = 0

    def setup(self):
        """Set up training components."""
        # Create run directory
        if self.config.checkpoint.resume and not self.config.checkpoint.from_scratch:
            # Try to resume from latest run
            latest_run = self.checkpoint_manager.get_latest_run()
            if latest_run:
                self.checkpoint_manager.set_run_dir(latest_run)
                self.logger.info(f"Resuming from run: {latest_run}")
            else:
                self._create_new_run()
        else:
            self._create_new_run()

        # Initialize loggers
        self.metrics_logger = MetricsLogger(
            self.checkpoint_manager.run_dir / "metrics.jsonl"
        )

        if self.config.use_wandb:
            self.wandb_logger = WandBLogger(
                project=self.config.wandb_project,
                experiment_name=self.config.get_run_name(),
                config=self.config.to_dict(),
            )

        # Create TRL trainer
        self._create_trl_trainer()

    def _create_new_run(self):
        """Create a new training run."""
        run_dir = self.checkpoint_manager.create_new_run(
            config=self.config.to_dict()
        )
        self.logger.info(f"Created new run: {run_dir}")

    def _create_trl_trainer(self):
        """Create the underlying TRL GRPOTrainer."""
        grpo_config = GRPOConfig(
            output_dir=str(self.checkpoint_manager.run_dir / "trl_output"),
            per_device_train_batch_size=self.config.batch_size,
            num_train_epochs=self.config.num_epochs,
            num_generations=self.config.num_generations,
            max_prompt_length=self.config.max_prompt_length,
            max_completion_length=self.config.max_completion_length,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.checkpoint.save_every_n_steps,
            bf16=self.config.bf16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            # Generation settings
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            # GRPO specific
            kl_coef=self.config.kl_coef,
        )

        # Create reward function
        reward_fn = self._create_reward_function()

        self._trl_trainer = TRLGRPOTrainer(
            model=self.model,
            processing_class=self.processor,
            args=grpo_config,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            reward_funcs=reward_fn,
        )

    def _create_reward_function(self):
        """
        Create reward function for TRL trainer.

        The reward function computes base rewards and then applies
        the subclass-specific advantage transformation.
        """
        def reward_fn(completions, **kwargs):
            # Get ground truth from kwargs
            labels = kwargs.get("labels", [])
            tables = kwargs.get("tables", [])
            chart_types = kwargs.get("chart_types", [])
            reasonings = kwargs.get("reasonings", [])

            # Build ground truth
            ground_truth = {
                "label": labels[0] if labels else "",
                "table": tables[0] if tables else {},
                "chart_type": chart_types[0] if chart_types else "",
                "reasoning": reasonings[0] if reasonings else "",
            }

            # Compute raw rewards
            rewards = self.reward_aggregator.compute_rewards_tensor(
                completions, ground_truth
            )

            # Apply policy-specific transformation
            advantages = self.compute_advantages(rewards)

            return advantages

        return reward_fn

    @abstractmethod
    def compute_advantages(self, rewards: List[float]) -> List[float]:
        """
        Compute advantages from rewards.

        This is the key method that differs between GRPO, NSR, and W-REINFORCE.

        Args:
            rewards: Raw reward values for each rollout

        Returns:
            Advantage values for policy gradient
        """
        pass

    def train(self):
        """Run training."""
        self.setup()

        # Resume from checkpoint if needed
        if self.config.checkpoint.resume:
            self._resume_from_checkpoint()

        self.logger.info("Starting training...")
        self.logger.info(f"Policy method: {self.config.policy_method}")
        self.logger.info(f"HCPC enabled: {self.config.rewards.use_hcpc}")
        self.logger.info(f"CLC enabled: {self.config.rewards.use_clc}")

        # Train
        try:
            self._trl_trainer.train(resume_from_checkpoint=self._get_resume_path())
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        finally:
            self._cleanup()

        self.logger.info("Training complete")

    def _resume_from_checkpoint(self):
        """Resume from a checkpoint."""
        checkpoint_path = None

        if self.config.checkpoint.resume_from:
            checkpoint_path = Path(self.config.checkpoint.resume_from)
        else:
            checkpoint_path = self.checkpoint_manager.get_latest_checkpoint()

        if checkpoint_path and checkpoint_path.exists():
            self.logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            trainer_state = self.checkpoint_manager.load_checkpoint(
                checkpoint_path,
                self.model,
                self._trl_trainer.optimizer if hasattr(self._trl_trainer, 'optimizer') else None,
            )
            self._resume_step = trainer_state.get("global_step", 0)
            self.logger.info(f"Resuming from step {self._resume_step}")
        else:
            self.logger.info("No checkpoint found, starting from scratch")

    def _get_resume_path(self) -> Optional[str]:
        """Get path for TRL's resume_from_checkpoint."""
        if self._resume_step > 0:
            ckpt = self.checkpoint_manager.get_latest_checkpoint()
            if ckpt:
                return str(ckpt)
        return None

    def save_checkpoint(self, step: int, metrics: Dict[str, float] = None):
        """Save a checkpoint."""
        self.checkpoint_manager.save_checkpoint(
            step=step,
            model=self.model,
            optimizer=self._trl_trainer.optimizer if hasattr(self._trl_trainer, 'optimizer') else None,
            scheduler=self._trl_trainer.lr_scheduler if hasattr(self._trl_trainer, 'lr_scheduler') else None,
            metrics=metrics,
        )
        self.logger.info(f"Saved checkpoint at step {step}")

    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to all loggers."""
        self.metrics_logger.log(metrics, step=step)

        if self.wandb_logger:
            self.wandb_logger.log(metrics, step=step)

    def _cleanup(self):
        """Cleanup after training."""
        self.checkpoint_manager.mark_run_complete()

        if self.wandb_logger:
            self.wandb_logger.finish()


class PolicyMethodMixin:
    """
    Mixin that provides utility methods for policy computations.
    """

    def normalize_rewards(self, rewards: List[float]) -> List[float]:
        """Normalize rewards to zero mean, unit variance."""
        if not rewards:
            return rewards

        mean_r = sum(rewards) / len(rewards)
        var_r = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
        std_r = max(var_r ** 0.5, 1e-8)

        return [(r - mean_r) / std_r for r in rewards]

    def get_max_reward(self, config: TrainingConfig) -> float:
        """Estimate maximum possible reward for threshold calculation."""
        # Base rewards max: format(2) + acc(1) + len(2) + token(2) + type(1) + table(2) + process(1) = 11
        max_base = 11.0

        # HCPC max: (w_type + w_table + w_reason) if all correct and diverse
        max_hcpc = 0.0
        if config.rewards.use_hcpc:
            max_hcpc = config.rewards.w_type + config.rewards.w_table + config.rewards.w_reason

        # CLC max: w_clc
        max_clc = config.rewards.w_clc if config.rewards.use_clc else 0.0

        return max_base + max_hcpc + max_clc
