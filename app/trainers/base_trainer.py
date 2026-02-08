"""Base trainer class for HCPC-RLVR."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path

import torch
from trl import GRPOConfig, GRPOTrainer as TRLGRPOTrainer
import inspect

from configs import TrainingConfig
from rewards import RewardAggregator
from utils.checkpointing import CheckpointManager
from utils.logging_utils import get_logger, WandBLogger, MetricsLogger
import logging
from collections import deque


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
        self._completion_log_step = 0
        self._reward_log_step = 0
        self._training_log_path = None
        self._metrics_window = deque(maxlen=self.config.log_metrics_window)
        self._metrics_step = 0

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
        self._training_log_path = self.checkpoint_manager.run_dir / "train.log"
        self._attach_file_logger()

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
        config_kwargs = {
            "output_dir": str(self.checkpoint_manager.run_dir / "trl_output"),
            "per_device_train_batch_size": self.config.batch_size,
            "num_train_epochs": self.config.num_epochs,
            "num_generations": self.config.num_generations,
            "max_prompt_length": self.config.max_prompt_length,
            "max_completion_length": self.config.max_completion_length,
            "learning_rate": self.config.learning_rate,
            "warmup_ratio": self.config.warmup_ratio,
            "weight_decay": self.config.weight_decay,
            "max_grad_norm": self.config.max_grad_norm,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "logging_steps": self.config.logging_steps,
            "save_steps": self.config.checkpoint.save_every_n_steps,
            "bf16": self.config.bf16,
            "gradient_checkpointing": self.config.gradient_checkpointing,
            # Generation settings
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            # GRPO specific (optional depending on TRL version)
            "kl_coef": self.config.kl_coef,
        }

        # Filter kwargs to those supported by the installed TRL version
        allowed = set(inspect.signature(GRPOConfig).parameters.keys())
        filtered_kwargs = {k: v for k, v in config_kwargs.items() if k in allowed}

        grpo_config = GRPOConfig(**filtered_kwargs)

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
        def _normalize_completion(completion):
            if completion is None:
                return ""
            if isinstance(completion, str):
                return completion
            if isinstance(completion, dict):
                # Attempt to pull a text field
                for key in ("text", "output", "completion"):
                    if key in completion and isinstance(completion[key], str):
                        return completion[key]
                content = completion.get("content")
                if isinstance(content, list):
                    parts = []
                    for item in content:
                        if isinstance(item, str):
                            parts.append(item)
                        elif isinstance(item, dict):
                            text = item.get("text")
                            if isinstance(text, str):
                                parts.append(text)
                    if parts:
                        return "\n".join(parts)
                if isinstance(content, str):
                    return content
                return str(completion)
            if isinstance(completion, list):
                parts = []
                for item in completion:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict):
                        text = item.get("text")
                        if isinstance(text, str):
                            parts.append(text)
                        else:
                            content = item.get("content")
                            if isinstance(content, str):
                                parts.append(content)
                            elif isinstance(content, list):
                                for sub in content:
                                    if isinstance(sub, dict) and isinstance(sub.get("text"), str):
                                        parts.append(sub.get("text"))
                return "\n".join(parts)
            return str(completion)

        def _truncate(text: str) -> str:
            if self.config.log_completions_max_chars <= 0:
                return text
            if len(text) <= self.config.log_completions_max_chars:
                return text
            return text[: self.config.log_completions_max_chars] + "...[truncated]"

        def _log_completions(completions, kwargs, rewards_breakdown=None):
            if not self.config.log_completions:
                return
            if self.config.log_completions_every <= 0:
                return
            if (self._completion_log_step % self.config.log_completions_every) != 0:
                return

            prompts = kwargs.get("prompts", None)
            labels = kwargs.get("labels", None)
            tables = kwargs.get("tables", None)
            chart_types = kwargs.get("chart_types", None)
            reasonings = kwargs.get("reasonings", None)
            if self.config.log_completions_max is None or self.config.log_completions_max < 0:
                n = len(completions)
            else:
                n = min(len(completions), self.config.log_completions_max)

            for i in range(n):
                raw = completions[i]
                norm = _normalize_completion(raw)
                prompt_preview = ""
                if isinstance(prompts, list) and i < len(prompts):
                    prompt_preview = str(prompts[i])
                label_preview = ""
                if isinstance(labels, list) and i < len(labels):
                    label_preview = str(labels[i])

                raw_preview = _truncate(repr(raw))
                reward_info = ""
                if rewards_breakdown and i < len(rewards_breakdown):
                    reward_info = f" | rewards={rewards_breakdown[i]}"

                gt_label = ""
                if isinstance(labels, list) and i < len(labels):
                    gt_label = str(_first_scalar(labels[i], ""))
                gt_table = ""
                if isinstance(tables, list) and i < len(tables):
                    gt_table = str(_first_scalar(tables[i], {}))
                gt_chart_type = ""
                if isinstance(chart_types, list) and i < len(chart_types):
                    gt_chart_type = str(_first_scalar(chart_types[i], ""))
                gt_reasoning = ""
                if isinstance(reasonings, list) and i < len(reasonings):
                    gt_reasoning = str(_first_scalar(reasonings[i], ""))

                msg = (
                    f"[completion_log step={self._completion_log_step} idx={i}] "
                    f"raw_type={type(raw).__name__} raw={raw_preview} | "
                    f"prompt={_truncate(prompt_preview)} | "
                    f"gt_label={_truncate(gt_label)} | "
                    f"gt_chart_type={_truncate(gt_chart_type)} | "
                    f"gt_table={_truncate(gt_table)} | "
                    f"gt_reasoning={_truncate(gt_reasoning)} | "
                    f"completion={_truncate(norm)}"
                    f"{reward_info}"
                )
                self.logger.info(msg)

        def _log_rewards(rewards_breakdown):
            if not self.config.log_rewards:
                return
            if self.config.log_rewards_every <= 0:
                return
            if (self._reward_log_step % self.config.log_rewards_every) != 0:
                return
            for i, breakdown in enumerate(rewards_breakdown):
                msg = (
                    f"[reward_log step={self._reward_log_step} idx={i}] "
                    f"{breakdown}"
                )
                self.logger.info(msg)

        def _update_and_log_metrics(rewards_breakdown):
            if self.config.log_metrics_every <= 0:
                return
            # Aggregate per-step averages
            if not rewards_breakdown:
                return
            keys = rewards_breakdown[0].keys()
            avg = {}
            for k in keys:
                vals = [rb.get(k, 0.0) for rb in rewards_breakdown]
                avg[k] = sum(vals) / max(len(vals), 1)

            self._metrics_window.append(avg)
            if (self._metrics_step % self.config.log_metrics_every) != 0:
                self._metrics_step += 1
                return

            # Rolling average
            roll = {}
            for k in avg.keys():
                vals = [m.get(k, 0.0) for m in self._metrics_window]
                roll[k] = sum(vals) / max(len(vals), 1)

            summary = {
                "step": self._metrics_step,
                "window": len(self._metrics_window),
                "avg_total": roll.get("base_total", 0.0) + roll.get("hcpc", 0.0) + roll.get("clc", 0.0),
                "avg_base_total": roll.get("base_total", 0.0),
                "avg_base_accuracy": roll.get("base_accuracy", 0.0),
                "avg_base_format": roll.get("base_format", 0.0),
                "avg_base_table": roll.get("base_table", 0.0),
                "avg_base_type": roll.get("base_chart_type", 0.0),
                "avg_hcpc": roll.get("hcpc", 0.0),
                "avg_clc": roll.get("clc", 0.0),
            }

            self.logger.info(f"[metrics_summary] {summary}")
            if self.metrics_logger is not None:
                self.metrics_logger.log(summary, step=self._metrics_step)
            self._metrics_step += 1

        def _get_list(kwargs, keys):
            for k in keys:
                if k in kwargs:
                    v = kwargs.get(k)
                    if isinstance(v, list):
                        return v
                    return [v]
            return []

        def _first_scalar(value, default):
            if value is None:
                return default
            if isinstance(value, list):
                if not value:
                    return default
                return _first_scalar(value[0], default)
            return value

        def reward_fn(completions, **kwargs):
            # Get ground truth from kwargs
            labels = _get_list(kwargs, ["labels", "label", "answers", "answer"])
            tables = _get_list(kwargs, ["tables", "table"])
            chart_types = _get_list(kwargs, ["chart_types", "chart_type", "type"])
            reasonings = _get_list(kwargs, ["reasonings", "reasoning", "rationale"])

            # Build ground truth
            ground_truth = {
                "label": _first_scalar(labels, ""),
                "table": _first_scalar(tables, {}),
                "chart_type": _first_scalar(chart_types, ""),
                "reasoning": _first_scalar(reasonings, ""),
            }

            # Normalize completions to strings
            normalized = [_normalize_completion(c) for c in completions]

            # Compute rewards with full breakdown
            results = self.reward_aggregator.compute(normalized, ground_truth)
            rewards = [r.total for r in results]
            rewards_breakdown = [r.breakdown for r in results]

            # Log completions and rewards
            _log_completions(completions, kwargs, rewards_breakdown=rewards_breakdown)
            _log_rewards(rewards_breakdown)
            _update_and_log_metrics(rewards_breakdown)
            self._completion_log_step += 1
            self._reward_log_step += 1

            # Apply policy-specific transformation
            advantages = self.compute_advantages(rewards)

            return advantages

        return reward_fn

    def _attach_file_logger(self):
        """Attach a single file handler to the experiment logger."""
        if not self._training_log_path:
            return
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                try:
                    if Path(handler.baseFilename) == Path(self._training_log_path):
                        return
                except Exception:
                    pass
        file_handler = logging.FileHandler(self._training_log_path)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)

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
