"""Logging utilities for training and evaluation."""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# Try to import wandb, but don't fail if not available
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    experiment_name: Optional[str] = None,
) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file
        experiment_name: Optional experiment name for logger

    Returns:
        Configured logger
    """
    logger_name = experiment_name or "hcpc_rlvr"
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "hcpc_rlvr") -> logging.Logger:
    """Get logger by name."""
    return logging.getLogger(name)


class WandBLogger:
    """Wrapper for Weights & Biases logging."""

    def __init__(
        self,
        project: str = "hcpc-rlvr",
        experiment_name: Optional[str] = None,
        config: Optional[Dict] = None,
        enabled: bool = True,
    ):
        """
        Initialize WandB logger.

        Args:
            project: WandB project name
            experiment_name: Run name
            config: Config dict to log
            enabled: Whether to enable WandB
        """
        self.enabled = enabled and WANDB_AVAILABLE
        self.run = None

        if self.enabled:
            self.run = wandb.init(
                project=project,
                name=experiment_name,
                config=config,
                reinit=True,
            )

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to WandB."""
        if self.enabled and self.run:
            wandb.log(metrics, step=step)

    def log_table(self, name: str, data: list, columns: list):
        """Log a table to WandB."""
        if self.enabled and self.run:
            table = wandb.Table(columns=columns, data=data)
            wandb.log({name: table})

    def finish(self):
        """Finish the WandB run."""
        if self.enabled and self.run:
            wandb.finish()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()


class MetricsLogger:
    """
    Simple metrics logger that writes to JSONL file.

    Useful for offline analysis and when WandB is not available.
    """

    def __init__(self, log_path: str):
        """
        Initialize metrics logger.

        Args:
            log_path: Path to JSONL file
        """
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to file."""
        import json

        entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            **metrics
        }

        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def read_all(self) -> list:
        """Read all logged metrics."""
        import json

        if not self.log_path.exists():
            return []

        entries = []
        with open(self.log_path) as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
        return entries


class ProgressLogger:
    """Simple progress logger for training."""

    def __init__(
        self,
        total_steps: int,
        log_every: int = 10,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize progress logger.

        Args:
            total_steps: Total number of training steps
            log_every: Log every N steps
            logger: Logger to use
        """
        self.total_steps = total_steps
        self.log_every = log_every
        self.logger = logger or get_logger()
        self.start_time = datetime.now()

    def log_step(
        self,
        step: int,
        metrics: Dict[str, float],
        prefix: str = "Train",
    ):
        """
        Log training step.

        Args:
            step: Current step
            metrics: Metrics to log
            prefix: Log message prefix
        """
        if step % self.log_every != 0:
            return

        # Calculate ETA
        elapsed = (datetime.now() - self.start_time).total_seconds()
        steps_per_sec = step / elapsed if elapsed > 0 else 0
        remaining_steps = self.total_steps - step
        eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0

        eta_str = self._format_time(eta_seconds)
        progress = step / self.total_steps * 100

        # Format metrics
        metrics_str = " | ".join(
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in metrics.items()
        )

        self.logger.info(
            f"{prefix} | Step {step}/{self.total_steps} ({progress:.1f}%) | "
            f"{metrics_str} | ETA: {eta_str}"
        )

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds as human readable time."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"


def log_config(config, logger: Optional[logging.Logger] = None):
    """Log configuration parameters."""
    logger = logger or get_logger()
    logger.info("=" * 50)
    logger.info("Configuration:")
    logger.info("=" * 50)

    if hasattr(config, "to_dict"):
        config_dict = config.to_dict()
    elif hasattr(config, "__dict__"):
        config_dict = config.__dict__
    else:
        config_dict = dict(config)

    for key, value in config_dict.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for k, v in value.items():
                logger.info(f"    {k}: {v}")
        else:
            logger.info(f"  {key}: {value}")

    logger.info("=" * 50)
