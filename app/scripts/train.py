#!/usr/bin/env python3
"""
Main training script for HCPC-RLVR.

Usage:
    # Train with a predefined experiment
    python scripts/train.py --experiment grpo_baseline

    # Train with HCPC
    python scripts/train.py --experiment nsr_hcpc

    # Resume from checkpoint
    python scripts/train.py --experiment grpo_baseline --resume

    # Resume from specific checkpoint
    python scripts/train.py --experiment grpo_baseline --resume-from outputs/grpo_baseline/run_xxx/checkpoints/step_100

    # Start fresh (new run, keeps old checkpoints)
    python scripts/train.py --experiment grpo_baseline --from-scratch

    # Quick test with subset
    python scripts/train.py --experiment grpo_baseline --subset-size 100

    # List available experiments
    python scripts/train.py --list-experiments

    # List runs for an experiment
    python scripts/train.py --experiment grpo_baseline --list-runs
"""

import argparse
import sys
import os
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs import get_experiment_config, list_experiments, EXPERIMENTS
from data import load_training_dataset
from models import load_model_for_training
from trainers import get_trainer
from utils.logging_utils import setup_logging, log_config
from utils.checkpointing import CheckpointManager
import random
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="HCPC-RLVR Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Experiment selection
    parser.add_argument(
        "--experiment", "-e",
        type=str,
        choices=list(EXPERIMENTS.keys()),
        help="Experiment configuration to use",
    )

    # Checkpoint management
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint of latest run",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume from specific checkpoint path",
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Start new run (don't delete old checkpoints)",
    )

    # Dataset options
    parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="Use subset of data for quick iteration",
    )

    # Training overrides
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=None,
        help="Override number of rollouts per sample",
    )

    # Reward options
    parser.add_argument(
        "--no-hcpc",
        action="store_true",
        help="Disable HCPC reward",
    )

    # Logging
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB logging",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Force bf16 on",
    )
    parser.add_argument(
        "--no-bf16",
        action="store_true",
        help="Force bf16 off",
    )
    parser.add_argument(
        "--torch-dtype-auto",
        action="store_true",
        help="Force torch_dtype=auto",
    )
    parser.add_argument(
        "--no-torch-dtype-auto",
        action="store_true",
        help="Force torch_dtype auto off (use bf16/float32 setting)",
    )

    # Information commands
    parser.add_argument(
        "--list-experiments",
        action="store_true",
        help="List available experiments and exit",
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="List runs for the specified experiment",
    )

    # Paths
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./cache",
        help="Cache directory for datasets",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Disable wandb completely when requested to avoid interactive prompts
    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["WANDB_SILENT"] = "true"

    # Handle information commands
    if args.list_experiments:
        list_experiments()
        return

    if args.list_runs:
        if not args.experiment:
            print("Error: --experiment required for --list-runs")
            return
        manager = CheckpointManager(args.experiment, args.output_dir)
        runs = manager.list_runs()
        if not runs:
            print(f"No runs found for experiment: {args.experiment}")
        else:
            print(f"Runs for {args.experiment}:")
            for run in runs:
                print(f"  {run['run_id']} - step {run.get('latest_step', 0)} - {run.get('status', 'unknown')}")
        return

    # Require experiment for training
    if not args.experiment:
        print("Error: --experiment required for training")
        print("Use --list-experiments to see available options")
        return

    # Build config with overrides
    overrides = {
        "output_dir": args.output_dir,
        "cache_dir": args.cache_dir,
    }

    if args.subset_size:
        overrides["subset_size"] = args.subset_size
    if args.num_epochs:
        overrides["num_epochs"] = args.num_epochs
    if args.batch_size:
        overrides["batch_size"] = args.batch_size
    if args.learning_rate:
        overrides["learning_rate"] = args.learning_rate
    if args.num_generations:
        overrides["num_generations"] = args.num_generations
    if args.no_wandb:
        overrides["use_wandb"] = False
    if args.bf16 and args.no_bf16:
        print("Error: use only one of --bf16 or --no-bf16")
        return
    if args.bf16:
        overrides["bf16"] = True
    if args.no_bf16:
        overrides["bf16"] = False
    if args.torch_dtype_auto and args.no_torch_dtype_auto:
        print("Error: use only one of --torch-dtype-auto or --no-torch-dtype-auto")
        return
    if args.torch_dtype_auto:
        overrides["torch_dtype_auto"] = True
    if args.no_torch_dtype_auto:
        overrides["torch_dtype_auto"] = False

    # Checkpoint settings
    # --resume-from implies --resume
    overrides["checkpoint"] = {
        "resume": args.resume or (args.resume_from is not None),
        "resume_from": args.resume_from,
        "from_scratch": args.from_scratch,
    }

    # Reward settings
    if args.no_hcpc:
        overrides["rewards"] = {"use_hcpc": False}

    # Get config
    config = get_experiment_config(args.experiment, **overrides)

    # Align HF cache env with config (legacy behavior)
    os.environ["HF_HUB_CACHE"] = config.cache_dir
    os.environ["TRANSFORMERS_CACHE"] = config.cache_dir
    os.environ["HF_HOME"] = config.cache_dir
    os.environ["FLASH_ATTENTION_2_ENABLED"] = "1"

    # Set seeds for reproducibility (match legacy training)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    os.environ["PYTHONHASHSEED"] = str(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Ensure bf16 only when CUDA is available
    if not torch.cuda.is_available():
        config.bf16 = False

    # Setup logging
    logger = setup_logging(
        log_level="INFO",
        experiment_name=config.experiment_name,
    )
    log_config(config, logger)

    # Load model
    logger.info("Loading model...")
    model, processor = load_model_for_training(config)

    # Load dataset
    logger.info("Loading dataset...")
    train_dataset = load_training_dataset(config, processor)
    if config.use_python_list_dataset and not isinstance(train_dataset, list):
        train_dataset = [train_dataset[i] for i in range(len(train_dataset))]
    logger.info(f"Loaded {len(train_dataset)} training samples")

    # Get trainer class — pass use_hcpc so nsr+hcpc routes to NSRHCPCTrainer
    trainer_cls = get_trainer(config.policy_method, use_hcpc=config.rewards.use_hcpc)
    logger.info(f"Using trainer: {trainer_cls.__name__}")

    # Create trainer
    trainer = trainer_cls(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        config=config,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
