#!/usr/bin/env python3
"""
Training script — GRPO + GT-CLC (no HCPC).

Active reward components:
  base_accuracy    0-1.0   correct answer in <answer> tag
  base_format      0-1.4   structured <think>/<table>/<answer> output
  base_length      0-1.5   reasoning step count and length
  base_table       0-1.75  table extraction quality vs GT table
  gt_clc           0-1.0   GT answer value cited in reasoning steps

Disabled (always 0 in sanchit97/chart-rvr-grpo-train):
  base_chart_type  (GT chart_type field is empty)
  base_token_count (model produces multiple <think> tags)

GRPO advantage: reward_i - mean(rewards) / std(rewards)
GT-CLC mode:    answer_only (sanchit97 has no GT reasoning field)

Usage:
    # Full training run
    python scripts/train_grpo_gt_clc.py

    # Quick sanity check (100 samples, no wandb)
    python scripts/train_grpo_gt_clc.py --subset-size 100 --no-wandb

    # Resume from latest checkpoint
    python scripts/train_grpo_gt_clc.py --resume

    # Resume from a specific checkpoint
    python scripts/train_grpo_gt_clc.py --resume-from outputs/grpo_gt_clc/run_xxx/checkpoints/step_100

    # Override training length
    python scripts/train_grpo_gt_clc.py --num-epochs 3 --num-generations 8
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import random

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs import get_experiment_config
from data import load_training_dataset
from models import load_model_for_training
from trainers import get_trainer
from utils.logging_utils import setup_logging, log_config

EXPERIMENT = "grpo_gt_clc"


def parse_args():
    parser = argparse.ArgumentParser(
        description="GRPO + GT-CLC Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Checkpoint
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from latest checkpoint of the latest run",
    )
    parser.add_argument(
        "--resume-from", type=str, default=None,
        help="Resume from a specific checkpoint path",
    )
    parser.add_argument(
        "--from-scratch", action="store_true",
        help="Start a new run without deleting old checkpoints",
    )

    # Dataset
    parser.add_argument(
        "--subset-size", type=int, default=None,
        help="Limit training set size (useful for quick iteration)",
    )

    # Training overrides
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument(
        "--num-generations", type=int, default=None,
        help="Number of rollouts per sample (default from config: 4)",
    )

    # Logging
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable WandB logging",
    )

    # Paths
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--cache-dir", type=str, default="./cache")

    return parser.parse_args()


def main():
    args = parse_args()

    # Silence wandb before any imports that might trigger it
    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["WANDB_SILENT"] = "true"

    # Build config overrides from CLI args
    overrides = {
        "output_dir": args.output_dir,
        "cache_dir": args.cache_dir,
        "checkpoint": {
            "resume": args.resume or (args.resume_from is not None),
            "resume_from": args.resume_from,
            "from_scratch": args.from_scratch,
        },
    }
    if args.subset_size is not None:
        overrides["subset_size"] = args.subset_size
    if args.num_epochs is not None:
        overrides["num_epochs"] = args.num_epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        overrides["learning_rate"] = args.learning_rate
    if args.num_generations is not None:
        overrides["num_generations"] = args.num_generations
    if args.no_wandb:
        overrides["use_wandb"] = False

    config = get_experiment_config(EXPERIMENT, **overrides)

    # Point HuggingFace cache to the configured cache dir
    os.environ["HF_HUB_CACHE"] = config.cache_dir
    os.environ["TRANSFORMERS_CACHE"] = config.cache_dir
    os.environ["HF_HOME"] = config.cache_dir
    os.environ["FLASH_ATTENTION_2_ENABLED"] = "1"

    # Reproducibility
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    os.environ["PYTHONHASHSEED"] = str(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # bf16 requires CUDA
    if not torch.cuda.is_available():
        config.bf16 = False

    # Logging
    logger = setup_logging(log_level="INFO", experiment_name=config.experiment_name)
    log_config(config, logger)

    # Show exactly which reward signals are active
    r = config.rewards
    logger.info("=" * 50)
    logger.info("REWARD COMPONENTS")
    logger.info("=" * 50)
    logger.info(f"  base_accuracy    enabled")
    logger.info(f"  base_format      enabled  (max 1.4 — full 2.0 requires exact pattern)")
    logger.info(f"  base_length      enabled  (step count + char length)")
    logger.info(f"  base_table       enabled  (JSON parse + column/row match)")
    logger.info(f"  base_chart_type  {'enabled' if r.use_chart_type_reward else 'DISABLED — GT field empty in sanchit97'}")
    logger.info(f"  base_token_count {'enabled' if r.use_token_count_reward else 'DISABLED — multi-think pattern in Qwen'}")
    logger.info(f"  gt_clc           {'enabled' if r.use_gt_clc else 'disabled'}")
    if r.use_gt_clc:
        logger.info(f"    w_gt_clc={r.w_gt_clc}  answer_weight={r.answer_weight}  value_weight={r.value_weight}")
        logger.info(f"    mode: answer_only (sanchit97 has no GT reasoning)")
        logger.info(f"    signal: 0 or 1.0 per rollout for numeric answers, 0.5 neutral for text answers")
    logger.info(f"  hcpc             {'enabled' if r.use_hcpc else 'disabled'}")
    logger.info("=" * 50)

    # Load model
    logger.info("Loading model...")
    model, processor = load_model_for_training(config)

    # Load dataset
    # Fields passed to reward_fn via TRL kwargs:
    #   labels/label         → ground_truth["label"]    → GT-CLC answer target
    #   tables/table         → ground_truth["table"]    → GT-CLC table values (unused in answer_only)
    #   reasonings/reasoning → ground_truth["reasoning"] → GT-CLC key values (empty in sanchit97)
    logger.info("Loading dataset...")
    train_dataset = load_training_dataset(config, processor)
    if config.use_python_list_dataset and not isinstance(train_dataset, list):
        train_dataset = [train_dataset[i] for i in range(len(train_dataset))]
    logger.info(f"Loaded {len(train_dataset)} training samples")

    # GRPOTrainer: advantage = (reward - mean) / std
    # use_hcpc=False → routes to standard GRPOTrainer (not NSRHCPCTrainer)
    trainer_cls = get_trainer(config.policy_method, use_hcpc=config.rewards.use_hcpc)
    logger.info(f"Trainer:   {trainer_cls.__name__}")
    logger.info(f"Advantage: GRPO  (reward_i - mean(rewards)) / std(rewards)")

    trainer = trainer_cls(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        config=config,
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
