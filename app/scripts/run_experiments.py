#!/usr/bin/env python3
"""
Run all 6 experiments from the HCPC-RLVR methodology.

Experiments:
1. grpo_baseline:        GRPO (no HCPC) - Chart-RVR reproduction
2. grpo_hcpc:            GRPO + HCPC
3. nsr_baseline:         NSR (no HCPC)
4. nsr_hcpc:             NSR + HCPC
5. w_reinforce_baseline: W-REINFORCE (no HCPC)
6. w_reinforce_hcpc:     W-REINFORCE + HCPC (Full HCPC-RLVR)

Usage:
    # Run all experiments
    python scripts/run_experiments.py

    # Run specific experiments
    python scripts/run_experiments.py --experiments grpo_baseline nsr_hcpc

    # Run with subset for quick testing
    python scripts/run_experiments.py --subset-size 100

    # Skip training, only evaluate
    python scripts/run_experiments.py --eval-only

    # Dry run (show what would be done)
    python scripts/run_experiments.py --dry-run
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs import EXPERIMENTS, list_experiments


# Experiment order (for running sequentially)
EXPERIMENT_ORDER = [
    "grpo_baseline",
    "grpo_hcpc",
    "nsr_baseline",
    "nsr_hcpc",
    "w_reinforce_baseline",
    "w_reinforce_hcpc",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run HCPC-RLVR experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=None,
        help="Specific experiments to run (default: all)",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="Use subset of data for quick testing",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training, only evaluate existing checkpoints",
    )
    parser.add_argument(
        "--id-dataset",
        type=str,
        default="chartqa",
        help="In-distribution dataset for evaluation",
    )
    parser.add_argument(
        "--ood-dataset",
        type=str,
        default="evochart",
        help="Out-of-distribution dataset for evaluation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Output directory",
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default="experiment_results.json",
        help="File to save aggregated results",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without running",
    )

    return parser.parse_args()


def run_training(experiment: str, args) -> bool:
    """Run training for an experiment."""
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "train.py"),
        "--experiment", experiment,
        "--output-dir", args.output_dir,
    ]

    if args.subset_size:
        cmd.extend(["--subset-size", str(args.subset_size)])
    if args.num_epochs:
        cmd.extend(["--num-epochs", str(args.num_epochs)])

    print(f"\n{'='*60}")
    print(f"TRAINING: {experiment}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    if args.dry_run:
        return True

    result = subprocess.run(cmd)
    return result.returncode == 0


def run_evaluation(experiment: str, args) -> dict:
    """Run evaluation for an experiment."""
    checkpoint_path = Path(args.output_dir) / experiment / "best"

    if not checkpoint_path.exists():
        # Try latest run
        exp_dir = Path(args.output_dir) / experiment
        if exp_dir.exists():
            runs = sorted(exp_dir.glob("run_*"))
            if runs:
                checkpoint_path = runs[-1] / "checkpoints" / "latest"

    if not checkpoint_path.exists():
        print(f"No checkpoint found for {experiment}")
        return None

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "evaluate.py"),
        "--checkpoint", str(checkpoint_path),
        "--id-dataset", args.id_dataset,
        "--ood-dataset", args.ood_dataset,
        "--output", str(Path(args.output_dir) / experiment / "eval_results.json"),
    ]

    if args.subset_size:
        cmd.extend(["--subset", str(min(args.subset_size, 200))])

    print(f"\n{'='*60}")
    print(f"EVALUATING: {experiment}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    if args.dry_run:
        return {"status": "dry_run"}

    result = subprocess.run(cmd)

    # Load results
    results_path = Path(args.output_dir) / experiment / "eval_results.json"
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)

    return None


def print_summary(all_results: dict):
    """Print summary table of all results."""
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)

    header = f"{'Experiment':<25} {'ID Acc':>8} {'OOD Acc':>8} {'Gap':>8} {'C_table':>8} {'D_reason':>8}"
    print(header)
    print("-" * 80)

    for exp, results in all_results.items():
        if results is None:
            print(f"{exp:<25} {'N/A':>8}")
            continue

        id_acc = results.get("id", {}).get("accuracy", 0) * 100
        ood_acc = results.get("ood", {}).get("accuracy", 0) * 100
        gap = results.get("ood_gap", 0)
        c_table = results.get("ood", {}).get("c_table", 0)
        d_reason = results.get("ood", {}).get("d_reason", 0)

        print(f"{exp:<25} {id_acc:>7.1f}% {ood_acc:>7.1f}% {gap:>7.1f}% {c_table:>8.3f} {d_reason:>8.3f}")

    print("=" * 80)


def main():
    args = parse_args()

    # Determine which experiments to run
    experiments = args.experiments or EXPERIMENT_ORDER

    # Validate experiments
    for exp in experiments:
        if exp not in EXPERIMENTS:
            print(f"Unknown experiment: {exp}")
            list_experiments()
            return

    print(f"Experiments to run: {experiments}")

    all_results = {}

    for experiment in experiments:
        # Training
        if not args.eval_only:
            success = run_training(experiment, args)
            if not success:
                print(f"Training failed for {experiment}")
                continue

        # Evaluation
        results = run_evaluation(experiment, args)
        all_results[experiment] = results

    # Save aggregated results
    results_path = Path(args.output_dir) / args.results_file
    with open(results_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "experiments": all_results,
        }, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    # Print summary
    print_summary(all_results)


if __name__ == "__main__":
    main()
