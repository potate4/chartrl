"""
Training Monitoring and Visualization Script

Generates charts to visualize training progress and convergence.

Usage:
    # Monitor from wandb
    python monitor_training.py --wandb-run <run-id>

    # Monitor from checkpoint logs
    python monitor_training.py --checkpoint-dir grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-nsr-2025

    # Monitor multiple runs for comparison
    python monitor_training.py --compare \
        --runs nsr-run-id psr-run-id grpo-run-id w-reinforce-run-id \
        --labels NSR PSR GRPO W-REINFORCE
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd


def load_trainer_state(checkpoint_dir: str) -> Dict:
    """Load trainer_state.json from checkpoint directory"""
    state_file = Path(checkpoint_dir) / "trainer_state.json"
    if not state_file.exists():
        raise FileNotFoundError(f"trainer_state.json not found in {checkpoint_dir}")

    with open(state_file, 'r') as f:
        return json.load(f)


def plot_training_loss(states: Dict[str, Dict], output_path: str = "training_loss.png"):
    """
    Plot training loss curves for multiple runs.

    Args:
        states: Dict mapping run_name -> trainer_state
        output_path: Where to save plot
    """
    plt.figure(figsize=(12, 6))

    for run_name, state in states.items():
        log_history = state.get('log_history', [])
        if not log_history:
            print(f"Warning: No log history for {run_name}")
            continue

        # Extract loss values
        steps = []
        losses = []
        for entry in log_history:
            if 'loss' in entry:
                steps.append(entry['step'])
                losses.append(entry['loss'])

        if losses:
            plt.plot(steps, losses, marker='o', markersize=3, label=run_name, linewidth=2)

    plt.xlabel('Training Step', fontweight='bold')
    plt.ylabel('Loss', fontweight='bold')
    plt.title('Training Loss Over Time', fontweight='bold', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved training loss plot to {output_path}")
    plt.close()


def plot_learning_rate(states: Dict[str, Dict], output_path: str = "learning_rate.png"):
    """Plot learning rate schedule"""
    plt.figure(figsize=(12, 6))

    for run_name, state in states.items():
        log_history = state.get('log_history', [])

        steps = []
        lrs = []
        for entry in log_history:
            if 'learning_rate' in entry:
                steps.append(entry['step'])
                lrs.append(entry['learning_rate'])

        if lrs:
            plt.plot(steps, lrs, marker='o', markersize=3, label=run_name, linewidth=2)

    plt.xlabel('Training Step', fontweight='bold')
    plt.ylabel('Learning Rate', fontweight='bold')
    plt.title('Learning Rate Schedule', fontweight='bold', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved learning rate plot to {output_path}")
    plt.close()


def plot_rewards(states: Dict[str, Dict], output_path: str = "rewards.png"):
    """Plot reward progression (if available in logs)"""
    plt.figure(figsize=(12, 6))

    for run_name, state in states.items():
        log_history = state.get('log_history', [])

        steps = []
        rewards = []
        for entry in log_history:
            # Look for reward metrics (may vary by mode)
            if 'train/reward' in entry:
                steps.append(entry['step'])
                rewards.append(entry['train/reward'])
            elif 'rewards/mean' in entry:
                steps.append(entry['step'])
                rewards.append(entry['rewards/mean'])

        if rewards:
            plt.plot(steps, rewards, marker='o', markersize=3, label=run_name, linewidth=2)

    if not any(states.values()):
        print("No reward data found in logs")
        return

    plt.xlabel('Training Step', fontweight='bold')
    plt.ylabel('Average Reward', fontweight='bold')
    plt.title('Reward Progression Over Time', fontweight='bold', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved rewards plot to {output_path}")
    plt.close()


def plot_gradient_norm(states: Dict[str, Dict], output_path: str = "gradient_norm.png"):
    """Plot gradient norm (convergence indicator)"""
    plt.figure(figsize=(12, 6))

    for run_name, state in states.items():
        log_history = state.get('log_history', [])

        steps = []
        grad_norms = []
        for entry in log_history:
            if 'grad_norm' in entry:
                steps.append(entry['step'])
                grad_norms.append(entry['grad_norm'])

        if grad_norms:
            plt.plot(steps, grad_norms, marker='o', markersize=3, label=run_name, linewidth=2, alpha=0.7)

    if not any(states.values()):
        print("No gradient norm data found in logs")
        return

    plt.xlabel('Training Step', fontweight='bold')
    plt.ylabel('Gradient Norm', fontweight='bold')
    plt.title('Gradient Norm (Convergence Indicator)', fontweight='bold', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved gradient norm plot to {output_path}")
    plt.close()


def plot_nsr_specific_metrics(checkpoint_dir: str, output_path: str = "nsr_metrics.png"):
    """
    Plot NSR-specific metrics (positive/negative sample ratios).

    Reads from wandb logs if available.
    """
    # Try to load wandb logs
    logs_dir = Path(checkpoint_dir) / "logs"
    if not logs_dir.exists():
        print(f"No logs directory found in {checkpoint_dir}")
        return

    # Look for wandb event files
    import glob
    event_files = glob.glob(str(logs_dir / "events.out.tfevents.*"))

    if not event_files:
        print("No tensorboard event files found")
        return

    # Parse tensorboard events (requires tensorflow or tensorboard)
    try:
        from tensorboard.backend.event_processing import event_accumulator
        ea = event_accumulator.EventAccumulator(str(logs_dir))
        ea.Reload()

        # Extract metrics
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # PSR positive ratio
        if 'psr/positive_ratio' in ea.Tags()['scalars']:
            events = ea.Scalars('psr/positive_ratio')
            steps = [e.step for e in events]
            values = [e.value for e in events]
            axes[0, 0].plot(steps, values, marker='o', markersize=3, linewidth=2)
            axes[0, 0].set_title('PSR: Positive Sample Ratio', fontweight='bold')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Ratio')
            axes[0, 0].grid(True, alpha=0.3)

        # NSR negative ratio
        if 'nsr/negative_ratio' in ea.Tags()['scalars']:
            events = ea.Scalars('nsr/negative_ratio')
            steps = [e.step for e in events]
            values = [e.value for e in events]
            axes[0, 1].plot(steps, values, marker='o', markersize=3, linewidth=2, color='red')
            axes[0, 1].set_title('NSR: Negative Sample Ratio', fontweight='bold')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Ratio')
            axes[0, 1].grid(True, alpha=0.3)

        # W-REINFORCE pos/neg ratio
        if 'w_reinforce/pos_neg_ratio' in ea.Tags()['scalars']:
            events = ea.Scalars('w_reinforce/pos_neg_ratio')
            steps = [e.step for e in events]
            values = [e.value for e in events]
            axes[1, 0].plot(steps, values, marker='o', markersize=3, linewidth=2, color='purple')
            axes[1, 0].set_title('W-REINFORCE: Pos/Neg Sample Ratio', fontweight='bold')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Ratio')
            axes[1, 0].grid(True, alpha=0.3)

        # Average rewards comparison
        for metric_name, color in [('psr/avg_reward', 'blue'),
                                   ('nsr/avg_reward', 'red'),
                                   ('w_reinforce/avg_reward', 'purple')]:
            if metric_name in ea.Tags()['scalars']:
                events = ea.Scalars(metric_name)
                steps = [e.step for e in events]
                values = [e.value for e in events]
                mode = metric_name.split('/')[0].upper()
                axes[1, 1].plot(steps, values, marker='o', markersize=2,
                               linewidth=2, color=color, label=mode, alpha=0.8)

        axes[1, 1].set_title('Average Rewards by Mode', fontweight='bold')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Avg Reward')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved NSR-specific metrics to {output_path}")
        plt.close()

    except ImportError:
        print("tensorboard not installed, skipping NSR-specific metrics")


def print_training_summary(states: Dict[str, Dict]):
    """Print summary statistics for each run"""
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)

    for run_name, state in states.items():
        print(f"\n{run_name}:")
        print("-" * 40)

        log_history = state.get('log_history', [])
        if not log_history:
            print("  No log history available")
            continue

        # Extract final metrics
        final_entry = log_history[-1] if log_history else {}

        print(f"  Total steps: {final_entry.get('step', 'N/A')}")
        print(f"  Final loss: {final_entry.get('loss', 'N/A'):.4f}" if 'loss' in final_entry else "  Final loss: N/A")
        print(f"  Final LR: {final_entry.get('learning_rate', 'N/A'):.2e}" if 'learning_rate' in final_entry else "  Final LR: N/A")

        # Compute loss reduction
        losses = [e['loss'] for e in log_history if 'loss' in e]
        if len(losses) > 1:
            initial_loss = losses[0]
            final_loss = losses[-1]
            reduction = (initial_loss - final_loss) / initial_loss * 100
            print(f"  Loss reduction: {reduction:.1f}% ({initial_loss:.4f} → {final_loss:.4f})")

        # Training time
        if 'train_runtime' in state:
            runtime_hours = state['train_runtime'] / 3600
            print(f"  Training time: {runtime_hours:.2f} hours")

        # Convergence check
        if len(losses) > 10:
            recent_losses = losses[-10:]
            loss_std = np.std(recent_losses)
            loss_mean = np.mean(recent_losses)
            if loss_std / loss_mean < 0.01:
                print(f"  ✓ Converged (loss std/mean = {loss_std/loss_mean:.4f})")
            else:
                print(f"  ⚠ Not fully converged (loss std/mean = {loss_std/loss_mean:.4f})")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Monitor GRPO/NSR training progress")

    # Single checkpoint
    parser.add_argument('--checkpoint-dir', type=str,
                       help='Path to checkpoint directory')

    # Multiple checkpoints for comparison
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple runs')
    parser.add_argument('--checkpoints', type=str, nargs='+',
                       help='List of checkpoint directories to compare')
    parser.add_argument('--labels', type=str, nargs='+',
                       help='Labels for each checkpoint (optional)')

    # Output
    parser.add_argument('--output-dir', type=str, default='training_plots',
                       help='Directory to save plots (default: training_plots)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load checkpoint states
    states = {}

    if args.compare:
        if not args.checkpoints:
            print("Error: --checkpoints required when using --compare")
            return

        labels = args.labels if args.labels else [f"Run {i+1}" for i in range(len(args.checkpoints))]

        for checkpoint_dir, label in zip(args.checkpoints, labels):
            try:
                state = load_trainer_state(checkpoint_dir)
                states[label] = state
                print(f"✓ Loaded {label} from {checkpoint_dir}")
            except Exception as e:
                print(f"✗ Failed to load {checkpoint_dir}: {e}")

    elif args.checkpoint_dir:
        try:
            state = load_trainer_state(args.checkpoint_dir)
            run_name = Path(args.checkpoint_dir).name
            states[run_name] = state
            print(f"✓ Loaded {run_name}")
        except Exception as e:
            print(f"✗ Failed to load {args.checkpoint_dir}: {e}")
            return

        # Also plot NSR-specific metrics if available
        if any(mode in run_name.lower() for mode in ['psr', 'nsr', 'w-reinforce']):
            plot_nsr_specific_metrics(args.checkpoint_dir, output_dir / "nsr_metrics.png")

    else:
        print("Error: Either --checkpoint-dir or --compare with --checkpoints required")
        return

    if not states:
        print("No valid checkpoint states loaded")
        return

    # Print summary
    print_training_summary(states)

    # Generate plots
    print("\nGenerating plots...")
    plot_training_loss(states, output_dir / "training_loss.png")
    plot_learning_rate(states, output_dir / "learning_rate.png")
    plot_rewards(states, output_dir / "rewards.png")
    plot_gradient_norm(states, output_dir / "gradient_norm.png")

    print(f"\n✓ All plots saved to {output_dir}/")
    print("\nTo view plots, open:")
    for plot_file in output_dir.glob("*.png"):
        print(f"  {plot_file}")


if __name__ == "__main__":
    main()
