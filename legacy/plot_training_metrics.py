"""
Plot training metrics from GRPO/NSR/PSR/W-REINFORCE log files

Usage:
    python plot_training_metrics.py grpo.log
    python plot_training_metrics.py grpo.log nsr.log  # Compare multiple
"""

import re
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

def parse_log_file(log_path):
    """Extract training metrics from log file"""

    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find all metric dictionaries
    pattern = r"\{'loss':.*?\}"
    matches = re.findall(pattern, content)

    if not matches:
        print(f"Warning: No training metrics found in {log_path}")
        return None

    metrics = {
        'step': [],
        'epoch': [],
        'loss': [],
        'grad_norm': [],
        'learning_rate': [],
        'total_reward': [],
        'reward_std': [],
        'format_reward': [],
        'accuracy_reward': [],
        'chart_type_reward': [],
        'length_think_reward': [],
        'num_token_reward': [],
        'table_style_reward': [],
        'process_style_reward': [],
        'entropy': [],
        'mean_length': [],
    }

    for i, match in enumerate(matches, 1):
        # Convert string to dict
        try:
            # Replace single quotes with double quotes for JSON parsing
            # But we'll use eval for simplicity (safe here since it's our own log)
            metric_dict = eval(match)

            metrics['step'].append(i)
            metrics['epoch'].append(metric_dict.get('epoch', 0))
            metrics['loss'].append(metric_dict.get('loss', 0))
            metrics['grad_norm'].append(metric_dict.get('grad_norm', 0))
            metrics['learning_rate'].append(metric_dict.get('learning_rate', 0))
            metrics['total_reward'].append(metric_dict.get('reward', 0))
            metrics['reward_std'].append(metric_dict.get('reward_std', 0))
            metrics['format_reward'].append(metric_dict.get('rewards/format_reward/mean', 0))
            metrics['accuracy_reward'].append(metric_dict.get('rewards/accuracy_reward/mean', 0))
            metrics['chart_type_reward'].append(metric_dict.get('rewards/chart_type_reward/mean', 0))
            metrics['length_think_reward'].append(metric_dict.get('rewards/length_think_reward/mean', 0))
            metrics['num_token_reward'].append(metric_dict.get('rewards/num_token_reward/mean', 0))
            metrics['table_style_reward'].append(metric_dict.get('rewards/table_style_reward/mean', 0))
            metrics['process_style_reward'].append(metric_dict.get('rewards/process_style_reward/mean', 0))
            metrics['entropy'].append(metric_dict.get('entropy', 0))
            metrics['mean_length'].append(metric_dict.get('completions/mean_length', 0))

        except Exception as e:
            print(f"Warning: Could not parse metric at position {i}: {e}")
            continue

    return metrics


def plot_single_run(metrics, method_name, save_dir='plots'):
    """Create comprehensive plots for a single training run"""

    Path(save_dir).mkdir(exist_ok=True)

    steps = metrics['step']

    # Create figure with multiple subplots
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle(f'{method_name} Training Metrics', fontsize=16, fontweight='bold')

    # 1. Total Reward
    ax = axes[0, 0]
    ax.plot(steps, metrics['total_reward'], 'b-', linewidth=2, label='Total Reward')
    ax.fill_between(steps,
                     np.array(metrics['total_reward']) - np.array(metrics['reward_std']),
                     np.array(metrics['total_reward']) + np.array(metrics['reward_std']),
                     alpha=0.3, label='±1 std')
    ax.set_xlabel('Step')
    ax.set_ylabel('Reward')
    ax.set_title('Total Reward (Higher = Better)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Accuracy Reward (MOST IMPORTANT!)
    ax = axes[0, 1]
    ax.plot(steps, metrics['accuracy_reward'], 'g-', linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Step')
    ax.set_ylabel('Accuracy Reward')
    ax.set_title('Accuracy Reward (MOST IMPORTANT!)', fontweight='bold', color='darkgreen')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    # Add final value annotation
    final_acc = metrics['accuracy_reward'][-1]
    ax.axhline(y=final_acc, color='r', linestyle='--', alpha=0.5, label=f'Final: {final_acc:.2%}')
    ax.legend()

    # 3. Loss (can be negative in GRPO!)
    ax = axes[0, 2]
    loss_values = metrics['loss']
    colors = ['r' if l >= 0 else 'b' for l in loss_values]
    ax.scatter(steps, loss_values, c=colors, s=50, alpha=0.6)
    ax.plot(steps, loss_values, 'k--', alpha=0.3, linewidth=1)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Loss (Can be Negative in GRPO!)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.text(0.05, 0.95, 'Red = Positive\nBlue = Negative',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 4. Component Rewards
    ax = axes[1, 0]
    ax.plot(steps, metrics['format_reward'], label='Format', linewidth=2)
    ax.plot(steps, metrics['chart_type_reward'], label='Chart Type', linewidth=2)
    ax.plot(steps, metrics['accuracy_reward'], label='Accuracy', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Reward')
    ax.set_title('Main Reward Components', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Style Rewards
    ax = axes[1, 1]
    ax.plot(steps, metrics['table_style_reward'], label='Table Style', linewidth=2)
    ax.plot(steps, metrics['process_style_reward'], label='Process Style', linewidth=2)
    ax.plot(steps, metrics['length_think_reward'], label='Length', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Reward')
    ax.set_title('Style & Format Rewards', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Gradient Norm (Health Check)
    ax = axes[1, 2]
    ax.plot(steps, metrics['grad_norm'], 'purple', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Norm (Should be > 0)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Zero (BAD!)')
    ax.legend()

    # 7. Entropy (Exploration)
    ax = axes[2, 0]
    ax.plot(steps, metrics['entropy'], 'orange', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Entropy')
    ax.set_title('Entropy (Exploration)', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 8. Learning Rate Schedule
    ax = axes[2, 1]
    ax.plot(steps, metrics['learning_rate'], 'brown', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    # 9. Mean Completion Length
    ax = axes[2, 2]
    ax.plot(steps, metrics['mean_length'], 'teal', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Tokens')
    ax.set_title('Mean Completion Length', fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    save_path = Path(save_dir) / f'{method_name.lower().replace(" ", "_")}_metrics.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")

    # Also create a simplified "key metrics" plot
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
    fig2.suptitle(f'{method_name} - Key Metrics', fontsize=14, fontweight='bold')

    # Key metric 1: Accuracy
    axes2[0, 0].plot(steps, metrics['accuracy_reward'], 'g-', linewidth=3, marker='o', markersize=6)
    axes2[0, 0].set_xlabel('Step', fontsize=12)
    axes2[0, 0].set_ylabel('Accuracy Reward', fontsize=12)
    axes2[0, 0].set_title('Accuracy (MOST IMPORTANT)', fontsize=12, fontweight='bold')
    axes2[0, 0].grid(True, alpha=0.3)
    axes2[0, 0].set_ylim([0, 1])
    final_acc = metrics['accuracy_reward'][-1]
    axes2[0, 0].text(0.5, 0.95, f'Final: {final_acc:.1%}',
                     transform=axes2[0, 0].transAxes, fontsize=14,
                     ha='center', va='top', fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    # Key metric 2: Total Reward
    axes2[0, 1].plot(steps, metrics['total_reward'], 'b-', linewidth=3)
    axes2[0, 1].fill_between(steps,
                              np.array(metrics['total_reward']) - np.array(metrics['reward_std']),
                              np.array(metrics['total_reward']) + np.array(metrics['reward_std']),
                              alpha=0.3)
    axes2[0, 1].set_xlabel('Step', fontsize=12)
    axes2[0, 1].set_ylabel('Total Reward', fontsize=12)
    axes2[0, 1].set_title('Total Reward ± Std', fontsize=12, fontweight='bold')
    axes2[0, 1].grid(True, alpha=0.3)

    # Key metric 3: All Component Rewards Stacked
    axes2[1, 0].plot(steps, metrics['format_reward'], label='Format', linewidth=2)
    axes2[1, 0].plot(steps, metrics['accuracy_reward'], label='Accuracy', linewidth=2)
    axes2[1, 0].plot(steps, metrics['chart_type_reward'], label='Chart Type', linewidth=2)
    axes2[1, 0].plot(steps, metrics['table_style_reward'], label='Table Style', linewidth=2)
    axes2[1, 0].set_xlabel('Step', fontsize=12)
    axes2[1, 0].set_ylabel('Reward Component', fontsize=12)
    axes2[1, 0].set_title('Reward Breakdown', fontsize=12, fontweight='bold')
    axes2[1, 0].legend(fontsize=9)
    axes2[1, 0].grid(True, alpha=0.3)

    # Key metric 4: Training Health (Grad Norm)
    axes2[1, 1].plot(steps, metrics['grad_norm'], 'purple', linewidth=3)
    axes2[1, 1].set_xlabel('Step', fontsize=12)
    axes2[1, 1].set_ylabel('Gradient Norm', fontsize=12)
    axes2[1, 1].set_title('Training Health (Should be > 0)', fontsize=12, fontweight='bold')
    axes2[1, 1].grid(True, alpha=0.3)
    axes2[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)

    plt.tight_layout()

    save_path2 = Path(save_dir) / f'{method_name.lower().replace(" ", "_")}_key_metrics.png'
    plt.savefig(save_path2, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path2}")

    return fig, fig2


def plot_comparison(all_metrics, save_dir='plots'):
    """Compare multiple training runs"""

    Path(save_dir).mkdir(exist_ok=True)

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Comparison', fontsize=16, fontweight='bold')

    colors = ['blue', 'green', 'red', 'purple', 'orange']

    for idx, (method_name, metrics) in enumerate(all_metrics.items()):
        color = colors[idx % len(colors)]
        steps = metrics['step']

        # 1. Accuracy Comparison
        axes[0, 0].plot(steps, metrics['accuracy_reward'],
                        label=method_name, linewidth=2, marker='o',
                        markersize=4, color=color)

        # 2. Total Reward Comparison
        axes[0, 1].plot(steps, metrics['total_reward'],
                        label=method_name, linewidth=2, color=color)

        # 3. Loss Comparison
        axes[1, 0].plot(steps, metrics['loss'],
                        label=method_name, linewidth=2, color=color)

        # 4. Gradient Norm Comparison
        axes[1, 1].plot(steps, metrics['grad_norm'],
                        label=method_name, linewidth=2, color=color)

    # Configure subplots
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Accuracy Reward')
    axes[0, 0].set_title('Accuracy Comparison (MOST IMPORTANT)', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1])

    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Total Reward')
    axes[0, 1].set_title('Total Reward Comparison', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Loss Comparison', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Gradient Norm')
    axes[1, 1].set_title('Gradient Norm Comparison', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = Path(save_dir) / 'comparison_all_methods.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")

    # Create final accuracy bar chart
    fig2, ax = plt.subplots(figsize=(10, 6))

    methods = list(all_metrics.keys())
    final_accuracies = [metrics['accuracy_reward'][-1] for metrics in all_metrics.values()]

    bars = ax.bar(methods, final_accuracies, color=colors[:len(methods)], alpha=0.7, edgecolor='black')
    ax.set_ylabel('Final Accuracy Reward', fontsize=12)
    ax.set_title('Final Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, acc in zip(bars, final_accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1%}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()

    save_path2 = Path(save_dir) / 'final_accuracy_comparison.png'
    plt.savefig(save_path2, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path2}")

    return fig, fig2


def print_summary_stats(metrics, method_name):
    """Print summary statistics"""

    print(f"\n{'='*60}")
    print(f"SUMMARY: {method_name}")
    print(f"{'='*60}")

    print(f"\nAccuracy:")
    print(f"  Initial:  {metrics['accuracy_reward'][0]:.1%}")
    print(f"  Peak:     {max(metrics['accuracy_reward']):.1%} (step {metrics['step'][np.argmax(metrics['accuracy_reward'])]})")
    print(f"  Final:    {metrics['accuracy_reward'][-1]:.1%}")
    print(f"  Change:   {(metrics['accuracy_reward'][-1] - metrics['accuracy_reward'][0]):.1%}")

    print(f"\nTotal Reward:")
    print(f"  Initial:  {metrics['total_reward'][0]:.2f}")
    print(f"  Peak:     {max(metrics['total_reward']):.2f} (step {metrics['step'][np.argmax(metrics['total_reward'])]})")
    print(f"  Final:    {metrics['total_reward'][-1]:.2f}")
    print(f"  Mean:     {np.mean(metrics['total_reward']):.2f} ± {np.std(metrics['total_reward']):.2f}")

    print(f"\nLoss:")
    print(f"  Initial:  {metrics['loss'][0]:.4f}")
    print(f"  Final:    {metrics['loss'][-1]:.4f}")
    print(f"  Mean:     {np.mean(metrics['loss']):.4f}")
    print(f"  Note:     {'NEGATIVE values are normal in GRPO!' if any(l < 0 for l in metrics['loss']) else ''}")

    print(f"\nTraining Health:")
    print(f"  Grad Norm (mean):  {np.mean(metrics['grad_norm']):.4f}")
    print(f"  Grad Norm (final): {metrics['grad_norm'][-1]:.4f}")
    print(f"  Entropy (mean):    {np.mean(metrics['entropy']):.4f}")
    print(f"  Status:            {'✓ Healthy' if np.mean(metrics['grad_norm']) > 0.01 else '⚠ Check gradients'}")

    print(f"\n{'='*60}\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_training_metrics.py <log_file1> [log_file2] [log_file3] ...")
        print("\nExample:")
        print("  python plot_training_metrics.py grpo.log")
        print("  python plot_training_metrics.py grpo.log nsr.log")
        sys.exit(1)

    log_files = sys.argv[1:]
    all_metrics = {}

    print(f"\n{'='*60}")
    print("PARSING LOG FILES")
    print(f"{'='*60}\n")

    for log_file in log_files:
        if not Path(log_file).exists():
            print(f"Error: {log_file} not found")
            continue

        # Infer method name from filename
        method_name = Path(log_file).stem.upper()
        if 'grpo' in log_file.lower():
            method_name = 'GRPO'
        elif 'nsr' in log_file.lower():
            method_name = 'NSR'
        elif 'psr' in log_file.lower():
            method_name = 'PSR'
        elif 'wreinforce' in log_file.lower() or 'w-reinforce' in log_file.lower():
            method_name = 'W-REINFORCE'

        print(f"Parsing {log_file}... ({method_name})")

        metrics = parse_log_file(log_file)

        if metrics is None or len(metrics['step']) == 0:
            print(f"  ✗ No metrics found in {log_file}")
            continue

        print(f"  ✓ Found {len(metrics['step'])} training steps")

        all_metrics[method_name] = metrics

        # Print summary stats
        print_summary_stats(metrics, method_name)

        # Create individual plots
        plot_single_run(metrics, method_name)

    # Create comparison plots if multiple runs
    if len(all_metrics) > 1:
        print(f"\n{'='*60}")
        print("CREATING COMPARISON PLOTS")
        print(f"{'='*60}\n")
        plot_comparison(all_metrics)

    print(f"\n{'='*60}")
    print("✓ ALL PLOTS GENERATED!")
    print(f"{'='*60}")
    print(f"\nPlots saved to: ./plots/")
    print(f"\nGenerated files:")
    for method_name in all_metrics.keys():
        print(f"  - {method_name.lower().replace(' ', '_')}_metrics.png (detailed)")
        print(f"  - {method_name.lower().replace(' ', '_')}_key_metrics.png (simplified)")
    if len(all_metrics) > 1:
        print(f"  - comparison_all_methods.png")
        print(f"  - final_accuracy_comparison.png")
    print()


if __name__ == '__main__':
    main()
