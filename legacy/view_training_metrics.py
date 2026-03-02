"""
View training metrics from log files (no plotting required)
Simple text-based view of all metrics

Usage:
    python view_training_metrics.py grpo.log
    python view_training_metrics.py grpo.log nsr.log
"""

import re
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

    metrics = []

    for i, match in enumerate(matches, 1):
        try:
            metric_dict = eval(match)
            metric_dict['step'] = i
            metrics.append(metric_dict)
        except Exception as e:
            print(f"Warning: Could not parse metric at position {i}: {e}")
            continue

    return metrics


def print_metrics_table(metrics, method_name):
    """Print metrics in a nice table format"""

    print(f"\n{'='*120}")
    print(f"{method_name} - TRAINING METRICS")
    print(f"{'='*120}\n")

    # Header
    header = f"{'Step':<6} {'Loss':>8} {'Reward':>8} {'Acc':>6} {'Format':>6} {'Chart':>6} {'GradN':>8} {'LR':>10} {'Length':>8}"
    print(header)
    print("-" * 120)

    # Data rows
    for m in metrics:
        row = (
            f"{m.get('step', 0):<6} "
            f"{m.get('loss', 0):>8.4f} "
            f"{m.get('reward', 0):>8.2f} "
            f"{m.get('rewards/accuracy_reward/mean', 0):>6.1%} "
            f"{m.get('rewards/format_reward/mean', 0):>6.1f} "
            f"{m.get('rewards/chart_type_reward/mean', 0):>6.1%} "
            f"{m.get('grad_norm', 0):>8.4f} "
            f"{m.get('learning_rate', 0):>10.2e} "
            f"{m.get('completions/mean_length', 0):>8.1f}"
        )
        print(row)

    print("-" * 120)
    print()


def print_summary(metrics, method_name):
    """Print summary statistics"""

    if not metrics:
        return

    accuracies = [m.get('rewards/accuracy_reward/mean', 0) for m in metrics]
    rewards = [m.get('reward', 0) for m in metrics]
    losses = [m.get('loss', 0) for m in metrics]
    grad_norms = [m.get('grad_norm', 0) for m in metrics]

    print(f"\n{'='*80}")
    print(f"SUMMARY: {method_name}")
    print(f"{'='*80}")

    # Accuracy
    print(f"\n[ACCURACY] (Most Important!):")
    print(f"   Initial:     {accuracies[0]:>6.1%}")
    print(f"   Peak:        {max(accuracies):>6.1%}  (step {accuracies.index(max(accuracies)) + 1})")
    print(f"   Final:       {accuracies[-1]:>6.1%}")
    print(f"   Improvement: {(accuracies[-1] - accuracies[0]):>6.1%}")

    # Total Reward
    print(f"\n[TOTAL REWARD]:")
    print(f"   Initial:     {rewards[0]:>6.2f}")
    print(f"   Peak:        {max(rewards):>6.2f}  (step {rewards.index(max(rewards)) + 1})")
    print(f"   Final:       {rewards[-1]:>6.2f}")
    print(f"   Mean:        {sum(rewards)/len(rewards):>6.2f} +/- {(sum((r - sum(rewards)/len(rewards))**2 for r in rewards)/len(rewards))**0.5:.2f}")

    # Loss
    print(f"\n[LOSS]:")
    print(f"   Initial:     {losses[0]:>8.4f}")
    print(f"   Final:       {losses[-1]:>8.4f}")
    print(f"   Mean:        {sum(losses)/len(losses):>8.4f}")
    if any(l < 0 for l in losses):
        print(f"   Note:        [OK] Negative values are NORMAL in GRPO!")

    # Training Health
    print(f"\n[TRAINING HEALTH]:")
    print(f"   Grad Norm (mean):  {sum(grad_norms)/len(grad_norms):.4f}")
    print(f"   Grad Norm (final): {grad_norms[-1]:.4f}")
    avg_grad = sum(grad_norms)/len(grad_norms)
    status = "[OK] Healthy" if avg_grad > 0.01 else "[WARNING] Check gradients"
    print(f"   Status:            {status}")

    # Component Rewards
    print(f"\n[REWARD COMPONENTS] (Final Step):")
    final = metrics[-1]
    print(f"   Format:       {final.get('rewards/format_reward/mean', 0):>5.2f}")
    print(f"   Accuracy:     {final.get('rewards/accuracy_reward/mean', 0):>5.2f}")
    print(f"   Chart Type:   {final.get('rewards/chart_type_reward/mean', 0):>5.2f}")
    print(f"   Length:       {final.get('rewards/length_think_reward/mean', 0):>5.2f}")
    print(f"   Num Tokens:   {final.get('rewards/num_token_reward/mean', 0):>5.2f}")
    print(f"   Table Style:  {final.get('rewards/table_style_reward/mean', 0):>5.2f}")
    print(f"   Process:      {final.get('rewards/process_style_reward/mean', 0):>5.2f}")
    print(f"   TOTAL:        {final.get('reward', 0):>5.2f}")

    print(f"\n{'='*80}\n")


def compare_methods(all_metrics):
    """Print comparison table"""

    if len(all_metrics) < 2:
        return

    print(f"\n{'='*80}")
    print(f"METHOD COMPARISON")
    print(f"{'='*80}\n")

    # Table header
    print(f"{'Method':<15} {'Steps':>8} {'Final Acc':>12} {'Final Reward':>15} {'Avg Grad Norm':>15}")
    print("-" * 80)

    # Data rows
    for method_name, metrics in all_metrics.items():
        if not metrics:
            continue

        final_acc = metrics[-1].get('rewards/accuracy_reward/mean', 0)
        final_reward = metrics[-1].get('reward', 0)
        grad_norms = [m.get('grad_norm', 0) for m in metrics]
        avg_grad = sum(grad_norms) / len(grad_norms)

        print(f"{method_name:<15} {len(metrics):>8} {final_acc:>11.1%} {final_reward:>15.2f} {avg_grad:>15.4f}")

    print("-" * 80)

    # Winner
    best_method = max(all_metrics.items(),
                     key=lambda x: x[1][-1].get('rewards/accuracy_reward/mean', 0) if x[1] else 0)

    print(f"\n[BEST METHOD]: {best_method[0]} ({best_method[1][-1].get('rewards/accuracy_reward/mean', 0):.1%} accuracy)\n")
    print(f"{'='*80}\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python view_training_metrics.py <log_file1> [log_file2] ...")
        print("\nExample:")
        print("  python view_training_metrics.py grpo.log")
        print("  python view_training_metrics.py grpo.log nsr.log")
        sys.exit(1)

    log_files = sys.argv[1:]
    all_metrics = {}

    for log_file in log_files:
        if not Path(log_file).exists():
            print(f"Error: {log_file} not found")
            continue

        # Infer method name
        method_name = Path(log_file).stem.upper()
        if 'grpo' in log_file.lower():
            method_name = 'GRPO'
        elif 'nsr' in log_file.lower():
            method_name = 'NSR'
        elif 'psr' in log_file.lower():
            method_name = 'PSR'
        elif 'wreinforce' in log_file.lower() or 'w-reinforce' in log_file.lower():
            method_name = 'W-REINFORCE'

        print(f"\nParsing {log_file}...")

        metrics = parse_log_file(log_file)

        if metrics is None or len(metrics) == 0:
            print(f"[ERROR] No metrics found in {log_file}")
            continue

        print(f"[OK] Found {len(metrics)} training steps")

        all_metrics[method_name] = metrics

        # Print table and summary
        print_metrics_table(metrics, method_name)
        print_summary(metrics, method_name)

    # Print comparison if multiple methods
    if len(all_metrics) > 1:
        compare_methods(all_metrics)

    print("\n" + "="*80)
    print("[OK] ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nTo generate plots, install matplotlib:")
    print(f"  pip install matplotlib numpy")
    print(f"Then run:")
    print(f"  python plot_training_metrics.py {' '.join(log_files)}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
