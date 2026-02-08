#!/usr/bin/env python3
"""
Compare results across experiments.

Usage:
    python -m app.scripts.compare_experiments --output-dir outputs

This will:
1. Load metrics from all experiment runs
2. Generate comparison tables
3. Create ASCII visualizations
4. Output recommendations
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any
import sys


def load_experiment_metrics(output_dir: Path) -> Dict[str, Dict]:
    """Load metrics from all experiments."""
    results = {}

    for exp_dir in output_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        exp_name = exp_dir.name
        if not any(x in exp_name for x in ["grpo", "nsr", "reinforce"]):
            continue

        # Find latest run
        runs = sorted(exp_dir.glob("run_*"))
        if not runs:
            continue

        latest_run = runs[-1]

        # Load metrics
        metrics_file = latest_run / "metrics.jsonl"
        if not metrics_file.exists():
            continue

        metrics = []
        with open(metrics_file) as f:
            for line in f:
                if line.strip():
                    try:
                        metrics.append(json.loads(line))
                    except:
                        continue

        if metrics:
            results[exp_name] = {
                "run_dir": str(latest_run),
                "num_steps": len(metrics),
                "metrics": metrics,
                "final": metrics[-1] if metrics else {},
            }

    return results


def compute_experiment_summary(metrics: List[Dict]) -> Dict[str, float]:
    """Compute summary statistics for an experiment."""
    if not metrics:
        return {}

    # Get last 20% of training for "final" performance
    final_portion = metrics[int(len(metrics) * 0.8):]

    def avg(key):
        vals = [m.get(key, 0) for m in final_portion]
        return sum(vals) / len(vals) if vals else 0

    return {
        "final_accuracy": avg("avg_base_accuracy"),
        "final_format": avg("avg_base_format"),
        "final_total": avg("avg_total"),
        "final_table": avg("avg_base_table"),
        "final_type": avg("avg_base_type"),
        "final_hcpc": avg("avg_hcpc"),
        "num_steps": len(metrics),
    }


def print_comparison_table(results: Dict[str, Dict]):
    """Print comparison table."""
    print("\n" + "=" * 90)
    print("EXPERIMENT COMPARISON")
    print("=" * 90)

    # Header
    header = f"{'Experiment':<20} {'Steps':>6} {'Accuracy':>10} {'Format':>10} {'Table':>10} {'Total':>10}"
    print(header)
    print("-" * 90)

    summaries = {}
    for exp_name, data in sorted(results.items()):
        summary = compute_experiment_summary(data["metrics"])
        summaries[exp_name] = summary

        print(
            f"{exp_name:<20} "
            f"{summary.get('num_steps', 0):>6} "
            f"{summary.get('final_accuracy', 0):>10.3f} "
            f"{summary.get('final_format', 0):>10.3f} "
            f"{summary.get('final_table', 0):>10.3f} "
            f"{summary.get('final_total', 0):>10.3f}"
        )

    print("=" * 90)

    return summaries


def print_winner_analysis(summaries: Dict[str, Dict]):
    """Determine and explain the winner."""
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    if not summaries:
        print("No experiments to compare.")
        return

    # Find best by accuracy
    best_acc = max(summaries.items(), key=lambda x: x[1].get("final_accuracy", 0))
    print(f"\n📊 Best Accuracy: {best_acc[0]} ({best_acc[1].get('final_accuracy', 0):.3f})")

    # Find best by total reward
    best_total = max(summaries.items(), key=lambda x: x[1].get("final_total", 0))
    print(f"📈 Best Total Reward: {best_total[0]} ({best_total[1].get('final_total', 0):.3f})")

    # Compare GRPO vs NSR
    grpo_exps = {k: v for k, v in summaries.items() if "grpo" in k.lower()}
    nsr_exps = {k: v for k, v in summaries.items() if "nsr" in k.lower()}

    if grpo_exps and nsr_exps:
        grpo_avg_acc = sum(v.get("final_accuracy", 0) for v in grpo_exps.values()) / len(grpo_exps)
        nsr_avg_acc = sum(v.get("final_accuracy", 0) for v in nsr_exps.values()) / len(nsr_exps)

        print(f"\n🔬 GRPO avg accuracy: {grpo_avg_acc:.3f}")
        print(f"🔬 NSR avg accuracy: {nsr_avg_acc:.3f}")

        if nsr_avg_acc > grpo_avg_acc:
            print("   → NSR methods performing better (supports thesis!)")
        else:
            print("   → GRPO methods performing better")

    # Compare with/without HCPC
    hcpc_exps = {k: v for k, v in summaries.items() if "hcpc" in k.lower()}
    base_exps = {k: v for k, v in summaries.items() if "baseline" in k.lower()}

    if hcpc_exps and base_exps:
        hcpc_avg = sum(v.get("final_accuracy", 0) for v in hcpc_exps.values()) / len(hcpc_exps)
        base_avg = sum(v.get("final_accuracy", 0) for v in base_exps.values()) / len(base_exps)

        print(f"\n🎯 Baseline avg accuracy: {base_avg:.3f}")
        print(f"🎯 HCPC avg accuracy: {hcpc_avg:.3f}")

        if hcpc_avg > base_avg:
            improvement = ((hcpc_avg - base_avg) / base_avg * 100) if base_avg > 0 else 0
            print(f"   → HCPC improves by {improvement:.1f}% (supports thesis!)")
        else:
            print("   → HCPC not showing improvement yet")


def print_recommendations(summaries: Dict[str, Dict]):
    """Print recommendations based on results."""
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)

    recs = []

    # Check if format is low across all
    avg_format = sum(s.get("final_format", 0) for s in summaries.values()) / len(summaries) if summaries else 0
    if avg_format < 0.8:
        recs.append(
            "⚠️ Format scores are low across experiments.\n"
            "   Consider: Adjusting system prompt, increasing format reward weight."
        )

    # Check if accuracy is low
    avg_acc = sum(s.get("final_accuracy", 0) for s in summaries.values()) / len(summaries) if summaries else 0
    if avg_acc < 0.3:
        recs.append(
            "⚠️ Accuracy is low across experiments.\n"
            "   Consider: More training steps, easier data subset, lower learning rate."
        )

    # Check if HCPC is showing any effect
    hcpc_exps = [v for k, v in summaries.items() if "hcpc" in k.lower()]
    if hcpc_exps:
        avg_hcpc = sum(s.get("final_hcpc", 0) for s in hcpc_exps) / len(hcpc_exps)
        if avg_hcpc < 0.1:
            recs.append(
                "⚠️ HCPC reward is very low.\n"
                "   This means few rollouts are meeting the 'correct' threshold.\n"
                "   Consider: Lower table_sim_threshold, or more training first."
            )

    if not recs:
        recs.append("✅ Training looks reasonable. Continue to gather more data.")

    for rec in recs:
        print(f"\n{rec}")


def print_ascii_comparison(summaries: Dict[str, Dict], metric: str = "final_accuracy"):
    """Print ASCII bar chart comparison."""
    if not summaries:
        return

    print(f"\n{metric.upper()} COMPARISON")
    print("-" * 50)

    max_val = max(s.get(metric, 0) for s in summaries.values())
    if max_val == 0:
        max_val = 1

    bar_width = 40

    for exp_name in sorted(summaries.keys()):
        val = summaries[exp_name].get(metric, 0)
        bar_len = int((val / max_val) * bar_width)
        bar = "█" * bar_len + "░" * (bar_width - bar_len)
        print(f"{exp_name:<20} {bar} {val:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Compare experiment results")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory with experiments")
    parser.add_argument("--format", choices=["text", "json"], default="text")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Output directory not found: {output_dir}")
        sys.exit(1)

    # Load all experiments
    print(f"Loading experiments from {output_dir}...")
    results = load_experiment_metrics(output_dir)

    if not results:
        print("No experiment results found.")
        print("Run experiments first with: python -m app.scripts.run_experiments")
        sys.exit(1)

    print(f"Found {len(results)} experiments")

    if args.format == "json":
        summaries = {k: compute_experiment_summary(v["metrics"]) for k, v in results.items()}
        print(json.dumps(summaries, indent=2))
    else:
        # Print comparison
        summaries = print_comparison_table(results)

        # ASCII charts
        print_ascii_comparison(summaries, "final_accuracy")
        print_ascii_comparison(summaries, "final_total")

        # Analysis
        print_winner_analysis(summaries)

        # Recommendations
        print_recommendations(summaries)


if __name__ == "__main__":
    main()
