#!/usr/bin/env python3
"""
Analyze training logs and visualize convergence.

Usage:
    python -m app.scripts.analyze_training --run outputs/grpo_baseline/run_20260208_093000
    python -m app.scripts.analyze_training --log outputs/grpo_baseline/run_xxx/metrics.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import sys


def load_metrics(path: Path) -> List[Dict[str, Any]]:
    """Load metrics from JSONL file."""
    metrics = []
    with open(path) as f:
        for line in f:
            if line.strip():
                try:
                    metrics.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return metrics


def load_completions_log(path: Path) -> List[Dict[str, Any]]:
    """Parse completion logs from train.log file."""
    completions = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if "[completion_log" in line:
                # Extract key info
                try:
                    parts = line.split("|")
                    if len(parts) >= 3:
                        log_part = parts[-1].strip()
                        completions.append({"raw": log_part[:500]})
                except Exception:
                    continue
    return completions


def analyze_rewards(metrics: List[Dict]) -> Dict[str, Any]:
    """Analyze reward patterns."""
    if not metrics:
        return {"error": "No metrics"}

    # Extract key metrics over time
    steps = []
    accuracy = []
    format_scores = []
    total_rewards = []
    table_scores = []
    chart_type_scores = []

    for m in metrics:
        steps.append(m.get("step", len(steps)))
        accuracy.append(m.get("avg_base_accuracy", m.get("batch_accuracy", 0)))
        format_scores.append(m.get("avg_base_format", m.get("batch_format", 0)))
        total_rewards.append(m.get("avg_total", m.get("batch_total", 0)))
        table_scores.append(m.get("avg_base_table", m.get("batch_table", 0)))
        chart_type_scores.append(m.get("avg_base_type", m.get("batch_chart_type", 0)))

    # Compute trends
    def compute_trend(values):
        if len(values) < 10:
            return "insufficient_data"
        first_half = sum(values[:len(values)//2]) / (len(values)//2)
        second_half = sum(values[len(values)//2:]) / (len(values) - len(values)//2)
        diff = second_half - first_half
        if diff > 0.05:
            return "improving"
        elif diff < -0.05:
            return "declining"
        return "stable"

    return {
        "total_steps": len(metrics),
        "final_metrics": {
            "accuracy": accuracy[-1] if accuracy else 0,
            "format": format_scores[-1] if format_scores else 0,
            "total_reward": total_rewards[-1] if total_rewards else 0,
            "table": table_scores[-1] if table_scores else 0,
            "chart_type": chart_type_scores[-1] if chart_type_scores else 0,
        },
        "averages": {
            "accuracy": sum(accuracy) / len(accuracy) if accuracy else 0,
            "format": sum(format_scores) / len(format_scores) if format_scores else 0,
            "total_reward": sum(total_rewards) / len(total_rewards) if total_rewards else 0,
        },
        "trends": {
            "accuracy": compute_trend(accuracy),
            "format": compute_trend(format_scores),
            "total_reward": compute_trend(total_rewards),
        },
        "progression": {
            "steps": steps[::max(1, len(steps)//20)],  # Sample 20 points
            "accuracy": accuracy[::max(1, len(accuracy)//20)],
            "format": format_scores[::max(1, len(format_scores)//20)],
            "total_reward": total_rewards[::max(1, len(total_rewards)//20)],
        }
    }


def print_ascii_chart(values: List[float], title: str, width: int = 60, height: int = 10):
    """Print simple ASCII chart."""
    if not values:
        print(f"{title}: No data")
        return

    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val if max_val != min_val else 1

    print(f"\n{title}")
    print(f"Max: {max_val:.3f}")

    # Sample values to fit width
    if len(values) > width:
        step = len(values) / width
        sampled = [values[int(i * step)] for i in range(width)]
    else:
        sampled = values

    # Create chart
    for row in range(height, -1, -1):
        threshold = min_val + (row / height) * range_val
        line = ""
        for val in sampled:
            if val >= threshold:
                line += "█"
            else:
                line += " "
        if row == height:
            print(f"{max_val:>7.2f} |{line}|")
        elif row == 0:
            print(f"{min_val:>7.2f} |{line}|")
        else:
            print(f"        |{line}|")

    print(f"         {'─' * len(sampled)}")
    print(f"         Step 0{' ' * (len(sampled) - 10)}Step {len(values)}")


def diagnose_issues(analysis: Dict) -> List[str]:
    """Diagnose training issues from analysis."""
    issues = []

    trends = analysis.get("trends", {})
    avgs = analysis.get("averages", {})
    final = analysis.get("final_metrics", {})

    # Check format compliance
    if avgs.get("format", 0) < 0.5:
        issues.append(
            "❌ LOW FORMAT COMPLIANCE: Model not outputting required tags.\n"
            "   Cause: Model ignoring format instructions in system prompt.\n"
            "   Fix: Increase format_reward weight or add format examples to prompt."
        )
    elif avgs.get("format", 0) < 1.0:
        issues.append(
            "⚠️  PARTIAL FORMAT: Model has some tags but not all.\n"
            "   Cause: Model learning format incrementally.\n"
            "   Fix: Continue training, partial rewards should help."
        )

    # Check accuracy
    if avgs.get("accuracy", 0) < 0.2:
        issues.append(
            "❌ LOW ACCURACY: Model getting wrong answers.\n"
            "   Cause: Task too hard or data quality issues.\n"
            "   Fix: Check data labels, consider easier subset first."
        )

    # Check trends
    if trends.get("accuracy") == "declining":
        issues.append(
            "❌ ACCURACY DECLINING: Model getting worse over time.\n"
            "   Cause: Overfitting, mode collapse, or gradient issues.\n"
            "   Fix: Reduce learning rate, increase KL penalty, or add regularization."
        )

    if trends.get("total_reward") == "declining":
        issues.append(
            "⚠️  REWARD DECLINING: Total reward going down.\n"
            "   Cause: Model may be sacrificing some components for others.\n"
            "   Fix: Check individual reward component trends."
        )

    # Positive signs
    if trends.get("accuracy") == "improving" and avgs.get("accuracy", 0) > 0.3:
        issues.append(
            "✅ ACCURACY IMPROVING: Model learning to answer correctly."
        )

    if trends.get("format") == "improving":
        issues.append(
            "✅ FORMAT IMPROVING: Model learning output structure."
        )

    if not issues:
        issues.append("No specific issues detected. Continue training and monitor.")

    return issues


def main():
    parser = argparse.ArgumentParser(description="Analyze HCPC-RLVR training logs")
    parser.add_argument("--run", type=str, help="Path to run directory")
    parser.add_argument("--log", type=str, help="Path to metrics.jsonl file")
    parser.add_argument("--format", choices=["text", "json"], default="text")

    args = parser.parse_args()

    # Find metrics file
    if args.log:
        metrics_path = Path(args.log)
    elif args.run:
        run_dir = Path(args.run)
        metrics_path = run_dir / "metrics.jsonl"
        if not metrics_path.exists():
            # Try train.log parsing
            train_log = run_dir / "train.log"
            if train_log.exists():
                print(f"Parsing train.log from {train_log}")
                # Would need more sophisticated parsing
    else:
        print("Please provide --run or --log path")
        sys.exit(1)

    if not metrics_path.exists():
        print(f"Metrics file not found: {metrics_path}")
        sys.exit(1)

    # Load and analyze
    print(f"Loading metrics from {metrics_path}...")
    metrics = load_metrics(metrics_path)
    print(f"Found {len(metrics)} steps\n")

    analysis = analyze_rewards(metrics)

    if args.format == "json":
        print(json.dumps(analysis, indent=2))
    else:
        # Print summary
        print("=" * 60)
        print("TRAINING ANALYSIS SUMMARY")
        print("=" * 60)

        print(f"\nTotal Steps: {analysis['total_steps']}")

        print("\n--- Final Metrics ---")
        for k, v in analysis.get("final_metrics", {}).items():
            print(f"  {k}: {v:.3f}")

        print("\n--- Averages ---")
        for k, v in analysis.get("averages", {}).items():
            print(f"  {k}: {v:.3f}")

        print("\n--- Trends ---")
        for k, v in analysis.get("trends", {}).items():
            emoji = "📈" if v == "improving" else "📉" if v == "declining" else "➡️"
            print(f"  {emoji} {k}: {v}")

        # ASCII charts
        prog = analysis.get("progression", {})
        if prog.get("accuracy"):
            print_ascii_chart(prog["accuracy"], "Accuracy Over Time")
        if prog.get("format"):
            print_ascii_chart(prog["format"], "Format Score Over Time")
        if prog.get("total_reward"):
            print_ascii_chart(prog["total_reward"], "Total Reward Over Time")

        # Diagnosis
        print("\n" + "=" * 60)
        print("DIAGNOSIS")
        print("=" * 60)
        issues = diagnose_issues(analysis)
        for issue in issues:
            print(f"\n{issue}")

        print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
