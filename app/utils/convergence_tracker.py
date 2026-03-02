"""Convergence tracking and analysis for HCPC-RLVR training."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import statistics


@dataclass
class ConvergenceMetrics:
    """Metrics for tracking convergence."""
    step: int
    # Rolling averages
    avg_total_reward: float = 0.0
    avg_accuracy: float = 0.0
    avg_format: float = 0.0
    avg_token_count: float = 0.0
    avg_chart_type: float = 0.0
    avg_table: float = 0.0
    avg_process: float = 0.0
    avg_hcpc: float = 0.0
    # Variance (for stability)
    var_total_reward: float = 0.0
    var_accuracy: float = 0.0
    # Trends
    reward_trend: str = "unknown"  # improving, stable, declining
    accuracy_trend: str = "unknown"
    # Convergence indicators
    is_converging: bool = False
    convergence_score: float = 0.0


class ConvergenceTracker:
    """
    Tracks training convergence over time.

    Provides:
    - Rolling averages and variance
    - Trend detection (improving/stable/declining)
    - Convergence scoring
    - Early stopping recommendations
    """

    def __init__(
        self,
        window_size: int = 50,
        trend_window: int = 20,
        stability_threshold: float = 0.1,
        log_file: Optional[Path] = None,
    ):
        """
        Initialize convergence tracker.

        Args:
            window_size: Window for rolling statistics
            trend_window: Window for trend detection
            stability_threshold: Variance threshold for "stable"
            log_file: Optional path to write detailed logs
        """
        self.window_size = window_size
        self.trend_window = trend_window
        self.stability_threshold = stability_threshold
        self.log_file = log_file

        # History buffers
        self.reward_history: deque = deque(maxlen=window_size)
        self.accuracy_history: deque = deque(maxlen=window_size)
        self.format_history: deque = deque(maxlen=window_size)
        self.token_count_history: deque = deque(maxlen=window_size)
        self.chart_type_history: deque = deque(maxlen=window_size)
        self.table_history: deque = deque(maxlen=window_size)
        self.process_history: deque = deque(maxlen=window_size)
        self.hcpc_history: deque = deque(maxlen=window_size)

        # Full history for analysis
        self.all_metrics: List[Dict[str, Any]] = []
        self.step = 0

    def update(self, rewards_breakdown: List[Dict[str, float]]) -> ConvergenceMetrics:
        """
        Update tracker with new batch of rewards.

        Args:
            rewards_breakdown: List of reward dicts from aggregator

        Returns:
            Current convergence metrics
        """
        self.step += 1

        # Compute batch averages
        if not rewards_breakdown:
            return self._compute_metrics()

        batch_total = sum(rb.get("base_total", 0) + rb.get("hcpc", 0) for rb in rewards_breakdown) / len(rewards_breakdown)
        batch_accuracy = sum(rb.get("base_accuracy", 0) for rb in rewards_breakdown) / len(rewards_breakdown)
        batch_format = sum(rb.get("base_format", 0) for rb in rewards_breakdown) / len(rewards_breakdown)
        batch_token = sum(rb.get("base_token_count", 0) for rb in rewards_breakdown) / len(rewards_breakdown)
        batch_chart = sum(rb.get("base_chart_type", 0) for rb in rewards_breakdown) / len(rewards_breakdown)
        batch_table = sum(rb.get("base_table", 0) for rb in rewards_breakdown) / len(rewards_breakdown)
        batch_process = sum(rb.get("base_process", 0) for rb in rewards_breakdown) / len(rewards_breakdown)
        batch_hcpc = sum(rb.get("hcpc", 0) for rb in rewards_breakdown) / len(rewards_breakdown)

        # Update histories
        self.reward_history.append(batch_total)
        self.accuracy_history.append(batch_accuracy)
        self.format_history.append(batch_format)
        self.token_count_history.append(batch_token)
        self.chart_type_history.append(batch_chart)
        self.table_history.append(batch_table)
        self.process_history.append(batch_process)
        self.hcpc_history.append(batch_hcpc)

        # Store full metrics
        step_metrics = {
            "step": self.step,
            "batch_total": batch_total,
            "batch_accuracy": batch_accuracy,
            "batch_format": batch_format,
            "batch_token_count": batch_token,
            "batch_chart_type": batch_chart,
            "batch_table": batch_table,
            "batch_process": batch_process,
            "batch_hcpc": batch_hcpc,
            "individual_rewards": rewards_breakdown,
        }
        self.all_metrics.append(step_metrics)

        # Log to file
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(step_metrics, default=str) + "\n")

        return self._compute_metrics()

    def _compute_metrics(self) -> ConvergenceMetrics:
        """Compute current convergence metrics."""
        metrics = ConvergenceMetrics(step=self.step)

        if len(self.reward_history) < 2:
            return metrics

        # Rolling averages
        metrics.avg_total_reward = statistics.mean(self.reward_history)
        metrics.avg_accuracy = statistics.mean(self.accuracy_history)
        metrics.avg_format = statistics.mean(self.format_history)
        metrics.avg_token_count = statistics.mean(self.token_count_history)
        metrics.avg_chart_type = statistics.mean(self.chart_type_history)
        metrics.avg_table = statistics.mean(self.table_history)
        metrics.avg_process = statistics.mean(self.process_history)
        metrics.avg_hcpc = statistics.mean(self.hcpc_history)

        # Variance
        if len(self.reward_history) >= 3:
            metrics.var_total_reward = statistics.variance(self.reward_history)
            metrics.var_accuracy = statistics.variance(self.accuracy_history)

        # Trends
        metrics.reward_trend = self._compute_trend(list(self.reward_history))
        metrics.accuracy_trend = self._compute_trend(list(self.accuracy_history))

        # Convergence scoring
        metrics.convergence_score = self._compute_convergence_score(metrics)
        metrics.is_converging = metrics.convergence_score > 0.6

        return metrics

    def _compute_trend(self, values: List[float]) -> str:
        """Compute trend direction."""
        if len(values) < self.trend_window:
            return "unknown"

        recent = values[-self.trend_window:]
        first_half = statistics.mean(recent[:len(recent)//2])
        second_half = statistics.mean(recent[len(recent)//2:])

        diff = second_half - first_half
        threshold = 0.05 * abs(first_half) if first_half != 0 else 0.01

        if diff > threshold:
            return "improving"
        elif diff < -threshold:
            return "declining"
        else:
            return "stable"

    def _compute_convergence_score(self, metrics: ConvergenceMetrics) -> float:
        """
        Compute overall convergence score (0-1).

        Factors:
        - High accuracy (good)
        - Low variance (good)
        - Stable or improving trend (good)
        - High format compliance (good)
        """
        score = 0.0

        # Accuracy component (0-0.3)
        score += min(metrics.avg_accuracy, 1.0) * 0.3

        # Format compliance (0-0.2)
        score += min(metrics.avg_format / 2.0, 1.0) * 0.2

        # Token count (0-0.1)
        score += min(metrics.avg_token_count / 2.0, 1.0) * 0.1

        # Stability (0-0.2)
        if metrics.var_accuracy < self.stability_threshold:
            score += 0.2
        elif metrics.var_accuracy < self.stability_threshold * 2:
            score += 0.1

        # Trend (0-0.2)
        if metrics.reward_trend == "improving":
            score += 0.2
        elif metrics.reward_trend == "stable" and metrics.avg_accuracy > 0.5:
            score += 0.15
        elif metrics.reward_trend == "stable":
            score += 0.1

        return min(score, 1.0)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of training progress."""
        metrics = self._compute_metrics()

        return {
            "step": self.step,
            "samples_seen": self.step,
            "convergence": {
                "is_converging": metrics.is_converging,
                "score": round(metrics.convergence_score, 3),
                "reward_trend": metrics.reward_trend,
                "accuracy_trend": metrics.accuracy_trend,
            },
            "averages": {
                "total_reward": round(metrics.avg_total_reward, 3),
                "accuracy": round(metrics.avg_accuracy, 3),
                "format": round(metrics.avg_format, 3),
                "token_count": round(metrics.avg_token_count, 3),
                "chart_type": round(metrics.avg_chart_type, 3),
                "table": round(metrics.avg_table, 3),
                "process": round(metrics.avg_process, 3),
                "hcpc": round(metrics.avg_hcpc, 3),
            },
            "variance": {
                "total_reward": round(metrics.var_total_reward, 4),
                "accuracy": round(metrics.var_accuracy, 4),
            },
            "recommendations": self._get_recommendations(metrics),
        }

    def _get_recommendations(self, metrics: ConvergenceMetrics) -> List[str]:
        """Get training recommendations based on metrics."""
        recs = []

        # Format issues
        if metrics.avg_format < 0.5:
            recs.append("Low format compliance - model not using required tags. Check system prompt.")
        elif metrics.avg_format < 1.0:
            recs.append("Partial format compliance - model learning tags but order/structure needs work.")

        # Token count issues
        if metrics.avg_token_count < 0.5:
            recs.append("Token count low - model may be missing tags or duplicating them.")

        # Accuracy plateau
        if metrics.accuracy_trend == "stable" and metrics.avg_accuracy < 0.3:
            recs.append("Accuracy plateau at low level - consider increasing learning rate or checking data.")

        # Declining metrics
        if metrics.reward_trend == "declining":
            recs.append("Reward declining - possible overfitting. Consider early stopping or reducing LR.")
        if metrics.accuracy_trend == "declining":
            recs.append("Accuracy declining - check for mode collapse or gradient issues.")

        # High variance
        if metrics.var_accuracy > 0.2:
            recs.append("High accuracy variance - training unstable. Try larger batch or smaller LR.")

        # Good progress
        if metrics.is_converging and metrics.avg_accuracy > 0.5:
            recs.append("Training looks healthy! Continue monitoring.")

        return recs if recs else ["No specific recommendations - continue training."]

    def save_full_history(self, path: Path):
        """Save complete training history for analysis."""
        with open(path, "w") as f:
            json.dump({
                "steps": self.step,
                "window_size": self.window_size,
                "metrics_history": self.all_metrics,
                "final_summary": self.get_summary(),
            }, f, indent=2, default=str)

    def print_status(self, every_n: int = 10):
        """Print status update if at right step."""
        if self.step % every_n != 0:
            return

        summary = self.get_summary()

        print(f"\n{'='*60}")
        print(f"Step {self.step} - Convergence Score: {summary['convergence']['score']:.2f}")
        print(f"{'='*60}")
        print(f"Trends: reward={summary['convergence']['reward_trend']}, accuracy={summary['convergence']['accuracy_trend']}")
        print(f"Averages:")
        for k, v in summary['averages'].items():
            print(f"  {k}: {v:.3f}")
        print(f"Recommendations:")
        for rec in summary['recommendations']:
            print(f"  - {rec}")
        print()


def analyze_training_logs(log_path: Path) -> Dict[str, Any]:
    """
    Analyze training logs to understand model behavior.

    Args:
        log_path: Path to metrics.jsonl or convergence_history.json

    Returns:
        Analysis results
    """
    metrics = []

    # Load logs
    if log_path.suffix == ".jsonl":
        with open(log_path) as f:
            for line in f:
                if line.strip():
                    metrics.append(json.loads(line))
    else:
        with open(log_path) as f:
            data = json.load(f)
            metrics = data.get("metrics_history", [])

    if not metrics:
        return {"error": "No metrics found"}

    # Compute statistics
    analysis = {
        "total_steps": len(metrics),
        "reward_progression": [],
        "accuracy_progression": [],
        "format_progression": [],
        "issues_detected": [],
    }

    # Sample progression at regular intervals
    sample_points = min(20, len(metrics))
    step_size = max(1, len(metrics) // sample_points)

    for i in range(0, len(metrics), step_size):
        m = metrics[i]
        analysis["reward_progression"].append({
            "step": m.get("step", i),
            "reward": m.get("batch_total", 0),
        })
        analysis["accuracy_progression"].append({
            "step": m.get("step", i),
            "accuracy": m.get("batch_accuracy", 0),
        })
        analysis["format_progression"].append({
            "step": m.get("step", i),
            "format": m.get("batch_format", 0),
        })

    # Detect issues
    recent = metrics[-min(20, len(metrics)):]

    avg_format = sum(m.get("batch_format", 0) for m in recent) / len(recent)
    if avg_format < 0.5:
        analysis["issues_detected"].append({
            "issue": "low_format_compliance",
            "value": avg_format,
            "suggestion": "Model not learning format. Check prompt or increase format reward weight."
        })

    avg_accuracy = sum(m.get("batch_accuracy", 0) for m in recent) / len(recent)
    if avg_accuracy < 0.2:
        analysis["issues_detected"].append({
            "issue": "low_accuracy",
            "value": avg_accuracy,
            "suggestion": "Model getting wrong answers. Check data quality or reduce task difficulty."
        })

    return analysis
