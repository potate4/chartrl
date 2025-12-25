"""
Pass@k Curve Plotting for Chart Reasoning

Creates publication-quality plots comparing Pass@k across different methods.

Based on: "Decomposed Reinforcement Learning from Verifiable Feedback" (NSR paper)
Reproduces Figure 2 and Figure 3 style plots.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging


# Set publication-quality defaults
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['figure.titlesize'] = 13


# Color scheme for different methods (similar to paper)
METHOD_COLORS = {
    'SFT': '#7f7f7f',  # Gray
    'PPO': '#ff7f0e',  # Orange
    'GRPO': '#2ca02c',  # Green
    'PSR': '#d62728',  # Red
    'NSR': '#1f77b4',  # Blue
    'W-REINFORCE': '#9467bd',  # Purple
    'Weighted-REINFORCE': '#9467bd',  # Purple (alias)
}

# Line styles
METHOD_LINESTYLES = {
    'SFT': '--',
    'PPO': '-.',
    'GRPO': '-',
    'PSR': ':',
    'NSR': '-',
    'W-REINFORCE': '-',
    'Weighted-REINFORCE': '-',
}

# Markers
METHOD_MARKERS = {
    'SFT': 's',  # Square
    'PPO': '^',  # Triangle up
    'GRPO': 'o',  # Circle
    'PSR': 'v',  # Triangle down
    'NSR': 'D',  # Diamond
    'W-REINFORCE': '*',  # Star
    'Weighted-REINFORCE': '*',  # Star (alias)
}


def plot_pass_at_k_curve(
    results: Dict[str, Dict[int, float]],
    save_path: str,
    title: str = "Pass@k Comparison",
    ylabel: str = "Pass@k",
    figsize: Tuple[int, int] = (10, 6),
    show_grid: bool = True,
    log_scale: bool = True,
    ylim: Optional[Tuple[float, float]] = None,
    show_legend: bool = True,
    legend_loc: str = 'lower right'
):
    """
    Plot Pass@k curves for multiple methods.

    Args:
        results: Dict mapping method_name -> {k: pass_k}
        save_path: Path to save plot
        title: Plot title
        ylabel: Y-axis label
        figsize: Figure size (width, height)
        show_grid: Whether to show grid
        log_scale: Whether to use log scale for x-axis
        ylim: Y-axis limits (min, max)
        show_legend: Whether to show legend
        legend_loc: Legend location
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get k values (assume all methods have same k values)
    k_values = sorted(next(iter(results.values())).keys())

    # Plot each method
    for method_name, method_results in results.items():
        pass_values = [method_results[k] * 100 for k in k_values]  # Convert to percentage

        # Get style for this method
        color = METHOD_COLORS.get(method_name, None)
        linestyle = METHOD_LINESTYLES.get(method_name, '-')
        marker = METHOD_MARKERS.get(method_name, 'o')

        ax.plot(
            k_values,
            pass_values,
            marker=marker,
            markersize=6,
            linestyle=linestyle,
            linewidth=2,
            color=color,
            label=method_name,
            alpha=0.9
        )

    # Set x-axis to log scale if requested
    if log_scale:
        ax.set_xscale('log', base=2)
        ax.set_xticks(k_values)
        ax.set_xticklabels([str(k) for k in k_values])

    # Labels and title
    ax.set_xlabel('k (number of samples)', fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=15)

    # Y-axis limits
    if ylim:
        ax.set_ylim(ylim)
    else:
        # Auto ylim with some padding
        all_values = [v for method_results in results.values() for v in method_results.values()]
        ymin = max(0, min(all_values) * 100 - 5)
        ymax = min(100, max(all_values) * 100 + 5)
        ax.set_ylim(ymin, ymax)

    # Grid
    if show_grid:
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Legend
    if show_legend:
        ax.legend(loc=legend_loc, framealpha=0.9, edgecolor='black')

    # Tight layout
    plt.tight_layout()

    # Save
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"✓ Saved Pass@k curve to {save_path}")


def plot_pass_at_k_comparison_grid(
    results_by_dataset: Dict[str, Dict[str, Dict[int, float]]],
    save_path: str,
    title: str = "Pass@k Comparison Across Datasets",
    figsize: Tuple[int, int] = (15, 10),
    ncols: int = 2
):
    """
    Plot Pass@k curves for multiple datasets in a grid.

    Args:
        results_by_dataset: Dict mapping dataset_name -> method_name -> {k: pass_k}
        save_path: Path to save plot
        title: Overall plot title
        figsize: Figure size
        ncols: Number of columns in grid
    """
    n_datasets = len(results_by_dataset)
    nrows = (n_datasets + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_datasets == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (dataset_name, results) in enumerate(results_by_dataset.items()):
        ax = axes[idx]

        # Get k values
        k_values = sorted(next(iter(results.values())).keys())

        # Plot each method
        for method_name, method_results in results.items():
            pass_values = [method_results[k] * 100 for k in k_values]

            color = METHOD_COLORS.get(method_name, None)
            linestyle = METHOD_LINESTYLES.get(method_name, '-')
            marker = METHOD_MARKERS.get(method_name, 'o')

            ax.plot(
                k_values,
                pass_values,
                marker=marker,
                markersize=5,
                linestyle=linestyle,
                linewidth=1.5,
                color=color,
                label=method_name,
                alpha=0.9
            )

        # Formatting
        ax.set_xscale('log', base=2)
        ax.set_xticks(k_values)
        ax.set_xticklabels([str(k) for k in k_values], rotation=45)
        ax.set_xlabel('k', fontweight='bold')
        ax.set_ylabel('Pass@k (%)', fontweight='bold')
        ax.set_title(dataset_name, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        if idx == 0:  # Legend only on first subplot
            ax.legend(loc='lower right', framealpha=0.9, edgecolor='black', fontsize=8)

    # Hide unused subplots
    for idx in range(n_datasets, len(axes)):
        axes[idx].axis('off')

    # Overall title
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"✓ Saved grid comparison to {save_path}")


def plot_improvement_over_baseline(
    improvements: Dict[str, Dict[int, float]],
    save_path: str,
    baseline_name: str,
    title: str = "Improvement over Baseline (%)",
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot relative improvement over baseline method.

    Args:
        improvements: Dict mapping method_name -> {k: improvement_percentage}
        save_path: Path to save plot
        baseline_name: Name of baseline method
        title: Plot title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get k values
    k_values = sorted(next(iter(improvements.values())).keys())

    # Plot each method's improvement
    for method_name, method_improvements in improvements.items():
        if method_name == baseline_name:
            continue  # Skip baseline itself

        improvement_values = [method_improvements[k] for k in k_values]

        color = METHOD_COLORS.get(method_name, None)
        linestyle = METHOD_LINESTYLES.get(method_name, '-')
        marker = METHOD_MARKERS.get(method_name, 'o')

        ax.plot(
            k_values,
            improvement_values,
            marker=marker,
            markersize=6,
            linestyle=linestyle,
            linewidth=2,
            color=color,
            label=method_name,
            alpha=0.9
        )

    # Horizontal line at 0 (baseline)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5, label=f'{baseline_name} (baseline)')

    # Formatting
    ax.set_xscale('log', base=2)
    ax.set_xticks(k_values)
    ax.set_xticklabels([str(k) for k in k_values])
    ax.set_xlabel('k (number of samples)', fontweight='bold')
    ax.set_ylabel('Relative Improvement (%)', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(loc='best', framealpha=0.9, edgecolor='black')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"✓ Saved improvement plot to {save_path}")


def create_pass_at_k_bar_chart(
    results: Dict[str, Dict[int, float]],
    k_value: int,
    save_path: str,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 5)
):
    """
    Create bar chart comparing methods at a specific k value.

    Args:
        results: Dict mapping method_name -> {k: pass_k}
        k_value: Which k value to plot
        save_path: Path to save plot
        title: Plot title (default: f"Pass@{k_value} Comparison")
        figsize: Figure size
    """
    if title is None:
        title = f"Pass@{k_value} Comparison"

    fig, ax = plt.subplots(figsize=figsize)

    # Extract Pass@k for specific k
    methods = list(results.keys())
    pass_values = [results[method][k_value] * 100 for method in methods]

    # Colors
    colors = [METHOD_COLORS.get(method, '#1f77b4') for method in methods]

    # Bar chart
    bars = ax.bar(methods, pass_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{height:.1f}%',
            ha='center',
            va='bottom',
            fontweight='bold'
        )

    # Formatting
    ax.set_ylabel('Pass@k (%)', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_ylim(0, max(pass_values) * 1.15)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"✓ Saved bar chart to {save_path}")


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot Pass@k comparison curves")
    parser.add_argument('--results', type=str, nargs='+', required=True,
                       help='Paths to Pass@k result JSON files')
    parser.add_argument('--labels', type=str, nargs='+',
                       help='Method names (default: use filenames)')
    parser.add_argument('--output', type=str, default='pass_at_k_comparison.png',
                       help='Output plot path')
    parser.add_argument('--title', type=str, default='Pass@k Comparison',
                       help='Plot title')

    args = parser.parse_args()

    # Load results
    results = {}
    labels = args.labels if args.labels else [Path(f).stem for f in args.results]

    for label, result_file in zip(labels, args.results):
        with open(result_file, 'r') as f:
            data = json.load(f)
            results[label] = data['pass_at_k']

    # Plot
    plot_pass_at_k_curve(
        results=results,
        save_path=args.output,
        title=args.title
    )

    print(f"\n✓ Plot saved to {args.output}")
