"""
Pass@k Evaluation for Chart Reasoning

Computes Pass@k metrics for evaluating model diversity and robustness.

Pass@k = Probability that at least one of k samples is correct

Based on: "Evaluating Large Language Models Trained on Code" (Chen et al., 2021)
and "Decomposed Reinforcement Learning from Verifiable Feedback" (NSR paper)

Standard k values: {1, 2, 4, 8, 16, 32, 64, 128, 256}
"""

import numpy as np
import json
import logging
from typing import List, Dict, Any, Tuple
from pathlib import Path
from tqdm import tqdm


def compute_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Compute Pass@k using unbiased estimator.

    Formula: Pass@k = E[1 - (n-c choose k) / (n choose k)]

    Args:
        n: Total number of samples generated
        c: Number of correct samples among n
        k: Number of samples to consider

    Returns:
        pass_at_k: Probability that at least 1 of k samples is correct

    Examples:
        >>> compute_pass_at_k(n=100, c=10, k=1)
        0.1  # 10% of samples correct
        >>> compute_pass_at_k(n=100, c=10, k=10)
        0.65  # 65% chance at least 1 of 10 is correct
    """
    if n - c < k:
        # Not enough incorrect samples to choose k samples without including correct one
        return 1.0

    if c == 0:
        # No correct samples
        return 0.0

    # Unbiased estimator using combinatorics
    # Pass@k = 1 - P(all k samples are wrong)
    # P(all k wrong) = (n-c choose k) / (n choose k)

    def comb(n, k):
        """Compute binomial coefficient n choose k"""
        if k > n or k < 0:
            return 0
        if k == 0 or k == n:
            return 1
        # Use multiplicative formula to avoid overflow
        k = min(k, n - k)
        result = 1
        for i in range(k):
            result = result * (n - i) // (i + 1)
        return result

    try:
        # Compute: 1 - (n-c choose k) / (n choose k)
        prob_all_wrong = comb(n - c, k) / comb(n, k)
        pass_k = 1.0 - prob_all_wrong
        return pass_k
    except (ValueError, ZeroDivisionError):
        # Fallback for numerical issues
        logging.warning(f"Numerical issue in pass@k: n={n}, c={c}, k={k}")
        return float(c >= k)


def evaluate_pass_at_k_from_samples(
    samples_data: List[Dict[str, Any]],
    k_values: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256]
) -> Dict[int, float]:
    """
    Evaluate Pass@k from pre-generated samples.

    Args:
        samples_data: List of dicts with keys:
            - 'problem_id': Unique problem identifier
            - 'question': Problem question
            - 'answer': Ground truth answer
            - 'samples': List of generated samples
            - 'correct': List of booleans indicating correctness
        k_values: List of k values to evaluate

    Returns:
        results: Dict mapping k -> Pass@k score
    """
    results = {k: [] for k in k_values}

    logging.info(f"Evaluating Pass@k on {len(samples_data)} problems...")

    for problem_data in tqdm(samples_data, desc="Computing Pass@k"):
        n = len(problem_data['samples'])
        c = sum(problem_data['correct'])

        # Compute Pass@k for each k
        for k in k_values:
            if k <= n:
                pass_k = compute_pass_at_k(n=n, c=c, k=k)
                results[k].append(pass_k)

    # Average across problems
    avg_results = {}
    for k in k_values:
        if results[k]:
            avg_results[k] = np.mean(results[k])
            logging.info(f"  Pass@{k}: {avg_results[k]:.4f} ({avg_results[k]*100:.2f}%)")
        else:
            avg_results[k] = 0.0
            logging.warning(f"  Pass@{k}: No valid samples (k > n)")

    return avg_results


def load_samples_from_file(samples_path: str) -> List[Dict[str, Any]]:
    """
    Load pre-generated samples from JSON file.

    Expected format:
    [
        {
            "problem_id": 0,
            "question": "What is the average value?",
            "answer": "42.5",
            "samples": ["42.5", "43.0", ...],  # n samples
            "correct": [True, False, ...]  # correctness flags
        },
        ...
    ]

    Args:
        samples_path: Path to JSON file with samples

    Returns:
        samples_data: List of problem dicts
    """
    samples_path = Path(samples_path)
    if not samples_path.exists():
        raise FileNotFoundError(f"Samples file not found: {samples_path}")

    with open(samples_path, 'r', encoding='utf-8') as f:
        samples_data = json.load(f)

    logging.info(f"Loaded {len(samples_data)} problems from {samples_path}")

    # Validate format
    for i, problem in enumerate(samples_data):
        required_keys = ['problem_id', 'question', 'answer', 'samples', 'correct']
        for key in required_keys:
            if key not in problem:
                raise ValueError(f"Problem {i} missing required key: {key}")

        if len(problem['samples']) != len(problem['correct']):
            raise ValueError(f"Problem {i}: samples and correct lengths mismatch")

    return samples_data


def save_pass_at_k_results(
    results: Dict[int, float],
    output_path: str,
    metadata: Dict[str, Any] = None
):
    """
    Save Pass@k results to JSON file.

    Args:
        results: Dict mapping k -> Pass@k score
        output_path: Path to save results
        metadata: Optional metadata (model name, dataset, etc.)
    """
    output_data = {
        'pass_at_k': results,
        'k_values': sorted(results.keys()),
        'metadata': metadata or {}
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    logging.info(f"✓ Saved Pass@k results to {output_path}")


def compare_multiple_methods(
    results_files: List[str],
    method_names: List[str] = None
) -> Dict[str, Dict[int, float]]:
    """
    Load and compare Pass@k results from multiple methods.

    Args:
        results_files: List of paths to result JSON files
        method_names: Optional list of method names (default: use filenames)

    Returns:
        comparison: Dict mapping method_name -> {k: pass_k}
    """
    if method_names is None:
        method_names = [Path(f).stem for f in results_files]

    if len(method_names) != len(results_files):
        raise ValueError("method_names and results_files must have same length")

    comparison = {}

    for method_name, results_file in zip(method_names, results_files):
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        comparison[method_name] = data['pass_at_k']

    return comparison


def print_pass_at_k_table(results: Dict[str, Dict[int, float]]):
    """
    Print Pass@k comparison table.

    Args:
        results: Dict mapping method_name -> {k: pass_k}
    """
    # Get all k values
    k_values = sorted(next(iter(results.values())).keys())

    # Print header
    print("\n" + "=" * 80)
    print("Pass@k Comparison Table")
    print("=" * 80)

    # Print column headers
    header = "Method".ljust(20)
    for k in k_values:
        header += f"Pass@{k}".rjust(12)
    print(header)
    print("-" * 80)

    # Print results for each method
    for method_name, method_results in results.items():
        row = method_name[:20].ljust(20)
        for k in k_values:
            score = method_results.get(k, 0.0)
            row += f"{score*100:10.2f}%".rjust(12)
        print(row)

    print("=" * 80)


def compute_improvement_over_baseline(
    results: Dict[str, Dict[int, float]],
    baseline_name: str
) -> Dict[str, Dict[int, float]]:
    """
    Compute relative improvement over baseline.

    Args:
        results: Dict mapping method_name -> {k: pass_k}
        baseline_name: Name of baseline method

    Returns:
        improvements: Dict mapping method_name -> {k: improvement}
    """
    if baseline_name not in results:
        raise ValueError(f"Baseline '{baseline_name}' not found in results")

    baseline = results[baseline_name]
    improvements = {}

    for method_name, method_results in results.items():
        if method_name == baseline_name:
            improvements[method_name] = {k: 0.0 for k in baseline.keys()}
        else:
            improvements[method_name] = {
                k: (method_results[k] - baseline[k]) / baseline[k] * 100
                if baseline[k] > 0 else 0.0
                for k in baseline.keys()
            }

    return improvements


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Pass@k from generated samples")
    parser.add_argument('--samples-path', type=str, required=True,
                       help='Path to JSON file with generated samples')
    parser.add_argument('--output-path', type=str, required=True,
                       help='Path to save Pass@k results')
    parser.add_argument('--k-values', type=int, nargs='+',
                       default=[1, 2, 4, 8, 16, 32, 64, 128, 256],
                       help='K values to evaluate')
    parser.add_argument('--method-name', type=str, default='unknown',
                       help='Method name for metadata')

    args = parser.parse_args()

    # Load samples
    samples_data = load_samples_from_file(args.samples_path)

    # Compute Pass@k
    results = evaluate_pass_at_k_from_samples(samples_data, args.k_values)

    # Save results
    metadata = {
        'method': args.method_name,
        'num_problems': len(samples_data),
        'samples_per_problem': len(samples_data[0]['samples']) if samples_data else 0
    }
    save_pass_at_k_results(results, args.output_path, metadata)

    print("\n✓ Pass@k evaluation complete!")
