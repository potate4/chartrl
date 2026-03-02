#!/usr/bin/env python3
"""
Evaluation script for HCPC-RLVR models.

Usage:
    # Evaluate a checkpoint on EvoChart (OOD)
    python scripts/evaluate.py --checkpoint outputs/grpo_baseline/best --dataset evochart

    # Evaluate on ChartQA (ID)
    python scripts/evaluate.py --checkpoint outputs/nsr_hcpc_clc/best --dataset chartqa

    # Evaluate on both ID and OOD
    python scripts/evaluate.py --checkpoint outputs/w_reinforce_hcpc_clc/best --id-dataset chartqa --ood-dataset evochart

    # Quick evaluation with fewer samples
    python scripts/evaluate.py --checkpoint outputs/grpo_baseline/best --dataset evochart --num-samples 4 --subset 100

    # Save results to file
    python scripts/evaluate.py --checkpoint outputs/grpo_baseline/best --dataset evochart --output results.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import load_model_with_checkpoint
from data import load_eval_dataset
from evaluation import Evaluator, compute_ood_gap
from utils.logging_utils import setup_logging, get_logger
from utils.checkpointing import find_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(
        description="HCPC-RLVR Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Checkpoint
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        required=True,
        help="Path to checkpoint (can be run dir, experiment dir, or direct checkpoint)",
    )

    # Dataset options
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        help="Dataset to evaluate on (chartqa, evochart, chartqapro)",
    )
    parser.add_argument(
        "--id-dataset",
        type=str,
        help="In-distribution dataset for ID/OOD comparison",
    )
    parser.add_argument(
        "--ood-dataset",
        type=str,
        help="Out-of-distribution dataset for ID/OOD comparison",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Evaluate on subset of data",
    )

    # Generation settings
    parser.add_argument(
        "--num-samples", "-n",
        type=int,
        default=8,
        help="Number of samples per question",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature",
    )

    # Model settings
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Base model name",
    )

    # Output
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to save results JSON",
    )

    # Cache
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./cache",
        help="Cache directory",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging
    logger = setup_logging(log_level="INFO", experiment_name="evaluation")

    # Validate arguments
    if not args.dataset and not (args.id_dataset and args.ood_dataset):
        logger.error("Must specify --dataset or both --id-dataset and --ood-dataset")
        return

    # Find checkpoint
    checkpoint_path = find_checkpoint(args.checkpoint)
    if checkpoint_path is None:
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        return

    logger.info(f"Using checkpoint: {checkpoint_path}")

    # Load model
    logger.info("Loading model...")
    model, processor = load_model_with_checkpoint(
        str(checkpoint_path),
        args.base_model,
        cache_dir=args.cache_dir,
    )

    # Create evaluator
    evaluator = Evaluator(
        model=model,
        processor=processor,
        temperature=args.temperature,
    )

    results = {}

    if args.dataset:
        # Single dataset evaluation
        logger.info(f"Loading dataset: {args.dataset}")
        dataset = load_eval_dataset(
            args.dataset,
            cache_dir=args.cache_dir,
            subset_size=args.subset,
        )

        logger.info(f"Evaluating on {len(dataset)} samples...")
        results = evaluator.evaluate(dataset, num_samples=args.num_samples)

    else:
        # ID/OOD comparison
        logger.info(f"Loading ID dataset: {args.id_dataset}")
        id_dataset = load_eval_dataset(
            args.id_dataset,
            cache_dir=args.cache_dir,
            subset_size=args.subset,
        )

        logger.info(f"Loading OOD dataset: {args.ood_dataset}")
        ood_dataset = load_eval_dataset(
            args.ood_dataset,
            cache_dir=args.cache_dir,
            subset_size=args.subset,
        )

        results = evaluator.evaluate_id_ood(
            id_dataset,
            ood_dataset,
            num_samples=args.num_samples,
        )

    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    if "id" in results:
        print(f"\nIn-Distribution ({args.id_dataset}):")
        print(f"  Accuracy: {results['id']['accuracy']:.2%}")
        print(f"  C_table:  {results['id']['c_table']:.3f}")
        print(f"  D_reason: {results['id']['d_reason']:.3f}")
        print(f"  Coherence: {results['id']['coherence']:.3f}")

        print(f"\nOut-of-Distribution ({args.ood_dataset}):")
        print(f"  Accuracy: {results['ood']['accuracy']:.2%}")
        print(f"  C_table:  {results['ood']['c_table']:.3f}")
        print(f"  D_reason: {results['ood']['d_reason']:.3f}")
        print(f"  Coherence: {results['ood']['coherence']:.3f}")

        print(f"\nOOD Gap: {results['ood_gap']:.1f} percentage points")
    else:
        print(f"\nAccuracy: {results['accuracy']:.2%}")
        print(f"C_table:  {results['c_table']:.3f}")
        print(f"D_reason: {results['d_reason']:.3f}")
        print(f"Coherence: {results['coherence']:.3f}")

        if "pass_at_k" in results:
            print("\nPass@k:")
            for k, v in results["pass_at_k"].items():
                print(f"  Pass@{k}: {v:.2%}")

    print("=" * 50)

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
