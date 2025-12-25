"""
Generate samples for Pass@k evaluation

Generates n=256 samples per problem for computing Pass@k metrics.

Usage:
    python generate_samples.py \
        --checkpoint-path grpo-start-ckpts/qwen2-5-3b-grpo-final \
        --dataset-name evochart \
        --output-path samples_grpo.json \
        --num-samples 256
"""

import torch
import argparse
import json
import logging
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any

# Import from existing modules
from models import load_vlm
from dataset_process import load_benchmark
from prompts import format_prompt
from grpo_utils import accuracy_reward, extract_text_from_completion


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def generate_samples_for_problem(
    model,
    processor,
    problem: Dict[str, Any],
    num_samples: int = 256,
    temperature: float = 1.0,
    batch_size: int = 8
) -> List[str]:
    """
    Generate multiple samples for a single problem.

    Args:
        model: Vision-language model
        processor: Model processor
        problem: Problem dict with 'question' and 'image'
        num_samples: Number of samples to generate
        temperature: Sampling temperature (1.0 for diversity)
        batch_size: Batch size for generation

    Returns:
        samples: List of generated text responses
    """
    samples = []

    # Generate in batches to avoid OOM
    num_batches = (num_samples + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        batch_num_samples = min(batch_size, num_samples - len(samples))

        # Format prompt
        messages = format_prompt(problem['question'])

        # Prepare inputs
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = processor(
            text=[text] * batch_num_samples,
            images=[problem['image']] * batch_num_samples,
            return_tensors="pt",
            padding=True
        ).to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=768,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                num_return_sequences=1  # One per input
            )

        # Decode
        batch_samples = processor.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        samples.extend(batch_samples)

    return samples[:num_samples]


def check_sample_correctness(
    sample: str,
    ground_truth: str,
    problem_data: Dict[str, Any] = None
) -> bool:
    """
    Check if a generated sample is correct.

    Args:
        sample: Generated text
        ground_truth: Ground truth answer
        problem_data: Optional problem metadata

    Returns:
        is_correct: Whether sample is correct
    """
    # Use existing accuracy_reward function
    # Format as single-item lists for compatibility
    rewards = accuracy_reward(
        completions=[sample],
        label=[ground_truth]
    )

    # Reward of 1.0 means correct
    return rewards[0] >= 0.9


def generate_samples_for_dataset(
    checkpoint_path: str,
    dataset_name: str,
    num_samples: int = 256,
    temperature: float = 1.0,
    batch_size: int = 8,
    max_problems: int = None
) -> List[Dict[str, Any]]:
    """
    Generate samples for entire dataset.

    Args:
        checkpoint_path: Path to trained model checkpoint
        dataset_name: Name of dataset to evaluate
        num_samples: Number of samples per problem
        temperature: Sampling temperature
        batch_size: Batch size for generation
        max_problems: Maximum number of problems to evaluate (None = all)

    Returns:
        all_samples: List of dicts with samples and correctness
    """
    logging.info("=" * 80)
    logging.info("SAMPLE GENERATION FOR PASS@K EVALUATION")
    logging.info(f"  Checkpoint: {checkpoint_path}")
    logging.info(f"  Dataset: {dataset_name}")
    logging.info(f"  Samples per problem: {num_samples}")
    logging.info(f"  Temperature: {temperature}")
    logging.info("=" * 80)

    # Load model
    logging.info("Loading model...")
    model, processor = load_vlm(checkpoint_path)
    model.eval()

    # Load dataset
    logging.info(f"Loading {dataset_name} dataset...")
    dataset = load_benchmark(dataset_name)

    if max_problems:
        dataset = dataset.select(range(min(max_problems, len(dataset))))

    logging.info(f"Generating samples for {len(dataset)} problems...")

    all_samples = []

    for idx, problem in enumerate(tqdm(dataset, desc="Generating samples")):
        # Generate samples
        samples = generate_samples_for_problem(
            model=model,
            processor=processor,
            problem=problem,
            num_samples=num_samples,
            temperature=temperature,
            batch_size=batch_size
        )

        # Check correctness
        correct_flags = []
        for sample in samples:
            # Extract text from sample
            sample_text = extract_text_from_completion(sample)
            is_correct = check_sample_correctness(
                sample=sample_text,
                ground_truth=problem.get('label', problem.get('answer', '')),
                problem_data=problem
            )
            correct_flags.append(is_correct)

        # Store results
        problem_samples = {
            'problem_id': idx,
            'question': problem.get('question', ''),
            'answer': problem.get('label', problem.get('answer', '')),
            'samples': samples,
            'correct': correct_flags,
            'num_correct': sum(correct_flags),
            'num_total': len(samples)
        }

        all_samples.append(problem_samples)

        # Log progress
        if (idx + 1) % 10 == 0:
            avg_correct = sum(s['num_correct'] for s in all_samples) / sum(s['num_total'] for s in all_samples)
            logging.info(f"  [{idx+1}/{len(dataset)}] Avg correctness: {avg_correct:.2%}")

    # Final statistics
    total_samples = sum(s['num_total'] for s in all_samples)
    total_correct = sum(s['num_correct'] for s in all_samples)
    avg_correctness = total_correct / total_samples if total_samples > 0 else 0

    logging.info("=" * 80)
    logging.info("GENERATION COMPLETE")
    logging.info(f"  Total problems: {len(all_samples)}")
    logging.info(f"  Total samples: {total_samples}")
    logging.info(f"  Correct samples: {total_correct}")
    logging.info(f"  Average correctness: {avg_correctness:.2%}")
    logging.info("=" * 80)

    return all_samples


def save_samples(samples: List[Dict[str, Any]], output_path: str):
    """Save generated samples to JSON file"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to JSON-serializable format
    # (remove non-serializable objects like images)
    serializable_samples = []
    for sample in samples:
        serializable_sample = {
            'problem_id': sample['problem_id'],
            'question': sample['question'],
            'answer': sample['answer'],
            'samples': sample['samples'],
            'correct': sample['correct'],
            'num_correct': sample['num_correct'],
            'num_total': sample['num_total']
        }
        serializable_samples.append(serializable_sample)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_samples, f, indent=2)

    logging.info(f"✓ Saved {len(samples)} problem samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate samples for Pass@k evaluation")

    # Model and data
    parser.add_argument('--checkpoint-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--dataset-name', type=str, required=True,
                       help='Dataset name (chartqa, plotqa, evochart, etc.)')

    # Generation settings
    parser.add_argument('--num-samples', type=int, default=256,
                       help='Number of samples to generate per problem (default: 256)')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature (default: 1.0)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for generation (default: 8)')

    # Output
    parser.add_argument('--output-path', type=str, required=True,
                       help='Path to save generated samples (JSON)')

    # Optional
    parser.add_argument('--max-problems', type=int, default=None,
                       help='Maximum number of problems to evaluate (default: all)')

    args = parser.parse_args()

    # Setup
    setup_logging()

    # Generate samples
    samples = generate_samples_for_dataset(
        checkpoint_path=args.checkpoint_path,
        dataset_name=args.dataset_name,
        num_samples=args.num_samples,
        temperature=args.temperature,
        batch_size=args.batch_size,
        max_problems=args.max_problems
    )

    # Save
    save_samples(samples, args.output_path)

    print("\n✓ Sample generation complete!")
    print(f"  Samples saved to: {args.output_path}")
    print(f"\nNext steps:")
    print(f"  1. Evaluate Pass@k:")
    print(f"     python evaluation/pass_at_k.py --samples-path {args.output_path} --output-path results.json")
    print(f"  2. Plot curves:")
    print(f"     python evaluation/plotting.py --results results.json --output pass_at_k.png")


if __name__ == "__main__":
    main()
