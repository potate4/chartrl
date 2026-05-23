"""Full evaluation pipeline for HCPC-RLVR models."""

from typing import Dict, Any, List, Optional
from pathlib import Path
import json
from tqdm import tqdm

import torch
from datasets import Dataset

from .metrics import (
    compute_accuracy,
    extract_answer_from_output,
    compute_pass_at_k_spectrum,
)
from .diversity_metrics import compute_diversity_metrics, compute_ood_gap
from utils.parsing import parse_response
from utils.logging_utils import get_logger


class Evaluator:
    """
    Full evaluation pipeline for chart reasoning models.

    Computes:
    - Accuracy (Pass@1)
    - Pass@k for various k
    - Diversity metrics (C_table, D_reason, Coherence)
    - OOD gap (if both ID and OOD datasets provided)
    """

    def __init__(
        self,
        model,
        processor,
        device: str = "cuda",
        max_new_tokens: int = 768,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ):
        """
        Initialize evaluator.

        Args:
            model: Loaded model
            processor: Model processor
            device: Device to run on
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
        """
        self.model = model
        self.processor = processor
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.logger = get_logger("evaluator")

        # Move model to device
        if hasattr(model, "to"):
            self.model = model.to(device)
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        image,
        question: str,
        num_samples: int = 1,
    ) -> List[str]:
        """
        Generate responses for a question.

        Args:
            image: Chart image
            question: Question text
            num_samples: Number of samples to generate

        Returns:
            List of generated responses
        """
        from data.prompts import format_conversation

        # Prepare input
        conversation = format_conversation(question)

        # Apply chat template
        text = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Process inputs
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        # Generate
        outputs = []

        for _ in range(num_samples):
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
            )

            # Decode
            response = self.processor.decode(
                generated[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )
            outputs.append(response)

        return outputs

    def evaluate(
        self,
        dataset: Dataset,
        num_samples: int = 8,
        k_values: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate model on a dataset.

        Args:
            dataset: Evaluation dataset
            num_samples: Number of samples per question
            k_values: k values for Pass@k

        Returns:
            Evaluation results
        """
        if k_values is None:
            k_values = [1, 2, 4, 8]

        self.logger.info(f"Evaluating on {len(dataset)} samples with {num_samples} generations each")

        all_predictions = []
        all_labels = []
        all_correct = []  # For Pass@k
        all_diversity_metrics = []

        for example in tqdm(dataset, desc="Evaluating"):
            image = example["image"]
            question = example.get("query", example.get("question", ""))
            label = example.get("label", example.get("answer", ""))

            # Get ground truth for diversity metrics
            ground_truth = {
                "label": label,
                "chart_type": example.get("chart_type", ""),
                "table": example.get("table", {}),
            }

            # Generate responses
            responses = self.generate(image, question, num_samples)

            # Extract answers
            answers = [extract_answer_from_output(r) for r in responses]

            # Check correctness
            from .metrics import relaxed_accuracy
            correct = [relaxed_accuracy(a, label) for a in answers]

            all_predictions.append(answers[0])  # First for accuracy
            all_labels.append(label)
            all_correct.append(correct)

            # Compute diversity metrics
            div_metrics = compute_diversity_metrics(responses, ground_truth)
            all_diversity_metrics.append(div_metrics)

        # Aggregate results
        accuracy, _ = compute_accuracy(all_predictions, all_labels)

        # Pass@k
        pass_at_k = compute_pass_at_k_spectrum(all_correct, k_values)

        # Average diversity metrics
        avg_c_table = sum(m["c_table"] for m in all_diversity_metrics) / len(all_diversity_metrics)
        avg_d_reason = sum(m["d_reason"] for m in all_diversity_metrics) / len(all_diversity_metrics)
        avg_coherence = sum(m["coherence"] for m in all_diversity_metrics) / len(all_diversity_metrics)
        avg_correct_rate = sum(m["correct_rate"] for m in all_diversity_metrics) / len(all_diversity_metrics)

        results = {
            "accuracy": accuracy,
            "pass_at_k": pass_at_k,
            "c_table": avg_c_table,
            "d_reason": avg_d_reason,
            "coherence": avg_coherence,
            "correct_rate": avg_correct_rate,
            "num_samples": len(dataset),
            "num_generations": num_samples,
        }

        self.logger.info(f"Accuracy: {accuracy:.2%}")
        self.logger.info(f"Pass@k: {pass_at_k}")
        self.logger.info(f"C_table: {avg_c_table:.3f}, D_reason: {avg_d_reason:.3f}, Coherence: {avg_coherence:.3f}")

        return results

    def evaluate_id_ood(
        self,
        id_dataset: Dataset,
        ood_dataset: Dataset,
        num_samples: int = 8,
    ) -> Dict[str, Any]:
        """
        Evaluate on both ID and OOD datasets and compute gap.

        Args:
            id_dataset: In-distribution dataset
            ood_dataset: Out-of-distribution dataset
            num_samples: Samples per question

        Returns:
            Combined results with OOD gap
        """
        self.logger.info("Evaluating on in-distribution dataset...")
        id_results = self.evaluate(id_dataset, num_samples)

        self.logger.info("Evaluating on out-of-distribution dataset...")
        ood_results = self.evaluate(ood_dataset, num_samples)

        # Compute gap
        ood_gap = compute_ood_gap(id_results["accuracy"], ood_results["accuracy"])

        return {
            "id": id_results,
            "ood": ood_results,
            "ood_gap": ood_gap,
        }


def evaluate_model(
    checkpoint_path: str,
    dataset_name: str,
    base_model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    num_samples: int = 8,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function to evaluate a checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        dataset_name: Dataset to evaluate on
        base_model_name: Base model name
        num_samples: Samples per question
        output_path: Optional path to save results

    Returns:
        Evaluation results
    """
    from models import load_model_with_checkpoint
    from data import load_eval_dataset

    # Load model
    model, processor = load_model_with_checkpoint(
        checkpoint_path,
        base_model_name,
    )

    # Load dataset
    dataset = load_eval_dataset(dataset_name)

    # Evaluate
    evaluator = Evaluator(model, processor)
    results = evaluator.evaluate(dataset, num_samples)

    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    return results
