#!/usr/bin/env python3
"""
Compare Base Qwen2.5-VL-3B vs Chart-RVR GRPO-trained model.

Measures:
  - Accuracy (relaxed, 5% tolerance)
  - D_reason (reasoning diversity across rollouts)
  - C_table (table extraction consistency)

Thesis claim: GRPO increases accuracy but causes reasoning collapse (low D_reason).

Usage (Colab / Kaggle / RunPod):
    python scripts/compare_base_vs_grpo.py --num-samples 200 --num-rollouts 4
    python scripts/compare_base_vs_grpo.py --num-samples 50 --num-rollouts 4 --quick
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import torch
from tqdm import tqdm
from PIL import Image


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
GRPO_MODEL = "sanchit97/chart-rvr-3b"

SYSTEM_PROMPT = r"""
        You are a vision-language assistant. You are given a chart image and a query about the chart.
        Think step-by-step about how to answer the query based on the chart image and then provide the final answer.

        ### Output format
        Respond **with exactly two blocks in order and nothing else**:
        <think>
        First output the type of chart in <type>, \
        then output the underlying data table and finally, \
        think step-by-step about how to answer the query based on the chart image \
        and then provide the final answer.
        <type>
        Type of chart - one word from line, bar, stacked bar, pie, histogram, scatterplot, area, stacked area, bubble, treemap.
        </type>
        Next output the data table in the <table></table> tags
        <table>
        json table - for the chart image, output only a JSON object with: "columns": list of column headers, "rows": list-of-lists, one per data row
        No prose, no comments.
        1. Respond with **only** a JSON object
        2. The JSON must use exactly this schema:
            {
                "columns": [...],
                "rows": [[...], [...],..., [...]]
            }
        3. Do NOT output HTML, Markdown, or commentary. Any deviation gets zero reward.
        </table>
        Provide your reasoning here in steps:
        <step-1>: Provide a description of reasoning
        <step-2>: Gather ALL the appropriate data from the chart
        <step-3>: Break down the query into smaller parts and verify each part with the data
        ...
        <step-n>: Do the final calculation or reasoning to derive the answer
        </think>
        <answer>
        Final answer on a single line
        </answer>
        """


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Base vs GRPO comparison")
    parser.add_argument("--num-samples", type=int, default=200,
                        help="Number of eval samples per dataset")
    parser.add_argument("--num-rollouts", type=int, default=4,
                        help="Rollouts per sample (for D_reason)")
    parser.add_argument("--datasets", nargs="+",
                        default=["chartqa", "evochart"],
                        help="Eval datasets")
    parser.add_argument("--output-dir", type=str, default="./eval_results",
                        help="Directory to save results")
    parser.add_argument("--cache-dir", type=str, default="./cache",
                        help="HF cache directory")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 50 samples, 2 rollouts")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--skip-base", action="store_true",
                        help="Skip base model eval (if already done)")
    parser.add_argument("--skip-grpo", action="store_true",
                        help="Skip GRPO model eval (if already done)")
    return parser.parse_args()


def load_model(model_name: str, cache_dir: str):
    """Load model and processor. Uses 4-bit quantization to fit on T4 16GB."""
    from transformers import AutoProcessor, BitsAndBytesConfig
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
    except ImportError:
        try:
            from transformers import Qwen2VLForConditionalGeneration as ModelClass
        except ImportError:
            from transformers import AutoModelForVision2Seq as ModelClass

    print(f"\n{'='*60}")
    print(f"Loading: {model_name}")
    print(f"{'='*60}")

    processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)

    # 4-bit quantization: ~4GB VRAM instead of ~7GB, no CPU offloading
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    model = ModelClass.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,
        cache_dir=cache_dir,
    )
    model.eval()

    mem_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"Loaded (4-bit). GPU memory: {mem_gb:.1f} GB")
    return model, processor


def free_model(model):
    """Free GPU memory."""
    del model
    torch.cuda.empty_cache()
    import gc
    gc.collect()


def load_eval_data(dataset_name: str, num_samples: int, cache_dir: str):
    """Load evaluation dataset."""
    from datasets import load_dataset, Dataset

    print(f"Loading dataset: {dataset_name} ({num_samples} samples)...")

    if dataset_name == "chartqa":
        ds = load_dataset("HuggingFaceM4/ChartQA", split="test", cache_dir=cache_dir)
        # Normalize column names
        query_col = "query" if "query" in ds.column_names else "question"
        label_col = "label" if "label" in ds.column_names else "answer"
    elif dataset_name == "evochart":
        ds = load_dataset("MuyeHuang/EvoChart-QA-Benchmark", cache_dir=cache_dir)
        ds = ds["train"]  # EvoChart only has train split
        query_col = "query" if "query" in ds.column_names else "question"
        label_col = "label" if "label" in ds.column_names else "answer"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Shuffle and take subset
    ds = ds.shuffle(seed=42).select(range(min(num_samples, len(ds))))

    print(f"  Loaded {len(ds)} samples. Columns: {ds.column_names}")
    print(f"  Query column: {query_col}, Label column: {label_col}")
    return ds, query_col, label_col


@torch.no_grad()
def generate_rollouts(
    model, processor, image: Image.Image, question: str,
    num_rollouts: int, max_new_tokens: int, temperature: float,
) -> List[str]:
    """Generate multiple rollouts for one sample."""
    from qwen_vl_utils import process_vision_info

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ]},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    rollouts = []
    for _ in range(num_rollouts):
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.95,
            do_sample=True,
        )
        trimmed = generated_ids[0][inputs.input_ids.shape[1]:]
        output = processor.decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        rollouts.append(output)

    return rollouts


def extract_answer(text: str) -> str:
    """Extract answer from <answer>...</answer> tags."""
    import re
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def extract_reasoning(text: str) -> str:
    """Extract reasoning from between </table> and </think>."""
    import re
    match = re.search(r"</table>(.*?)</think>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: everything in <think> minus <type> and <table>
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if match:
        content = match.group(1)
        content = re.sub(r"<type>.*?</type>", "", content, flags=re.DOTALL)
        content = re.sub(r"<table>.*?</table>", "", content, flags=re.DOTALL)
        return content.strip()
    return ""


def relaxed_accuracy(pred: str, label: str, tolerance: float = 0.05) -> bool:
    """Check if prediction matches label (numeric tolerance or exact string)."""
    import re
    # Numeric cleaning for both
    def clean(s):
        s = s.strip().strip("'\"")
        s = re.sub(r"[$€£¥₹]", "", s)
        s = re.sub(r"(\d),(\d)", r"\1\2", s)
        if s.endswith("%"):
            s = s[:-1].strip()
        mults = {"million": 1e6, "billion": 1e9, "trillion": 1e12}
        for word, mult in mults.items():
            if s.lower().endswith(word):
                num_part = s[:len(s)-len(word)].strip()
                try:
                    return [float(num_part), float(num_part) * mult]
                except ValueError:
                    pass
        try:
            return [float(s)]
        except ValueError:
            return None

    pred_nums = clean(pred)
    label_nums = clean(label)

    if pred_nums is not None and label_nums is not None:
        for pv in pred_nums:
            for lv in label_nums:
                denom = abs(lv) + 1e-9
                if abs(pv - lv) / denom <= tolerance:
                    return True
        return False

    # String comparison
    return pred.strip().lower() == label.strip().lower()


_SBERT_MODEL = None


def _get_sbert():
    """Load SentenceTransformer once, reuse across all calls."""
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _SBERT_MODEL


def compute_d_reason(reasonings: List[str]) -> float:
    """
    Compute reasoning diversity = 1 - avg_pairwise_similarity.

    Uses sentence-transformers for semantic similarity.
    """
    valid = [r for r in reasonings if r and r.strip()]
    if len(valid) < 2:
        return 0.0

    import numpy as np

    model = _get_sbert()
    embeddings = model.encode(valid, convert_to_numpy=True)

    # Pairwise cosine similarity (upper triangle)
    n = len(embeddings)
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            cos = np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-8
            )
            sims.append((cos + 1) / 2)  # Normalize to [0, 1]

    avg_sim = np.mean(sims)
    return float(1.0 - avg_sim)  # Diversity = 1 - similarity


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------
def evaluate_model_on_dataset(
    model, processor, dataset, query_col: str, label_col: str,
    num_rollouts: int, max_new_tokens: int, temperature: float,
    model_tag: str,
) -> Dict[str, Any]:
    """Run full evaluation for one model on one dataset."""

    results = []
    correct_count = 0
    d_reasons = []

    for i, example in enumerate(tqdm(dataset, desc=f"[{model_tag}]")):
        image = example["image"]
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        question = str(example.get(query_col, ""))
        label = str(example.get(label_col, ""))

        try:
            rollouts = generate_rollouts(
                model, processor, image, question,
                num_rollouts, max_new_tokens, temperature,
            )
        except Exception as e:
            print(f"\n  Error on sample {i}: {e}")
            rollouts = [""] * num_rollouts

        # Extract answers and reasoning from each rollout
        answers = [extract_answer(r) for r in rollouts]
        reasonings = [extract_reasoning(r) for r in rollouts]

        # Accuracy: use first rollout (greedy-like)
        is_correct = relaxed_accuracy(answers[0], label) if answers[0] else False
        correct_count += int(is_correct)

        # D_reason across all rollouts
        d_reason = compute_d_reason(reasonings)
        d_reasons.append(d_reason)

        results.append({
            "idx": i,
            "question": question,
            "label": label,
            "answers": answers,
            "correct": is_correct,
            "d_reason": d_reason,
            "first_rollout": rollouts[0][:500],  # truncate for storage
        })

        # Progress
        if (i + 1) % 10 == 0:
            acc_so_far = correct_count / (i + 1)
            d_so_far = sum(d_reasons) / len(d_reasons)
            print(f"\n  [{model_tag}] {i+1}/{len(dataset)} | "
                  f"Acc: {acc_so_far:.3f} | D_reason: {d_so_far:.3f}")

    # Aggregate
    n = len(dataset)
    accuracy = correct_count / n if n > 0 else 0.0
    avg_d_reason = sum(d_reasons) / len(d_reasons) if d_reasons else 0.0

    return {
        "model": model_tag,
        "accuracy": accuracy,
        "d_reason": avg_d_reason,
        "num_samples": n,
        "num_rollouts": num_rollouts,
        "num_correct": correct_count,
        "per_sample": results,
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    if args.quick:
        args.num_samples = 50
        args.num_rollouts = 2
        print("Quick mode: 50 samples, 2 rollouts")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {}

    for ds_name in args.datasets:
        print(f"\n{'#'*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'#'*60}")

        dataset, query_col, label_col = load_eval_data(
            ds_name, args.num_samples, args.cache_dir
        )

        # --- Base model ---
        if not args.skip_base:
            model, processor = load_model(BASE_MODEL, args.cache_dir)
            base_results = evaluate_model_on_dataset(
                model, processor, dataset, query_col, label_col,
                args.num_rollouts, args.max_new_tokens, args.temperature,
                model_tag=f"base_{ds_name}",
            )
            all_results[f"base_{ds_name}"] = base_results
            free_model(model)

            # Save intermediate
            with open(output_dir / f"base_{ds_name}_{timestamp}.json", "w") as f:
                json.dump(base_results, f, indent=2, default=str)

        # --- GRPO model ---
        if not args.skip_grpo:
            model, processor = load_model(GRPO_MODEL, args.cache_dir)
            grpo_results = evaluate_model_on_dataset(
                model, processor, dataset, query_col, label_col,
                args.num_rollouts, args.max_new_tokens, args.temperature,
                model_tag=f"grpo_{ds_name}",
            )
            all_results[f"grpo_{ds_name}"] = grpo_results
            free_model(model)

            with open(output_dir / f"grpo_{ds_name}_{timestamp}.json", "w") as f:
                json.dump(grpo_results, f, indent=2, default=str)

    # --- Summary ---
    print("\n")
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Model':<25} {'Dataset':<15} {'Accuracy':>10} {'D_reason':>10}")
    print("-" * 70)

    for key, res in all_results.items():
        parts = key.split("_", 1)
        model_tag = parts[0].upper()
        ds_tag = parts[1] if len(parts) > 1 else key
        print(f"{model_tag:<25} {ds_tag:<15} {res['accuracy']:>10.3f} {res['d_reason']:>10.3f}")

    # Show deltas
    for ds_name in args.datasets:
        base_key = f"base_{ds_name}"
        grpo_key = f"grpo_{ds_name}"
        if base_key in all_results and grpo_key in all_results:
            acc_delta = all_results[grpo_key]["accuracy"] - all_results[base_key]["accuracy"]
            d_delta = all_results[grpo_key]["d_reason"] - all_results[base_key]["d_reason"]
            print(f"\n  [{ds_name}] GRPO vs Base:")
            print(f"    Accuracy: {acc_delta:+.3f}")
            print(f"    D_reason: {d_delta:+.3f}")
            if acc_delta > 0 and d_delta < 0:
                print(f"    --> Confirms reasoning collapse: accuracy UP, diversity DOWN")

    # Save combined
    summary_path = output_dir / f"comparison_{timestamp}.json"
    with open(summary_path, "w") as f:
        summary = {k: {kk: vv for kk, vv in v.items() if kk != "per_sample"}
                   for k, v in all_results.items()}
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_dir}/")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
