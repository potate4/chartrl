#!/usr/bin/env python3
"""
Modular evaluation script for HCPC-RLVR.

Runs ONE model on ONE dataset per invocation. Run multiple times with
different args to build up your evaluation matrix.

Usage:
    # Base Qwen model on ChartQA (200 samples)
    python scripts/eval_run.py --dataset chartqa --subset 200

    # Base Qwen on EvoChart
    python scripts/eval_run.py --dataset evochart --subset 200

    # LoRA checkpoint on ChartQA
    python scripts/eval_run.py --dataset chartqa --subset 200 \
      --checkpoint outputs/grpo_baseline/.../checkpoint-70

    # Custom generation settings
    python scripts/eval_run.py --dataset chartqa --subset 100 \
      --checkpoint path/to/ckpt --num-samples 4 --temperature 0.7

    # Resume a crashed run (picks up where it left off)
    python scripts/eval_run.py --resume outputs/eval_results/base_chartqa_20260219_143201
"""

import argparse
import json
import sys
import time
import logging
from pathlib import Path
from datetime import datetime

# Add app/ to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from datasets import Dataset

from data.dataset import load_eval_dataset
from data.prompts import format_conversation
from evaluation.metrics import (
    relaxed_accuracy,
    extract_answer_from_output,
    compute_pass_at_k_spectrum,
)
from evaluation.diversity_metrics import compute_diversity_metrics
from utils.parsing import parse_response, check_format_compliance
from rewards.base_rewards import accuracy_reward as type_aware_accuracy


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate one model on one dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model
    p.add_argument("--checkpoint", "-c", type=str, default=None,
                   help="Path to LoRA checkpoint dir. Omit for base model.")
    p.add_argument("--base-model", type=str,
                   default="Qwen/Qwen2.5-VL-3B-Instruct",
                   help="Base model name (default: Qwen2.5-VL-3B)")
    # Dataset
    p.add_argument("--dataset", "-d", type=str, default=None,
                   help="Dataset: chartqa, evochart, chartqapro, or HF path")
    p.add_argument("--split", type=str, default="test",
                   help="Dataset split (default: test)")
    p.add_argument("--subset", type=int, default=None,
                   help="Evaluate on first N samples")

    # Generation
    p.add_argument("--num-samples", "-n", type=int, default=8,
                   help="Rollouts per question (for Pass@k / diversity)")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=768)

    # Output
    p.add_argument("--output-dir", type=str, default="outputs/eval_results",
                   help="Root dir for result folders")
    p.add_argument("--run-name", type=str, default=None,
                   help="Run label (auto-generated if omitted)")

    # Resume
    p.add_argument("--resume", type=str, default=None,
                   help="Path to existing run dir to resume (reads per_sample.jsonl)")

    # Misc
    p.add_argument("--cache-dir", type=str, default="./cache")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(args):
    """Load base or checkpoint model in bfloat16."""
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel

    processor = AutoProcessor.from_pretrained(
        args.base_model, trust_remote_code=True, cache_dir=args.cache_dir,
    )

    print(f"Loading base model: {args.base_model}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
        cache_dir=args.cache_dir,
    )

    if args.checkpoint:
        print(f"Loading LoRA adapter: {args.checkpoint}")
        model = PeftModel.from_pretrained(model, args.checkpoint, is_trainable=False)

    model.eval()
    return model, processor


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_responses(model, processor, image, question, args):
    """Generate num_samples responses for a single question (batched)."""
    conversation = format_conversation(question)
    text = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True,
    )

    # Process image ONCE (expensive CPU op), then replicate text tensors for batch
    single_inputs = processor(
        text=[text], images=[image], return_tensors="pt", padding=True,
    )

    # Vision inputs (pixel_values, image_grid_thw) are NOT batch-dim-first in
    # Qwen2.5-VL — they represent patches/images, so repeat them N times.
    # Text inputs (input_ids, attention_mask) ARE batch-first, so expand dim 0.
    vision_keys = {"pixel_values", "image_grid_thw", "pixel_values_videos",
                   "video_grid_thw"}

    device = next(model.parameters()).device
    n = args.num_samples
    inputs = {}
    for k, v in single_inputs.items():
        if not hasattr(v, "to"):
            inputs[k] = v
            continue
        v = v.to(device)
        if k in vision_keys:
            # Repeat along dim 0: [P, ...] -> [P*n, ...]
            inputs[k] = v.repeat(n, *([1] * (v.dim() - 1)))
        elif v.shape[0] == 1:
            # Expand batch dim: [1, ...] -> [n, ...]
            inputs[k] = v.expand(n, *v.shape[1:]).contiguous()
        else:
            inputs[k] = v

    prompt_len = inputs["input_ids"].shape[1]
    generated = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=True,
    )

    outputs = []
    for i in range(generated.shape[0]):
        response = processor.decode(
            generated[i][prompt_len:],
            skip_special_tokens=True,
        )
        outputs.append(response)

    return outputs


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def _load_resumed_state(run_dir):
    """Load previous results from a crashed run's per_sample.jsonl."""
    per_sample_path = Path(run_dir) / "per_sample.jsonl"
    if not per_sample_path.exists():
        return 0, [], []

    records = []
    with open(per_sample_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return len(records), records, [r["idx"] for r in records]


def run_eval(args):
    # --- Resume mode: load config from previous run ---
    resuming = False
    skip_indices = set()
    prev_records = []

    if args.resume:
        resume_dir = Path(args.resume)
        if not resume_dir.exists():
            print(f"Resume dir not found: {resume_dir}")
            sys.exit(1)

        # Load previous config — always use previous run's settings on resume
        prev_config_path = resume_dir / "config.json"
        if not prev_config_path.exists():
            print(f"No config.json found in {resume_dir}")
            sys.exit(1)
        with open(prev_config_path) as f:
            prev_config = json.load(f)
        for key in ["dataset", "checkpoint", "base_model", "split", "subset",
                    "num_samples", "temperature", "top_p", "max_new_tokens",
                    "cache_dir", "seed"]:
            if key in prev_config:
                setattr(args, key.replace("-", "_"), prev_config[key])

        num_done, prev_records, done_indices = _load_resumed_state(resume_dir)
        skip_indices = set(done_indices)
        resuming = True
        run_dir = resume_dir
        run_name = resume_dir.name
        stamp = prev_config.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_tag = Path(args.checkpoint).name if args.checkpoint else "base"
        run_name = args.run_name or f"{model_tag}_{args.dataset}_{stamp}"
        run_dir = Path(args.output_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

    model_tag = Path(args.checkpoint).name if args.checkpoint else "base"

    # Set up logging to both console and file
    log_path = run_dir / "eval.log"
    logger = logging.getLogger("eval_run")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(str(log_path), mode="a" if resuming else "w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Save config (only on fresh run)
    if not resuming:
        config = vars(args).copy()
        config["run_name"] = run_name
        config["timestamp"] = stamp
        with open(run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    logger.info("=" * 60)
    if resuming:
        logger.info(f"RESUMING evaluation ({len(skip_indices)} samples already done)")
    else:
        logger.info("HCPC-RLVR Evaluation")
    logger.info("=" * 60)
    logger.info(f"Model:      {model_tag}")
    logger.info(f"Base:       {args.base_model}")
    logger.info(f"Checkpoint: {args.checkpoint or 'None (base model)'}")
    logger.info(f"Dataset:    {args.dataset} (split={args.split})")
    logger.info(f"Subset:     {args.subset or 'all'}")
    logger.info(f"Rollouts:   {args.num_samples}")
    logger.info(f"Output:     {run_dir}")
    logger.info("=" * 60)

    # Load model
    logger.info("Loading model...")
    t0 = time.time()
    model, processor = load_model(args)
    logger.info(f"Model loaded in {time.time() - t0:.1f}s")

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}...")
    dataset = load_eval_dataset(
        args.dataset, split=args.split,
        cache_dir=args.cache_dir, subset_size=args.subset,
    )
    logger.info(f"Dataset loaded: {len(dataset)} samples")

    # Per-sample log file (append if resuming)
    per_sample_path = run_dir / "per_sample.jsonl"
    psf = open(per_sample_path, "a" if resuming else "w")

    # Accumulators — reload from previous records if resuming
    total_correct = 0
    total_done = 0
    all_correct_flags = []  # for Pass@k: list of list[bool]
    all_c_table = []
    all_d_reason = []
    all_coherence = []
    all_correct_rate = []
    all_format_ok = 0
    all_times = []

    if resuming:
        for rec in prev_records:
            total_done += 1
            total_correct += int(rec.get("relaxed_accuracy", False))
            all_correct_flags.append(rec.get("correct", []))
            div = rec.get("diversity", {})
            all_c_table.append(div.get("c_table", 0.0))
            all_d_reason.append(div.get("d_reason", 0.0))
            all_coherence.append(div.get("coherence", 0.0))
            all_correct_rate.append(div.get("correct_rate", 0.0))
            fmt_ok = rec.get("format_compliance", {}).get("fully_compliant", False)
            all_format_ok += int(fmt_ok)
            all_times.append(rec.get("time_seconds", 0.0))
        logger.info(f"Restored {total_done} previous results (Acc so far: {total_correct/total_done:.3f})")

    total_start = time.time()
    remaining = len(dataset) - len(skip_indices)
    logger.info(f"Samples remaining: {remaining}")

    for idx in range(len(dataset)):
        if idx in skip_indices:
            continue
        sample_start = time.time()
        example = dataset[idx]

        # Extract fields (handle multiple possible column names)
        image = example.get("image")
        question = example.get("query") or example.get("question") or ""
        label = example.get("label") or example.get("answer") or ""
        chart_type = example.get("chart_type", "")
        table = example.get("table", {})
        if isinstance(label, list):
            label = label[0] if label else ""
        label = str(label)

        # Generate responses
        raw_outputs = generate_responses(model, processor, image, question, args)

        # Extract answers
        answers = [extract_answer_from_output(r) for r in raw_outputs]

        # Score each rollout
        correct_flags = [relaxed_accuracy(a, label) for a in answers]
        all_correct_flags.append(correct_flags)

        # Type-aware accuracy on first prediction (reward-style)
        first_output = raw_outputs[0]
        reward_acc = type_aware_accuracy(first_output, label)

        # Primary prediction = first rollout
        first_answer = answers[0]
        first_correct = correct_flags[0]
        total_correct += int(first_correct)
        total_done += 1
        running_acc = total_correct / total_done

        # Parse structured output of first response
        parsed_first = parse_response(first_output)
        fmt_check = check_format_compliance(first_output)
        if fmt_check.get("fully_compliant", False):
            all_format_ok += 1

        # Diversity metrics (across all rollouts)
        ground_truth = {"label": label, "chart_type": chart_type, "table": table}
        try:
            div = compute_diversity_metrics(raw_outputs, ground_truth)
        except Exception:
            div = {"c_table": 0.0, "d_reason": 0.0, "coherence": 0.0, "correct_rate": 0.0}

        all_c_table.append(div["c_table"])
        all_d_reason.append(div["d_reason"])
        all_coherence.append(div["coherence"])
        all_correct_rate.append(div["correct_rate"])

        sample_time = time.time() - sample_start
        all_times.append(sample_time)

        # Console log
        status = "OK" if first_correct else "WRONG"
        q_short = question[:50] + ("..." if len(question) > 50 else "")
        logger.info(
            f"[{total_done}/{len(dataset)}] "
            f"Q: \"{q_short}\" | Label: {label} | Pred: {first_answer} | "
            f"{status} | Acc: {running_acc:.3f} ({sample_time:.1f}s)"
        )

        # Write per-sample record
        record = {
            "idx": idx,
            "question": question,
            "label": label,
            "chart_type": chart_type,
            "predictions": answers,
            "raw_outputs": raw_outputs,
            "correct": correct_flags,
            "reward_accuracy": reward_acc,
            "relaxed_accuracy": first_correct,
            "running_accuracy": running_acc,
            "parsed_first": {
                "type": parsed_first.get("type", ""),
                "answer": parsed_first.get("answer", ""),
                "parse_success": parsed_first.get("parse_success", False),
                "table_parse_success_strict": parsed_first.get("table_parse_success_strict", False),
            },
            "format_compliance": fmt_check,
            "diversity": {
                "c_table": div["c_table"],
                "d_reason": div["d_reason"],
                "coherence": div["coherence"],
                "correct_rate": div["correct_rate"],
            },
            "time_seconds": round(sample_time, 2),
        }
        psf.write(json.dumps(record) + "\n")
        psf.flush()

    psf.close()
    session_time = time.time() - total_start
    total_time = sum(all_times)  # actual inference time across all sessions

    # Aggregate metrics
    accuracy = total_correct / total_done if total_done else 0.0
    k_values = [k for k in [1, 2, 4, 8] if k <= args.num_samples]
    pass_at_k = compute_pass_at_k_spectrum(all_correct_flags, k_values)
    # Convert int keys to strings for JSON
    pass_at_k = {str(k): v for k, v in pass_at_k.items()}

    avg = lambda lst: sum(lst) / len(lst) if lst else 0.0

    summary = {
        "model": model_tag,
        "checkpoint": args.checkpoint,
        "base_model": args.base_model,
        "dataset": args.dataset,
        "split": args.split,
        "subset_size": total_done,
        "num_generations_per_sample": args.num_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "accuracy": round(accuracy, 4),
        "pass_at_k": pass_at_k,
        "c_table": round(avg(all_c_table), 4),
        "d_reason": round(avg(all_d_reason), 4),
        "coherence": round(avg(all_coherence), 4),
        "correct_rate": round(avg(all_correct_rate), 4),
        "format_compliance_rate": round(all_format_ok / total_done, 4) if total_done else 0.0,
        "avg_time_per_sample": round(avg(all_times), 2),
        "total_time_seconds": round(total_time, 1),
        "total_time_human": _format_time(total_time),
        "session_time_seconds": round(session_time, 1),
        "num_samples_evaluated": total_done,
        "timestamp": stamp,
    }

    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print final summary
    logger.info("")
    logger.info("=" * 40)
    logger.info("       FINAL RESULTS")
    logger.info("=" * 40)
    logger.info(f"Model:        {model_tag}")
    logger.info(f"Dataset:      {args.dataset} ({total_done} samples)")
    logger.info(f"Accuracy:     {accuracy:.2%}")
    for k, v in pass_at_k.items():
        logger.info(f"Pass@{k}:      {v:.2%}")
    logger.info(f"C_table:      {summary['c_table']:.3f}")
    logger.info(f"D_reason:     {summary['d_reason']:.3f}")
    logger.info(f"Coherence:    {summary['coherence']:.3f}")
    logger.info(f"Correct Rate: {summary['correct_rate']:.3f}")
    logger.info(f"Format Rate:  {summary['format_compliance_rate']:.2%}")
    logger.info(f"Avg Time:     {summary['avg_time_per_sample']:.1f}s/sample")
    logger.info(f"Total Time:   {summary['total_time_human']}")
    logger.info("=" * 40)
    logger.info(f"Results saved to: {run_dir}")

    return summary


def _format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    else:
        h, rem = divmod(int(seconds), 3600)
        m, s = divmod(rem, 60)
        return f"{h}h {m}m {s}s"


if __name__ == "__main__":
    args = parse_args()
    if not args.resume and not args.dataset:
        print("Error: --dataset is required (unless using --resume)")
        sys.exit(1)
    torch.manual_seed(args.seed)
    run_eval(args)
