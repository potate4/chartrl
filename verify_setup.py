#!/usr/bin/env python3
"""
Pre-training verification script for Chart-RVR
Run this before starting GRPO training to catch issues early
"""

import os
import sys

def check(condition, message, fix=""):
    """Check a condition and print result"""
    status = "✓" if condition else "✗"
    color = "\033[92m" if condition else "\033[91m"
    reset = "\033[0m"
    print(f"{color}{status}{reset} {message}")
    if not condition and fix:
        print(f"  → Fix: {fix}")
    return condition

def main():
    print("\n" + "="*60)
    print("Chart-RVR Setup Verification")
    print("="*60 + "\n")

    all_checks = []

    # 1. GPU Check
    print("1. GPU Check:")
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        all_checks.append(check(has_cuda, "CUDA available",
                               "Select A100 GPU in Colab Runtime settings"))
        if has_cuda:
            gpu_name = torch.cuda.get_device_name(0)
            is_a100 = "A100" in gpu_name
            all_checks.append(check(is_a100, f"GPU: {gpu_name}",
                                   "Recommend A100 for training"))

            # Check memory
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            has_enough_mem = mem_gb >= 35
            all_checks.append(check(has_enough_mem, f"GPU Memory: {mem_gb:.1f} GB",
                                   "Need ≥35GB for GRPO training"))
    except Exception as e:
        all_checks.append(check(False, f"GPU check failed: {e}",
                               "Install PyTorch with CUDA"))

    print()

    # 2. Dependencies Check
    print("2. Dependencies Check:")

    # PyTorch
    try:
        import torch
        all_checks.append(check(True, f"PyTorch {torch.__version__}"))
    except:
        all_checks.append(check(False, "PyTorch", "pip install torch==2.6.0"))

    # Flash Attention
    try:
        import flash_attn
        all_checks.append(check(True, f"Flash Attention {flash_attn.__version__}"))
    except:
        all_checks.append(check(False, "Flash Attention",
                               "pip install flash-attn==2.7.4.post1 --no-build-isolation"))

    # TRL
    try:
        from trl import GRPOTrainer, GRPOConfig
        all_checks.append(check(True, "TRL (GRPOTrainer found)"))
    except:
        all_checks.append(check(False, "TRL", "pip install trl==0.12.0"))

    # Transformers
    try:
        import transformers
        all_checks.append(check(True, f"Transformers {transformers.__version__}"))
    except:
        all_checks.append(check(False, "Transformers", "pip install transformers==4.53.1"))

    # Datasets
    try:
        import datasets
        all_checks.append(check(True, f"Datasets {datasets.__version__}"))
    except:
        all_checks.append(check(False, "Datasets", "pip install datasets==3.6.0"))

    # Accelerate
    try:
        import accelerate
        all_checks.append(check(True, f"Accelerate {accelerate.__version__}"))
    except:
        all_checks.append(check(False, "Accelerate", "pip install accelerate==1.7.0"))

    # DeepSpeed
    try:
        import deepspeed
        all_checks.append(check(True, f"DeepSpeed {deepspeed.__version__}"))
    except:
        all_checks.append(check(False, "DeepSpeed", "pip install deepspeed==0.15.3"))

    # PEFT
    try:
        import peft
        all_checks.append(check(True, f"PEFT {peft.__version__}"))
    except:
        all_checks.append(check(False, "PEFT", "pip install peft==0.15.2"))

    print()

    # 3. Code Modifications Check
    print("3. Code Modifications Check:")

    # Check main.py cache_dir
    try:
        with open("main.py", "r") as f:
            content = f.read()
            has_correct_cache = "cache_dir = '/content/hf_cache'" in content
            all_checks.append(check(has_correct_cache,
                                   "main.py: cache_dir updated",
                                   "Update cache_dir in main.py line 43"))
    except:
        all_checks.append(check(False, "main.py not found", "cd to chartrl directory"))

    # Check dataset_process.py cache_dir
    try:
        with open("dataset_process.py", "r") as f:
            content = f.read()
            has_correct_cache = "cache_dir = '/content/hf_cache'" in content
            all_checks.append(check(has_correct_cache,
                                   "dataset_process.py: cache_dir updated",
                                   "Update cache_dir in dataset_process.py line 40"))
    except:
        all_checks.append(check(False, "dataset_process.py not found"))

    # Check deepspeed_zero3.yaml
    try:
        with open("deepspeed_zero3.yaml", "r") as f:
            content = f.read()
            has_single_gpu = "num_processes: 1" in content
            all_checks.append(check(has_single_gpu,
                                   "deepspeed_zero3.yaml: num_processes=1",
                                   "Update num_processes in deepspeed_zero3.yaml line 16"))
    except:
        all_checks.append(check(False, "deepspeed_zero3.yaml not found"))

    print()

    # 4. Dataset Check
    print("4. Dataset Check:")
    cache_dir = '/content/hf_cache'
    dataset_path = f"{cache_dir}/grpo-chartrvr-train"

    has_cache_dir = os.path.exists(cache_dir)
    all_checks.append(check(has_cache_dir, f"Cache directory exists: {cache_dir}",
                           f"mkdir -p {cache_dir}"))

    has_dataset = os.path.exists(dataset_path)
    all_checks.append(check(has_dataset, f"Training dataset downloaded: {dataset_path}",
                           "Run: from datasets import load_dataset; load_dataset('sanchit97/chart-rvr-grpo-train')"))

    if has_dataset:
        # Check dataset size
        try:
            from datasets import load_from_disk
            ds = load_from_disk(dataset_path)
            num_samples = len(ds["train"])
            is_correct_size = 30000 <= num_samples <= 35000
            all_checks.append(check(is_correct_size,
                                   f"Dataset size: {num_samples} samples",
                                   "Expected ~34,200 samples"))
        except Exception as e:
            all_checks.append(check(False, f"Dataset loading failed: {e}"))

    print()

    # 5. Disk Space Check
    print("5. Disk Space Check:")
    try:
        import shutil
        total, used, free = shutil.disk_usage("/content")
        free_gb = free / (1024**3)
        has_space = free_gb >= 50
        all_checks.append(check(has_space,
                               f"Free disk space: {free_gb:.1f} GB",
                               "Need ≥50 GB free. Clear some space."))
    except:
        all_checks.append(check(False, "Could not check disk space"))

    print()

    # 6. HuggingFace Auth Check
    print("6. HuggingFace Check:")
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        is_logged_in = token is not None
        all_checks.append(check(is_logged_in, "HuggingFace logged in",
                               "Run: from huggingface_hub import login; login()"))
    except:
        all_checks.append(check(False, "HuggingFace auth check failed"))

    print()

    # Final Summary
    print("="*60)
    passed = sum(all_checks)
    total = len(all_checks)

    if passed == total:
        print(f"\n✓ All checks passed! ({passed}/{total})")
        print("\nYou're ready to start GRPO training!")
        print("\nRun:")
        print("  accelerate launch --config_file=deepspeed_zero3.yaml main.py \\")
        print("    --mode grpo --vlm-name qwen2-5-3b --dataset-name evochart --seed 2025")
        return 0
    else:
        print(f"\n✗ {total - passed} check(s) failed ({passed}/{total} passed)")
        print("\nFix the issues above before starting training.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
