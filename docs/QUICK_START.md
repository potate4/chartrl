# Chart-RVR GRPO Training - Quick Start Guide

**TL;DR:** Copy-paste commands for different training speeds.

---

## 🚀 Recommended: Fast Training (21 hours)

**Best for:** Initial validation and proof-of-concept

```bash
accelerate launch --config_file=deepspeed_zero3.yaml main.py \
  --mode grpo \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025 \
  --subset-size 10000 \
  --num-epochs 1 \
  --num-generations 2
```

**What this does:**
- Trains on 10K samples (instead of 34K)
- 1 epoch (instead of 4)
- 2 generations per sample (instead of 4)
- **Time:** ~21 hours on A100 40GB
- **Expected performance:** ~65-70% of full training

---

## ⚡ Ultra-Fast Testing (2 hours)

**Best for:** Code testing and pipeline validation

```bash
accelerate launch --config_file=deepspeed_zero3.yaml main.py \
  --mode grpo \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025 \
  --subset-size 1000 \
  --num-epochs 1 \
  --num-generations 2
```

**What this does:**
- Trains on 1K samples (just 3% of full dataset)
- 1 epoch
- 2 generations
- **Time:** ~2 hours
- **Expected performance:** For testing only, won't match paper

---

## 📊 Moderate Training (3 days)

**Best for:** Balance between speed and performance

```bash
accelerate launch --config_file=deepspeed_zero3.yaml main.py \
  --mode grpo \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025 \
  --num-epochs 1 \
  --num-generations 2
```

**What this does:**
- Full 34K samples
- 1 epoch
- 2 generations
- **Time:** ~3 days
- **Expected performance:** ~75% of full training

---

## 🎯 Paper Configuration (24 days)

**Best for:** Full reproduction of paper results

```bash
accelerate launch --config_file=deepspeed_zero3.yaml main.py \
  --mode grpo \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025
```

**What this does:**
- Full 34K samples
- 4 epochs
- 4 generations
- **Time:** ~24 days
- **Expected performance:** Match paper results

---

## 🔧 Additional Speedups

### Add Speed Boost (+30% faster)

Add this flag to any command above for 20-30% speedup:

```bash
--disable-gradient-checkpointing
```

**Trade-off:** Uses ~15-18GB memory instead of 12-13GB (still fits on A100 40GB)

**Example:**
```bash
accelerate launch --config_file=deepspeed_zero3.yaml main.py \
  --mode grpo \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025 \
  --subset-size 10000 \
  --num-epochs 1 \
  --num-generations 2 \
  --disable-gradient-checkpointing
```

---

## 📋 All Available Arguments

```bash
--subset-size <number>           # Use subset (e.g., 10000, 5000, 1000)
--num-epochs <number>            # Number of epochs (default: 4)
--num-generations <number>       # Generations per sample (default: 4)
--batch-size <number>            # Per-device batch size (default: 2)
--disable-gradient-checkpointing # Disable for 30% speedup (more memory)
```

---

## ⏱️ Training Time Comparison

| Configuration | Command | Time | Speedup |
|---------------|---------|------|---------|
| **Ultra-Fast** | `--subset-size 1000 --num-epochs 1 --num-generations 2` | 2 hours | 288× |
| **Fast** ⭐ | `--subset-size 10000 --num-epochs 1 --num-generations 2` | 21 hours | 27× |
| **Moderate** | `--num-epochs 1 --num-generations 2` | 3 days | 8× |
| **Safe** | `--num-epochs 1` | 6 days | 4× |
| **Paper** | (no flags) | 24 days | 1× |

---

## 📝 What to Expect

### During Training

You should see:
```
✓ LoRA adapters applied successfully
trainable params: 18,576,384 || all params: 3,773,199,360 || trainable%: 0.4923
```

Then training progress:
```
  0% 10/8549 [08:30<12:45:19, 52.66s/it]
Format rewards: [0.0, 2.0, 0.0, 2.0]
Rewards Accuracy: [1.0, 0.0, 1.0, 1.0]
...
```

### GPU Memory

- **With gradient checkpointing:** 12-13GB
- **Without gradient checkpointing:** 15-18GB

Check with: `nvidia-smi`

---

## 🎓 Multi-Stage Training Strategy

**Recommended workflow:**

### Stage 1: Validate (21 hours)
```bash
# Fast training to verify everything works
accelerate launch --config_file=deepspeed_zero3.yaml main.py \
  --mode grpo \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025 \
  --subset-size 10000 \
  --num-epochs 1 \
  --num-generations 2
```

### Stage 2: Scale Up (3 days)
```bash
# If Stage 1 looks good, train on full dataset
accelerate launch --config_file=deepspeed_zero3.yaml main.py \
  --mode grpo \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025 \
  --num-epochs 1 \
  --num-generations 2
```

### Stage 3 (Optional): Full Training (6 days)
```bash
# For best results
accelerate launch --config_file=deepspeed_zero3.yaml main.py \
  --mode grpo \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025 \
  --num-epochs 1
```

---

## 🆘 Troubleshooting

### Training Too Slow?
→ Use `--subset-size 10000 --num-epochs 1 --num-generations 2`

### Out of Memory?
→ Remove `--disable-gradient-checkpointing` flag
→ Reduce `--batch-size 1`

### Want to Test Code Quickly?
→ Use `--subset-size 1000`

### Resume Training After Crash?
→ Just re-run the same command (auto-resumes from checkpoint)

---

## 📚 More Info

For detailed documentation, see:
- **GRPO_TRAINING_GUIDE.md** - Complete guide with all options
- **COLAB_EXECUTION_GUIDE.md** - Colab setup instructions
- **CHANGES_MADE.md** - Code modifications from original

---

## 🎯 TL;DR

**Just want to start training ASAP?**

Run this:
```bash
accelerate launch --config_file=deepspeed_zero3.yaml main.py \
  --mode grpo \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025 \
  --subset-size 10000 \
  --num-epochs 1 \
  --num-generations 2
```

Training completes in ~21 hours. Good enough to validate the approach works!
