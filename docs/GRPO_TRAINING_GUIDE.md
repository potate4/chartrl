# Chart-RVR GRPO Training Guide

Complete guide for training Chart-RVR model using GRPO (Group Relative Policy Optimization) with configurable speed optimizations.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Training Configurations](#training-configurations)
3. [Command-Line Arguments](#command-line-arguments)
4. [Speed Optimization Strategies](#speed-optimization-strategies)
5. [Training Time Estimates](#training-time-estimates)
6. [Monitoring Training](#monitoring-training)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Configuration](#advanced-configuration)

---

## Quick Start

### Default Training (Paper Configuration)

```bash
# Full training as described in paper (~24 days on A100 40GB)
accelerate launch --config_file=deepspeed_zero3.yaml main.py \
  --mode grpo \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025
```

### Fast Training (Recommended for Testing)

```bash
# Fast training with subset (~21 hours on A100 40GB)
accelerate launch --config_file=deepspeed_zero3.yaml main.py \
  --mode grpo \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025 \
  --subset-size 10000 \
  --num-epochs 1 \
  --num-generations 2
```

### Ultra-Fast Testing

```bash
# Ultra-fast for code testing (~2 hours)
accelerate launch --config_file=deepspeed_zero3.yaml main.py \
  --mode grpo \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025 \
  --subset-size 1000 \
  --num-epochs 1 \
  --num-generations 2
```

---

## Training Configurations

### Configuration Presets

| Preset | Samples | Epochs | Generations | Time | Use Case |
|--------|---------|--------|-------------|------|----------|
| **Paper (Default)** | 34K | 4 | 4 | ~24 days | Full reproduction |
| **Fast** | 10K | 1 | 2 | ~21 hours | Quick validation |
| **Moderate** | 34K | 1 | 2 | ~3 days | Good balance |
| **Safe** | 34K | 1 | 4 | ~6 days | Conservative speedup |
| **Ultra-Fast** | 1K | 1 | 2 | ~2 hours | Code testing |

### Preset Commands

#### Paper (Default)
```bash
accelerate launch --config_file=deepspeed_zero3.yaml main.py \
  --mode grpo \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025
```

#### Fast (Recommended)
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

#### Moderate
```bash
accelerate launch --config_file=deepspeed_zero3.yaml main.py \
  --mode grpo \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025 \
  --num-epochs 1 \
  --num-generations 2
```

#### Safe
```bash
accelerate launch --config_file=deepspeed_zero3.yaml main.py \
  --mode grpo \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025 \
  --num-epochs 1
```

#### Ultra-Fast
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

---

## Command-Line Arguments

### Core Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mode` | str | required | Training mode (use `grpo` for GRPO training) |
| `--vlm-name` | str | required | Vision-language model name (use `qwen2-5-3b`) |
| `--dataset-name` | str | required | Validation dataset (use `evochart`) |
| `--seed` | int | 2025 | Random seed for reproducibility |

### Speed Optimization Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--subset-size` | int | None | Number of training samples to use (e.g., 10000). None = full dataset (34K) |
| `--num-epochs` | int | 4 | Number of training epochs. Reduce to 1-2 for faster training |
| `--num-generations` | int | 4 | Generations per sample for GRPO. Reduce to 2 for 50% speedup |
| `--batch-size` | int | 2 | Per-device batch size. Increase to 4 for fewer steps (uses more memory) |
| `--disable-gradient-checkpointing` | flag | False | Disable gradient checkpointing for 20-30% speedup (uses ~15-18GB memory) |

### Examples

**Use 10K sample subset:**
```bash
--subset-size 10000
```

**Train for 1 epoch instead of 4:**
```bash
--num-epochs 1
```

**Use 2 generations instead of 4 (50% faster):**
```bash
--num-generations 2
```

**Increase batch size for fewer steps:**
```bash
--batch-size 4
```

**Disable gradient checkpointing for speed:**
```bash
--disable-gradient-checkpointing
```

**Combine multiple optimizations:**
```bash
--subset-size 10000 --num-epochs 1 --num-generations 2
```

---

## Speed Optimization Strategies

### 1. Reduce Training Samples (`--subset-size`)

**What it does:**
- Trains on a subset of the full 34,194 samples
- Useful for quick validation or proof-of-concept

**Trade-offs:**
- ✅ Proportional speedup (10K samples = 3× faster)
- ✅ Good for testing if approach works
- ❌ May not reach paper's full performance
- ❌ Less diverse training data

**Recommendations:**
- **10,000 samples**: Good balance for validation
- **5,000 samples**: Quick testing
- **1,000 samples**: Code/pipeline testing only

**Example:**
```bash
# Train on 10K samples instead of 34K
--subset-size 10000
```

---

### 2. Reduce Epochs (`--num-epochs`)

**What it does:**
- Trains for fewer passes through the dataset
- Paper uses 4 epochs, but 1-2 often sufficient

**Trade-offs:**
- ✅ 4× speedup (1 epoch vs 4 epochs)
- ✅ Still covers all training data once
- ✅ Can resume training if needed
- ❌ May not fully converge
- ❌ Lower final performance

**Recommendations:**
- **1 epoch**: Quick validation, ~80% of full performance
- **2 epochs**: Better convergence, ~90% of full performance
- **4 epochs**: Full paper configuration

**Example:**
```bash
# Train for 1 epoch instead of 4
--num-epochs 1
```

---

### 3. Reduce Generations (`--num-generations`)

**What it does:**
- GRPO generates multiple completions per sample to compute rewards
- Reducing from 4→2 cuts generation time in half

**Trade-offs:**
- ✅ ~50% speedup (2 generations vs 4)
- ✅ GRPO still works with 2 generations
- ❌ Less diverse samples for policy gradient estimation
- ❌ Potentially noisier gradient estimates

**Recommendations:**
- **2 generations**: Good balance, still effective
- **4 generations**: Paper configuration (more stable)

**Example:**
```bash
# Use 2 generations instead of 4
--num-generations 2
```

---

### 4. Increase Batch Size (`--batch-size`)

**What it does:**
- Processes more samples per step
- Reduces total number of training steps

**Trade-offs:**
- ✅ 2× speedup (batch size 4 vs 2)
- ✅ More stable gradients
- ❌ Uses more GPU memory (~15-20GB)
- ❌ May not fit in 40GB with current settings

**Recommendations:**
- **2**: Default, fits comfortably in 40GB
- **4**: If you have memory headroom (test first!)

**Example:**
```bash
# Increase batch size to 4
--batch-size 4
```

**⚠️ Warning:** May cause OOM. Monitor with `nvidia-smi`.

---

### 5. Disable Gradient Checkpointing (`--disable-gradient-checkpointing`)

**What it does:**
- Gradient checkpointing trades memory for computation
- Disabling it speeds up training but uses more memory

**Trade-offs:**
- ✅ 20-30% speedup
- ❌ Memory increases from 12GB → 15-18GB
- ⚠️ Still fits in A100 40GB

**Recommendations:**
- Enable (default): If you want to be safe with memory
- Disable: If you want speed and have 40GB GPU

**Example:**
```bash
# Disable gradient checkpointing for speedup
--disable-gradient-checkpointing
```

---

## Training Time Estimates

### Time Calculation Formula

```
Steps = (num_samples / (batch_size × 2)) × num_epochs
Time = Steps × time_per_step
```

**Note:** `× 2` because gradient_accumulation_steps=2

### Estimated Times on A100 40GB

| Configuration | Samples | Epochs | Gens | Batch | Time/Step | Total Time |
|---------------|---------|--------|------|-------|-----------|------------|
| Paper (Default) | 34K | 4 | 4 | 2 | 60s | ~24 days |
| Safe | 34K | 1 | 4 | 2 | 60s | ~6 days |
| Moderate | 34K | 1 | 2 | 2 | 30s | ~3 days |
| Fast | 10K | 1 | 2 | 2 | 30s | ~21 hours |
| Fast + No Ckpt | 10K | 1 | 2 | 2 | 21s | ~15 hours |
| Ultra-Fast | 1K | 1 | 2 | 2 | 30s | ~2 hours |

**Variables:**
- **time_per_step** depends on:
  - Number of generations (4 gens ≈ 60s, 2 gens ≈ 30s)
  - Gradient checkpointing (enabled ≈ +30% time)
  - Model size and hardware

---

## Monitoring Training

### Check Training Progress

**GPU Usage:**
```bash
# Monitor GPU memory and utilization
nvidia-smi

# Continuous monitoring (updates every 2 seconds)
watch -n 2 nvidia-smi
```

**Expected GPU Memory:**
- With gradient checkpointing: **12-13GB**
- Without gradient checkpointing: **15-18GB**
- If you see >20GB, there may be an issue

**Training Logs:**
```bash
# View live training output
tail -f <training_log_file>

# Check WandB logs (if enabled)
wandb sync ./wandb/offline-run-*
```

### What to Look For

**Successful Training Indicators:**

1. **LoRA Loaded:**
   ```
   ✓ LoRA adapters applied successfully
   trainable params: 18,576,384 || all params: 3,773,199,360 || trainable%: 0.4923
   ```

2. **Reward Functions Working:**
   ```
   Format rewards: [0.0, 2.0, 0.0, 2.0]
   Rewards Accuracy: [1.0, 0.0, 1.0, 1.0]
   Length Rewards: [1.25, 1.5, 1.0, 1.25]
   Count rewards: [0, 2, 0, 2]
   Graph Type Rewards: [1.0, 1.0, 0.0, 1.0]
   Table style rewards: [1.625, 0.75, 2.25, 1.375]
   Process Style Rewards: [0.69, 0.76, 0.65, 0.70]
   ```

3. **Training Progress:**
   ```
   0% 10/34194 [08:30<2000:25:19, 52.66s/it]
   ```

4. **Checkpoints Saving:**
   ```
   Saving model checkpoint to grpo-start-ckpts/.../checkpoint-500
   ```

### Reward Trends

**Early Training (Steps 0-1000):**
- Format rewards: Mostly 0.0, occasional 2.0
- Accuracy: Low (0.0-0.3)
- Table/Process: Variable (0.0-1.0)

**Mid Training (Steps 1000-5000):**
- Format rewards: More consistent 2.0
- Accuracy: Improving (0.3-0.7)
- Table/Process: Stabilizing (0.5-0.8)

**Late Training (Steps 5000+):**
- Format rewards: Mostly 2.0
- Accuracy: High (0.7-1.0)
- Table/Process: Stable (0.6-0.9)

---

## Troubleshooting

### Issue: Training Too Slow

**Problem:** Training will take weeks

**Solutions:**
1. Use `--subset-size 10000` for faster iteration
2. Use `--num-epochs 1` instead of 4
3. Use `--num-generations 2` instead of 4
4. Use `--disable-gradient-checkpointing` for 30% speedup
5. Combine multiple optimizations

**Example:**
```bash
--subset-size 10000 --num-epochs 1 --num-generations 2 --disable-gradient-checkpointing
```

---

### Issue: Out of Memory (OOM)

**Problem:** `CUDA out of memory` error

**Solutions:**

1. **Reduce batch size:**
   ```bash
   --batch-size 1
   ```

2. **Enable gradient checkpointing** (if disabled):
   ```bash
   # Remove --disable-gradient-checkpointing flag
   ```

3. **Reduce max lengths** (edit main.py):
   ```python
   max_prompt_length = 3072,  # Reduce from 4096
   max_completion_length = 512,  # Reduce from 768
   ```

4. **Reduce generations:**
   ```bash
   --num-generations 2
   ```

---

### Issue: Reward Functions Failing

**Problem:** Seeing errors like `Failed to parse JSON` or `Failed to compare tables`

**Diagnosis:** This is **NORMAL** in early training!

**Explanation:**
- Model hasn't learned output format yet
- Reward functions handle errors gracefully (give 0.0 reward)
- Model learns from these failures
- Should improve as training progresses

**When to Worry:**
- If errors persist after 5,000+ steps
- If ALL completions fail (check model loading)

---

### Issue: Training Crashes/Stops

**Problem:** Training stops unexpectedly

**Solutions:**

1. **Check Colab runtime:**
   - Colab may disconnect after 12-24 hours
   - Use Colab Pro for longer sessions

2. **Check disk space:**
   ```bash
   df -h
   ```

3. **Resume from checkpoint:**
   ```bash
   # Training auto-resumes from last checkpoint
   # Just re-run the same command
   ```

4. **Save to Google Drive** (optional):
   ```python
   # Mount Drive
   from google.colab import drive
   drive.mount('/content/drive')

   # Symlink checkpoint directory
   !ln -s /content/drive/MyDrive/chartrl-checkpoints grpo-start-ckpts
   ```

---

### Issue: Low Accuracy Rewards

**Problem:** Accuracy rewards stay low throughout training

**Diagnosis:**

1. **Check if format rewards are improving:**
   - If format rewards are low, model can't learn to answer correctly
   - Focus on format first

2. **Check training data:**
   - Verify dataset has correct labels
   - Check example completions

3. **Normal pattern:**
   - Accuracy improves slower than format
   - May take 2-3 epochs to see significant improvement

---

## Advanced Configuration

### Resume Training from Checkpoint

Training automatically resumes from the last checkpoint if you re-run the same command:

```bash
# Just re-run - it will auto-detect and resume
accelerate launch --config_file=deepspeed_zero3.yaml main.py \
  --mode grpo \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025 \
  --subset-size 10000 \
  --num-epochs 1
```

Checkpoints are saved every 500 steps to:
```
grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-2025/checkpoint-{step}/
```

---

### Evaluate Trained Model

After training completes, evaluate on all benchmarks:

```bash
# Evaluate on ChartQA
python main.py --mode eval --vlm-name qwen2-5-3b --dataset-name chartqa

# Evaluate on PlotQA
python main.py --mode eval --vlm-name qwen2-5-3b --dataset-name plotqa

# Evaluate on ChartFC
python main.py --mode eval --vlm-name qwen2-5-3b --dataset-name chartfc

# Evaluate on EvoChart
python main.py --mode eval --vlm-name qwen2-5-3b --dataset-name evochart

# Evaluate on ChartQA Pro
python main.py --mode eval --vlm-name qwen2-5-3b --dataset-name chartqapro

# Evaluate on ChartBench
python main.py --mode eval --vlm-name qwen2-5-3b --dataset-name chartbench
```

---

### Multi-Stage Training Strategy

**Recommended approach for best results:**

**Stage 1: Fast Validation (21 hours)**
```bash
# Train on 10K samples to validate approach
accelerate launch --config_file=deepspeed_zero3.yaml main.py \
  --mode grpo \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025 \
  --subset-size 10000 \
  --num-epochs 1 \
  --num-generations 2
```

**Stage 2: Medium Training (3 days)**
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

**Stage 3: Full Training (6 days)** - Optional
```bash
# For best results, increase to 4 generations
accelerate launch --config_file=deepspeed_zero3.yaml main.py \
  --mode grpo \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025 \
  --num-epochs 1
```

---

### Custom Learning Rate

If you want to adjust learning rate (default: 1e-5):

Edit `main.py` around line 671:
```python
learning_rate = 5e-6,  # Try different values: 5e-6, 1e-5, 2e-5
```

---

### Custom LoRA Configuration

If you want to adjust LoRA rank (default: 8):

Edit `main.py` around line 616-622:
```python
grpo_peft_config = LoraConfig(
    r=16,  # Try different ranks: 8, 16, 32, 64
    lora_alpha=32,  # Usually 2 × rank
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    task_type="CAUSAL_LM",
)
```

---

## Performance Expectations

### Expected Results by Configuration

| Configuration | ChartQA | PlotQA | ChartFC | Training Time |
|---------------|---------|--------|---------|---------------|
| **Paper (Full)** | ~85% | ~79% | ~78% | ~24 days |
| **Safe (1 epoch, 4 gens)** | ~80% | ~74% | ~73% | ~6 days |
| **Moderate (1 epoch, 2 gens)** | ~75% | ~70% | ~68% | ~3 days |
| **Fast (10K, 1 epoch, 2 gens)** | ~65% | ~60% | ~58% | ~21 hours |

*Note: These are estimates. Actual performance may vary.*

---

## Summary of Speed vs Quality Trade-offs

| Optimization | Speedup | Quality Impact | Recommended |
|--------------|---------|----------------|-------------|
| 1 epoch (vs 4) | 4× | -10% to -15% | ✅ Yes |
| 2 gens (vs 4) | 2× | -5% to -10% | ✅ Yes |
| 10K samples (vs 34K) | 3× | -15% to -20% | ✅ For testing |
| No grad ckpt | 1.3× | 0% | ⚠️ If memory allows |
| Batch size 4 (vs 2) | 2× | 0% to +5% | ⚠️ May OOM |

**Best Combination for Quick Results:**
- 10K samples + 1 epoch + 2 generations = **27× speedup** (~21 hours)
- Expected performance: ~65-70% of full training
- Good for validation and proof-of-concept

---

## Questions?

If you encounter issues not covered in this guide:

1. Check training logs for specific error messages
2. Monitor GPU memory with `nvidia-smi`
3. Verify all 7 reward functions are printing values
4. Check that LoRA parameters show ~0.49% trainable

For additional help, refer to:
- `COLAB_EXECUTION_GUIDE.md` - Colab-specific setup
- `CHANGES_MADE.md` - Code modifications from original
- Paper: Chart-RVR (Reinforcement Learning with Verifiable Rewards for Chart Question Answering)
