# Training on Kaggle 2x T4 GPUs Guide

Complete guide for running Chart-RVR GRPO training on Kaggle with 2x T4 GPUs (16GB each).

---

## ⚠️ Important: Speed Expectations

**TL;DR: 2x T4 will be SLOWER than 1x A100, but it's FREE!**

### Performance Comparison

| Setup | Speed | Cost | Recommendation |
|-------|-------|------|----------------|
| **1x A100 (Colab Pro)** | Fastest | $10-20/month | Best for speed |
| **2x T4 (Kaggle)** | 10-20× slower | FREE | Best for free option |

**Why 2x T4 is slower:**
- T4 is older architecture (Turing vs Ampere)
- A100 has ~10-20× better AI training performance
- 2x T4 gives only ~1.3-1.7× speedup vs 1x T4 (communication overhead)
- **Net result:** 2x T4 ≈ 50-70% speed of 1x A100

**Training time estimates:**
- Fast config (10K samples): ~40-50 hours on 2x T4 vs ~21 hours on 1x A100
- Full config (34K samples): Weeks on 2x T4 vs days on 1x A100

**When to use Kaggle 2x T4:**
- ✅ You don't want to pay for Colab Pro
- ✅ You have time to wait
- ✅ You want to test/validate before spending money
- ❌ You need results quickly

---

## 🚀 Setup Instructions

### Step 1: Create Kaggle Notebook

1. Go to [Kaggle.com](https://www.kaggle.com/)
2. Create new notebook
3. **Settings** → **Accelerator** → Select **"2 x T4 GPU"**
4. **Settings** → **Persistence** → Enable (to save checkpoints)

### Step 2: Upload Code Files

Upload these files to Kaggle:
- `main.py` (with configurable arguments)
- `grpo_utils.py`
- `dataset_process.py`
- `models.py`
- `prompts.py`
- `utils.py`
- `metrics.py`
- `deepspeed_zero3_2gpu.yaml` ← **Use this for 2 GPUs**
- `requirements.txt`

### Step 3: Install Dependencies

```python
# In Kaggle notebook cell
!pip install -q -r requirements.txt
```

---

## 💾 Memory Optimization for T4 (16GB each)

### Required Changes for T4

**You MUST reduce memory usage to fit in 16GB per GPU:**

#### Option 1: Reduce Batch Size + Use QLoRA (Recommended)

```bash
# In Kaggle notebook
!accelerate launch --config_file=deepspeed_zero3_2gpu.yaml main.py \
  --mode grpo \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025 \
  --subset-size 10000 \
  --num-epochs 1 \
  --num-generations 2 \
  --batch-size 1 \
  --disable-gradient-checkpointing
```

**If OOM, add 4-bit quantization:**

Modify `main.py` around line 91 (where model is loaded):

```python
# Add quantization config
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Update model loading in models.py
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    quantization_config=bnb_config,  # Add this
    torch_dtype=torch.bfloat16,
    cache_dir=cache_dir,
)
```

**Trade-off:** Training will be slower (~1.5-2× slower with 4-bit), but will fit in memory.

#### Option 2: Reduce Sequence Lengths

Edit `main.py` around lines 666-669:

```python
max_prompt_length = 2048,  # Reduced from 4096
max_completion_length = 384,  # Reduced from 768
```

---

## 📝 Training Commands for Kaggle 2x T4

### Ultra-Fast Test (Verify Setup)

```bash
# Should complete in ~4-6 hours
!accelerate launch --config_file=deepspeed_zero3_2gpu.yaml main.py \
  --mode grpo \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025 \
  --subset-size 1000 \
  --num-epochs 1 \
  --num-generations 2 \
  --batch-size 1
```

### Fast Training (Recommended)

```bash
# Should complete in ~40-50 hours
!accelerate launch --config_file=deepspeed_zero3_2gpu.yaml main.py \
  --mode grpo \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025 \
  --subset-size 10000 \
  --num-epochs 1 \
  --num-generations 2 \
  --batch-size 1
```

### With 4-bit Quantization (If OOM)

First modify `main.py` to add 4-bit quantization (see above), then:

```bash
# Slower but uses less memory
!accelerate launch --config_file=deepspeed_zero3_2gpu.yaml main.py \
  --mode grpo \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025 \
  --subset-size 10000 \
  --num-epochs 1 \
  --num-generations 2 \
  --batch-size 1
```

---

## 🔍 Monitoring on Kaggle

### Check GPU Usage

```python
!nvidia-smi
```

**Expected on 2x T4:**
- GPU 0: 10-14GB / 16GB
- GPU 1: 10-14GB / 16GB

**If you see >15GB:** Reduce batch size or add 4-bit quantization

### Check Training Progress

Training logs will show:
```
[Rank 0] LoRA adapters applied
[Rank 1] LoRA adapters applied
trainable params: 18,576,384 || all params: 3,773,199,360 || trainable%: 0.4923

GRPO TRAINING CONFIGURATION:
  Training samples: 10000
  Epochs: 1
  Batch size: 1
  Generations per sample: 2
  Gradient checkpointing: False
```

---

## ⚠️ T4 Limitations

### 1. No Flash Attention 2

T4 uses Turing architecture which doesn't support Flash Attention 2.

**Impact:**
- Slightly slower training
- Slightly higher memory usage
- **Solution:** Already handled by code

### 2. Slower Generation

T4 generation speed is ~10× slower than A100.

**Impact:**
- GRPO generates 2-4 completions per sample
- This is your bottleneck
- **Solution:** Use `--num-generations 2` instead of 4

### 3. Kaggle Time Limits

Kaggle has execution time limits:
- **12 hours** for GPU sessions (may disconnect)
- **Can resume** from checkpoints

**Solution:**
- Training auto-saves every 500 steps
- Re-run same command to resume

---

## 🆘 Troubleshooting

### Issue: Out of Memory on T4

**Error:** `CUDA out of memory`

**Solutions (try in order):**

1. **Reduce batch size to 1:**
   ```bash
   --batch-size 1
   ```

2. **Reduce sequence lengths:**
   Edit `main.py` lines 666-669:
   ```python
   max_prompt_length = 2048,
   max_completion_length = 384,
   ```

3. **Add 4-bit quantization:**
   See "Option 1" above

4. **Reduce generations:**
   ```bash
   --num-generations 2
   ```

5. **Enable CPU offloading:**
   Edit `deepspeed_zero3_2gpu.yaml`:
   ```yaml
   offload_optimizer_device: cpu
   offload_param_device: cpu
   ```

---

### Issue: Training Very Slow

**This is normal on T4!**

**Expectations:**
- ~100-150 seconds per step (vs ~50-70s on A100)
- 10K samples ≈ 40-50 hours
- 34K samples ≈ 7-10 days

**To speed up:**
- Use smaller subset (`--subset-size 5000`)
- Reduce generations (`--num-generations 2`)
- Switch to Colab Pro A100 if you need results faster

---

### Issue: Kaggle Session Timeout

**Problem:** Kaggle disconnects after 12 hours

**Solution:**
Training auto-saves checkpoints every 500 steps. Just re-run the same command:

```bash
# Will resume from last checkpoint automatically
!accelerate launch --config_file=deepspeed_zero3_2gpu.yaml main.py \
  --mode grpo \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025 \
  --subset-size 10000 \
  --num-epochs 1 \
  --num-generations 2 \
  --batch-size 1
```

---

### Issue: Multi-GPU Not Working

**Error:** Only using 1 GPU

**Check:**
1. Verify 2 GPUs available:
   ```python
   !nvidia-smi
   ```

2. Verify config file:
   ```bash
   !cat deepspeed_zero3_2gpu.yaml | grep num_processes
   ```
   Should show: `num_processes: 2`

3. Check logs for:
   ```
   [Rank 0] ...
   [Rank 1] ...
   ```

---

## 📊 Performance Expectations

### Training Time Estimates (2x T4)

| Configuration | Samples | Time | vs A100 |
|---------------|---------|------|---------|
| **Ultra-Fast** | 1K | 4-6 hours | ~2× slower |
| **Fast** | 10K | 40-50 hours | ~2× slower |
| **Moderate** | 34K | 7-10 days | ~2-3× slower |

### Model Quality

**Same quality as A100 training!**
- Hardware doesn't affect model quality
- Only affects training speed
- Final model will be identical

---

## 🎯 Recommended Workflow

### Stage 1: Quick Test (4-6 hours)

```bash
# Verify everything works
!accelerate launch --config_file=deepspeed_zero3_2gpu.yaml main.py \
  --mode grpo \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025 \
  --subset-size 1000 \
  --num-epochs 1 \
  --num-generations 2 \
  --batch-size 1
```

### Stage 2: Fast Training (40-50 hours)

```bash
# If Stage 1 works, train on 10K
!accelerate launch --config_file=deepspeed_zero3_2gpu.yaml main.py \
  --mode grpo \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025 \
  --subset-size 10000 \
  --num-epochs 1 \
  --num-generations 2 \
  --batch-size 1
```

### Stage 3: Decide

**If Stage 2 results are good:**
- Option A: Continue on Kaggle with full 34K (7-10 days, FREE)
- Option B: Switch to Colab Pro A100 for faster training (3 days, $10-20)

---

## 💰 Cost Analysis

| Platform | Setup | Time (10K) | Time (34K) | Cost |
|----------|-------|-----------|-----------|------|
| **Kaggle** | 2x T4 | 40-50h | 7-10 days | **FREE** |
| **Colab Pro** | 1x A100 | 21h | 3 days | $10-20/month |
| **Colab Pro+** | 1x A100 | 21h | 3 days | $50/month (more GPU hours) |

**Recommendation:**
1. Start on Kaggle (FREE) to validate approach
2. If you need results faster, upgrade to Colab Pro A100

---

## 🔗 Summary

**2x T4 on Kaggle:**
- ✅ FREE
- ✅ Works for GRPO training
- ✅ Same model quality as A100
- ❌ 2-3× slower than A100
- ❌ Requires memory optimizations
- ❌ 12-hour session limits

**Best use case:**
- Testing and validation
- Long-running experiments where time isn't critical
- Learning and experimentation without cost

**When to use A100 instead:**
- You need results quickly
- You're willing to pay
- You want to iterate faster

---

## 📚 Additional Resources

- **GRPO_TRAINING_GUIDE.md** - Detailed training guide
- **QUICK_START.md** - Quick command reference
- **COLAB_EXECUTION_GUIDE.md** - Colab-specific setup

---

## Quick Reference Card

### Minimal Command (Copy-Paste)

```bash
# Upload all files to Kaggle first
# Select "2 x T4 GPU" in Accelerator settings
# Then run:

!pip install -q -r requirements.txt

!accelerate launch --config_file=deepspeed_zero3_2gpu.yaml main.py \
  --mode grpo \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025 \
  --subset-size 10000 \
  --num-epochs 1 \
  --num-generations 2 \
  --batch-size 1
```

Expected completion: **40-50 hours**

Monitor with: `!nvidia-smi`
