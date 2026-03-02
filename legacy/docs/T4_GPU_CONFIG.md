# T4 GPU Configuration (16GB VRAM)

## The Problem

POC guide was optimized for **A100 (40GB)**. T4 has only **16GB**, causing OOM with:
```bash
--batch-size 2 --disable-gradient-checkpointing
```

**Memory usage on T4:**
- Model (bf16): ~7.5GB
- Activations (no checkpointing): ~10-12GB per batch
- **Total: ~20GB → OOM on T4!**

---

## ✅ Solution: Enable Gradient Checkpointing

### Option 1: Enable Checkpointing (RECOMMENDED)

**Remove the `--disable-gradient-checkpointing` flag:**

```bash
# GRPO for T4 (Cell 6)
!accelerate launch --config_file=deepspeed_zero3.yaml main.py \
  --mode grpo \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025 \
  --subset-size 100 \
  --num-epochs 1 \
  --num-generations 2 \
  --batch-size 2 \
  2>&1 | tee poc_grpo.log
```

**What changed:**
- ❌ Removed: `--disable-gradient-checkpointing`
- ✅ Gradient checkpointing: **ENABLED** (default)
- ✅ Memory usage: ~8-10GB (fits on T4!)
- ⚠️ Slowdown: +20-30% (60 min instead of 30 min)

---

### Option 2: Reduce Batch Size

**If Option 1 still OOMs, reduce batch size to 1:**

```bash
# GRPO for T4 with batch_size=1
!accelerate launch --config_file=deepspeed_zero3.yaml main.py \
  --mode grpo \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025 \
  --subset-size 100 \
  --num-epochs 1 \
  --num-generations 2 \
  --batch-size 1 \
  2>&1 | tee poc_grpo.log
```

**What changed:**
- ✅ Batch size: 2 → 1
- ✅ Memory: ~6-8GB (definitely fits T4!)
- ⚠️ Slowdown: 2x longer (100 steps instead of 50)

---

### Option 3: Both (Ultra Safe for T4)

**Enable checkpointing AND reduce batch size:**

```bash
# GRPO for T4 (safest config)
!accelerate launch --config_file=deepspeed_zero3.yaml main.py \
  --mode grpo \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025 \
  --subset-size 100 \
  --num-epochs 1 \
  --num-generations 2 \
  --batch-size 1 \
  2>&1 | tee poc_grpo.log
```

**What changed:**
- ✅ Gradient checkpointing: ENABLED
- ✅ Batch size: 1
- ✅ Memory: ~5-7GB (T4 has plenty of headroom)
- ⚠️ Slowdown: ~2.5x (75 min instead of 30 min)

---

## NSR Configuration for T4

**Same fixes apply to NSR:**

```bash
# NSR for T4 (Cell 7)
!accelerate launch --config_file=deepspeed_zero3.yaml main.py \
  --mode nsr \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025 \
  --subset-size 100 \
  --num-epochs 1 \
  --num-generations 4 \
  --batch-size 1 \
  --reward-threshold 0.5 \
  2>&1 | tee poc_nsr.log
```

**Note:** NSR uses `--num-generations 4` (vs GRPO's 2), so it needs more memory. Batch size 1 is safer.

---

## Did NSR Integration Change Memory Usage?

**No!** NSR integration only adds reward filtering logic (CPU-side), not GPU memory.

**What causes OOM:**
1. ❌ `--disable-gradient-checkpointing` (saves activations in GPU memory)
2. ❌ T4's 16GB limit vs A100's 40GB
3. ❌ POC guide was written for A100

---

## Memory Comparison

| Config | GPU | Batch Size | Checkpointing | Memory | Time (100 samples) |
|--------|-----|------------|---------------|--------|-------------------|
| **POC (A100)** | A100 40GB | 2 | Disabled | ~20GB | 30 min |
| **T4 Option 1** | T4 16GB | 2 | **Enabled** | ~10GB | 60 min |
| **T4 Option 2** | T4 16GB | 1 | Disabled | ~10GB | 60 min |
| **T4 Option 3** | T4 16GB | 1 | **Enabled** | ~7GB | 75 min |

---

## Recommended Config for T4

**Use Option 1** (enable checkpointing, keep batch_size=2):

### Complete T4 POC Commands

```python
# Cell 6: GRPO for T4
print("="*80)
print("TRAINING 1/2: GRPO BASELINE (T4 CONFIG)")
print("="*80)

!accelerate launch --config_file=deepspeed_zero3.yaml main.py \
  --mode grpo \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025 \
  --subset-size 100 \
  --num-epochs 1 \
  --num-generations 2 \
  --batch-size 2 \
  2>&1 | tee poc_grpo.log

print("\n" + "="*80)
print("✓ GRPO training complete")
print("="*80)
```

```python
# Cell 7: NSR for T4
print("="*80)
print("TRAINING 2/2: NSR (T4 CONFIG)")
print("="*80)

!accelerate launch --config_file=deepspeed_zero3.yaml main.py \
  --mode nsr \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025 \
  --subset-size 100 \
  --num-epochs 1 \
  --num-generations 4 \
  --batch-size 1 \
  --reward-threshold 0.5 \
  2>&1 | tee poc_nsr.log

print("\n" + "="*80)
print("✓ NSR training complete")
print("="*80)
```

**Time estimate:**
- GRPO: ~60 min (vs 30 min on A100)
- NSR: ~90 min (vs 30 min on A100)
- **Total: ~2.5 hours**

---

## Verify T4 GPU

**Check before running:**

```python
# Cell 1 - Verify GPU
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Should output:
# Tesla T4, 15360 MiB  (16GB)

# If you see A100, use the original POC commands!
```

---

## Summary

**OOM Cause:**
- ✅ T4 has 16GB (not 40GB like A100)
- ✅ `--disable-gradient-checkpointing` uses too much memory
- ❌ NSR integration did NOT change memory usage

**Fix:**
- ✅ Remove `--disable-gradient-checkpointing` flag
- ✅ Or reduce `--batch-size 2` to `--batch-size 1`

**Recommended:**
- GRPO: batch_size=2, checkpointing enabled
- NSR: batch_size=1, checkpointing enabled (safer due to 4 generations)

**Ready to run on T4!** 🚀
