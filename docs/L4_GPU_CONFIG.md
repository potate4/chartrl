# L4 GPU Configuration (24GB VRAM)

## L4 Specifications

- **VRAM:** 24GB (50% more than T4, 60% of A100)
- **Architecture:** Ada Lovelace (newer & faster than T4)
- **Performance:** ~2x faster than T4 for inference/training

---

## ✅ Recommended Config for L4

**L4 has enough memory for the original POC commands with gradient checkpointing enabled!**

### Memory Budget

**Original A100 config (no checkpointing):**
- Model: ~7.5GB
- Activations: ~10-12GB
- **Total: ~20GB** → Fits on L4 with 4GB headroom ✓

**L4 config (with checkpointing - safer):**
- Model: ~7.5GB
- Activations: ~3-4GB (checkpointing reduces by ~70%)
- **Total: ~11GB** → Plenty of headroom on L4 ✓

---

## Recommended Commands for L4

### Option 1: With Gradient Checkpointing (RECOMMENDED)

**Safer, still fast, leaves headroom for other processes:**

**GRPO:**
```bash
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

**NSR:**
```bash
!accelerate launch --config_file=deepspeed_zero3.yaml main.py \
  --mode nsr \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025 \
  --subset-size 100 \
  --num-epochs 1 \
  --num-generations 4 \
  --batch-size 2 \
  --reward-threshold 0.5 \
  2>&1 | tee poc_nsr.log
```

**Time:**
- GRPO: ~45 min (faster than T4, slower than A100)
- NSR: ~60 min
- **Total: ~1.75 hours**

---

### Option 2: Without Gradient Checkpointing (FASTER)

**If you want maximum speed (like A100 config):**

**GRPO:**
```bash
!accelerate launch --config_file=deepspeed_zero3.yaml main.py \
  --mode grpo \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025 \
  --subset-size 100 \
  --num-epochs 1 \
  --num-generations 2 \
  --batch-size 2 \
  --disable-gradient-checkpointing \
  2>&1 | tee poc_grpo.log
```

**NSR:**
```bash
!accelerate launch --config_file=deepspeed_zero3.yaml main.py \
  --mode nsr \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025 \
  --subset-size 100 \
  --num-epochs 1 \
  --num-generations 4 \
  --batch-size 2 \
  --reward-threshold 0.5 \
  --disable-gradient-checkpointing \
  2>&1 | tee poc_nsr.log
```

**Time:**
- GRPO: ~35 min
- NSR: ~45 min
- **Total: ~1.25 hours**

**Note:** Uses ~20GB, should fit but leaves less headroom. Use Option 1 if you get OOM.

---

## GPU Comparison

| GPU | VRAM | GRPO Time | NSR Time | Total | Config |
|-----|------|-----------|----------|-------|--------|
| **A100** | 40GB | 30 min | 30 min | 1 hour | batch=2, no checkpointing |
| **L4** | 24GB | 35-45 min | 45-60 min | 1.5-1.75 hrs | batch=2, optional checkpointing |
| **T4** | 16GB | 60 min | 90 min | 2.5 hrs | batch=1-2, must checkpoint |

---

## Which Option to Use?

### Use Option 1 (with checkpointing) if:
- ✅ You want safety margin for memory
- ✅ You're running other processes in Colab
- ✅ You don't mind 20-30% slower training

### Use Option 2 (no checkpointing) if:
- ✅ You want maximum speed
- ✅ You're only running training (no other processes)
- ✅ You don't mind monitoring for OOM

**Recommendation: Start with Option 1.** If it's stable and you want more speed, try Option 2.

---

## Verify L4 GPU

```python
# Check GPU
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Should output:
# NVIDIA L4, 22528 MiB  (~24GB)
```

---

## Complete L4 POC (Recommended Config)

```python
# Cell 1: Verify L4
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# Should show: NVIDIA L4, 22528 MiB

# Cell 2-5A: Setup (dependencies, SSL fixes, etc.)
# ... (run all setup cells from POC_GRPO_VS_NSR.md)

# Cell 6: GRPO (Option 1 - with checkpointing)
print("="*80)
print("TRAINING 1/2: GRPO BASELINE (L4 CONFIG)")
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

print("\n✓ GRPO complete (~45 min)")

# Cell 7: NSR (Option 1 - with checkpointing)
print("="*80)
print("TRAINING 2/2: NSR (L4 CONFIG)")
print("="*80)

!accelerate launch --config_file=deepspeed_zero3.yaml main.py \
  --mode nsr \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025 \
  --subset-size 100 \
  --num-epochs 1 \
  --num-generations 4 \
  --batch-size 2 \
  --reward-threshold 0.5 \
  2>&1 | tee poc_nsr.log

print("\n✓ NSR complete (~60 min)")

# Cell 8-14: Compare results
# ... (run all comparison cells)
```

---

## Summary

**L4 is perfect for this POC!**

- ✅ 24GB VRAM (enough for original POC config with checkpointing)
- ✅ ~1.75 hours total (vs 1 hour on A100, 2.5 hours on T4)
- ✅ batch_size=2, gradient checkpointing enabled
- ✅ No special modifications needed

**Start with Option 1 (checkpointing enabled) - it's the sweet spot for L4!** 🚀
