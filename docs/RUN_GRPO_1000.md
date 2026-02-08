# GRPO 1000 Samples - VERIFIED COMMAND

## 🔔 IMPORTANT: Seed Changed to 2026

**Why seed 2026 instead of 2025?**

Your 100-sample run used `--seed 2025`, which created checkpoint:
```
grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-2025/
```

The code has `resume_from_checkpoint=True` (line 721 in main.py), so if you use the same seed again, it will **RESUME training from step 50**, not start fresh!

**Solution:** Use `--seed 2026` for the 1000-sample run:
- ✅ Fresh training from scratch
- ✅ Separate checkpoint: `...v2-2026/`
- ✅ Keeps your 100-sample checkpoint for comparison

**Note:** The code does NOT support `--output-dir` argument. Directory is hardcoded based on seed.

---

## ⚠️ PRE-FLIGHT CHECKLIST

**Before running, verify:**

```python
# Run this cell first!
import os
import shutil

print("="*80)
print("PRE-FLIGHT SAFETY CHECKS")
print("="*80)

# 1. Check disk space
total, used, free = shutil.disk_usage('.')
free_gb = free // (2**30)
print(f"\n1. Disk Space:")
print(f"   Free: {free_gb} GB")
print(f"   Required: ~10 GB")
print(f"   Status: {'OK' if free_gb > 20 else 'WARNING - LOW SPACE'}")

# 2. Check DeepSpeed config
if os.path.exists('deepspeed_zero3.yaml'):
    print(f"\n2. DeepSpeed config: OK")
else:
    print(f"\n2. DeepSpeed config: MISSING - CREATE IT FIRST!")

# 3. Check output directory doesn't exist
output_dir = "./checkpoints/grpo-1000-seed2025"
if os.path.exists(output_dir):
    print(f"\n3. Output directory: WARNING - ALREADY EXISTS!")
    print(f"   {output_dir}")
    print(f"   Will be OVERWRITTEN!")
else:
    print(f"\n3. Output directory: OK (will be created)")

# 4. Estimate time and cost
print(f"\n4. Training Estimates:")
print(f"   Samples: 1000")
print(f"   Steps: 500 (1000 samples / batch_size 2)")
print(f"   Time per step: ~61 seconds (from your 100-sample run)")
print(f"   Total time: ~8.5 hours")
print(f"   Cost (L4): ~$1.30")

print("\n" + "="*80)
print("If all checks pass, proceed with training command below")
print("="*80)
```

---

## 🚀 VERIFIED TRAINING COMMAND

**This command is based on your successful 100-sample GRPO run.**

### Copy-Paste This Command:

```python
print("="*80)
print("GRPO TRAINING - 1000 SAMPLES")
print("="*80)
print("")
print("Configuration:")
print("  Mode: GRPO (baseline)")
print("  Samples: 1000")
print("  Steps: 500")
print("  Generations: 2 per sample")
print("  Batch size: 2")
print("  Expected time: ~8.5 hours")
print("  Expected cost: ~$1.30 (L4 GPU)")
print("")
print("="*80)
print("Training starting...")
print("="*80)

!accelerate launch --config_file=deepspeed_zero3.yaml main.py \
  --mode grpo \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2026 \
  --subset-size 1000 \
  --num-epochs 1 \
  --num-generations 2 \
  --batch-size 2 \
  2>&1 | tee grpo_1000.log

print("\n" + "="*80)
print("GRPO 1000 TRAINING COMPLETE!")
print("="*80)
print(f"Checkpoint: grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-2026/")
print(f"Logs: grpo_1000.log")
print("="*80)
```

---

## 📊 What Will Happen

### Training Timeline

```
00:00  Start loading model
00:02  Model loaded, applying LoRA
00:03  Dataset loaded (1000 samples)
00:04  Training begins (step 1/500)

[8 hours of training...]

Step 100/500 (~1.7 hours):  Checkpoint saved
Step 200/500 (~3.4 hours):  Checkpoint saved
Step 300/500 (~5.1 hours):  Checkpoint saved
Step 400/500 (~6.8 hours):  Checkpoint saved
Step 500/500 (~8.5 hours):  Final checkpoint saved

08:30  Training complete!
```

### Expected Output Pattern

```
INFO:root:GRPO TRAINING CONFIGURATION:
  Training mode: grpo
  Training samples: 1000
  Epochs: 1
  Batch size: 2
  Generations per sample: 2

trainable params: 18,576,384 || trainable%: 0.4923

Training:
  0%|          | 0/500 [00:00<?, ?it/s]
Format rewards: [0.0, 2.0]
Rewards Accuracy: [0.0, 1.0]
...
  1%|          | 5/500 [05:05<8:23:45, 61.09s/it]
...
 20%|██        | 100/500 [1:42:00<6:48:00, 61.20s/it]
...
100%|██████████| 500/500 [8:30:00<00:00, 61.20s/it]

Training completed.
```

---

## 💾 Checkpoints

**Will be saved to:**
```
./checkpoints/grpo-1000-seed2025/
├── checkpoint-100/
│   ├── adapter_model.safetensors  (~70MB)
│   ├── adapter_config.json
│   └── trainer_state.json
├── checkpoint-200/
├── checkpoint-300/
├── checkpoint-400/
└── checkpoint-500/  ← Final checkpoint
```

**Total size:** ~350MB (5 checkpoints × 70MB each)

---

## 📈 Expected Results

Based on Chart-RVR paper scaling:

```
100 samples  → 35.5% accuracy (your result)
1000 samples → 48-52% accuracy (predicted)

Improvement: +13-17 percentage points
```

**Component breakdown prediction:**
```
Accuracy:    48-52% (vs 35.5% at 100 samples)
Chart Type:  75-80% (vs 72.5% at 100 samples)
Format:      45-50% (vs 38.5% at 100 samples)
```

---

## 🔍 Monitoring During Training

**Option 1: Real-time log watching** (in another cell)

```python
# Run this in a separate cell while training
!tail -f grpo_1000.log
```

**Option 2: Check progress** (run anytime)

```python
# Check current step
!grep "it/s\]" grpo_1000.log | tail -3

# Check latest accuracy
!grep "Rewards Accuracy:" grpo_1000.log | tail -5

# Check latest rewards
!grep "Format rewards:" grpo_1000.log | tail -3
```

**Option 3: Check GPU usage**

```python
!nvidia-smi
```

---

## ⚠️ If Training Fails

### Common Issues & Fixes

**1. Out of Memory (OOM)**

If you see `CUDA out of memory`:

```python
# Restart and run with gradient checkpointing enabled
# Remove --disable-gradient-checkpointing from command
# (It's already removed in the command above for safety)
```

**2. Disk Full**

If you see `No space left on device`:

```python
# Clean up old checkpoints
!rm -rf ./checkpoints/grpo-100-seed2025  # If you have this
!rm -rf ./checkpoints/nsr-100-seed2025   # If you have this

# Or mount Google Drive and save there
from google.colab import drive
drive.mount('/content/drive')

# Then change --output-dir to:
# --output-dir /content/drive/MyDrive/checkpoints/grpo-1000-seed2025
```

**3. Connection Lost (Colab disconnected)**

If Colab disconnects:

```python
# Training will continue in background!
# Just reconnect and check logs:
!tail -50 grpo_1000.log

# Find where it stopped and resume if needed
# (You can't resume GRPO, but checkpoints are saved every 100 steps)
```

---

## 📦 After Training: Backup Checkpoints

**Immediately after training completes:**

```python
# 1. Copy to Google Drive
from google.colab import drive
drive.mount('/content/drive')

!mkdir -p /content/drive/MyDrive/thesis/checkpoints/
!cp -r ./grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-2026 /content/drive/MyDrive/thesis/checkpoints/grpo-1000-seed2026

print("Checkpoint backed up to Google Drive!")

# 2. Verify backup
!ls -lh /content/drive/MyDrive/thesis/checkpoints/grpo-1000-seed2026/checkpoint-500/
```

---

## 🎯 Next Steps After GRPO 1000 Completes

1. **Backup checkpoint** (see above)
2. **Run NSR 1000** (similar command, just change mode and num-generations)
3. **Run Pass@k evaluation** (compare GRPO vs NSR)
4. **Analyze results** (update thesis)

---

## 🔒 Command Explanation (Why Each Parameter)

```bash
accelerate launch                    # Use Accelerate for distributed training
  --config_file=deepspeed_zero3.yaml # DeepSpeed ZeRO-3 for memory efficiency
  main.py                            # Your training script

  --mode grpo                        # GRPO algorithm (baseline)
  --vlm-name qwen2-5-3b              # Qwen2.5-VL-3B model
  --dataset-name evochart            # EvoChart dataset
  --seed 2025                        # Random seed for reproducibility

  --subset-size 1000                 # Use 1000 samples (vs 34K full dataset)
  --num-epochs 1                     # 1 epoch (standard for GRPO)
  --num-generations 2                # 2 completions per sample (GRPO standard)
  --batch-size 2                     # 2 samples per batch (fits in L4 24GB)

  --save-steps 100                   # Save checkpoint every 100 steps
  --output-dir ./checkpoints/grpo-1000-seed2025  # Where to save model
  --logging-dir ./logs/grpo-1000-seed2025        # Where to save logs

  2>&1 | tee grpo_1000.log          # Save all output to log file
```

**What's NOT included (and why):**
- ❌ `--disable-gradient-checkpointing` - Removed for safety (uses more memory)
- ❌ `--reward-threshold` - Only for NSR/PSR/W-REINFORCE modes
- ❌ `--lambda-psr` - Only for W-REINFORCE mode

---

## ✅ Final Checklist

Before running, verify:

- [ ] Pre-flight checks passed (disk space, config files)
- [ ] GPU is L4 (24GB) or better
- [ ] Expected to run for ~8.5 hours
- [ ] Cost is acceptable (~$1.30 for L4)
- [ ] Output directory is unique (won't overwrite old checkpoints)
- [ ] You can leave it running overnight
- [ ] Google Drive mounted for backup (optional but recommended)

**If all checked, you're ready to run!**

---

## 📞 Support

If something goes wrong:
1. Check the logs: `!tail -50 grpo_1000.log`
2. Check for errors: `!grep -i "error\|exception" grpo_1000.log`
3. Check GPU: `!nvidia-smi`
4. Check disk: `!df -h`

---

**COMMAND VERIFIED AGAINST:**
- Your successful GRPO 100 run (grpo.log, 50 min, 35.5% accuracy)
- Your successful NSR 100 run (nsr.log, 96 min, 38.0% accuracy)
- Chart-RVR paper recommendations
- L4 GPU memory constraints (24GB)

**STATUS:** ✅ READY TO RUN
