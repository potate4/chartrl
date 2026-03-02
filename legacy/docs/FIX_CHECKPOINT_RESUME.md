# Fix: Training Not Resuming from Checkpoint

## 🚨 Problem

You're experiencing: **Training restarts from scratch instead of resuming from checkpoint**

Even though the code has `resume_from_checkpoint=True`, it's not working!

---

## Why This Happens

### Issue 1: `resume_from_checkpoint=True` is Ambiguous

In the code (line 721):
```python
resume_from_checkpoint=True,  # This is too vague!
```

**Problem:** `True` tells the trainer "try to find a checkpoint", but:
- ❌ It might not find the checkpoint directory
- ❌ It might not recognize DeepSpeed checkpoint format
- ❌ It starts over if it can't find valid checkpoint

### Issue 2: DeepSpeed Checkpoints Have Special Structure

DeepSpeed ZeRO-3 saves checkpoints differently than regular HuggingFace:
```
checkpoint-50/
├── global_step50/          ← DeepSpeed format
│   ├── mp_rank_00_model_states.pt
│   ├── zero_pp_rank_0_mp_rank_00_optim_states.pt
│   └── ...
├── trainer_state.json
├── training_args.bin
└── ...
```

Regular HuggingFace expects:
```
checkpoint-50/
├── model.safetensors       ← Regular format
├── optimizer.pt
└── ...
```

---

## 🔍 Diagnosis: Run These Commands First

**In your Colab/training environment, run:**

```python
# 1. Check if checkpoint directory exists
!ls -la grpo-start-ckpts/

# 2. Check what's inside your checkpoint
!ls -la grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-2025/

# 3. Check if checkpoints were saved
!ls -la grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-2025/checkpoint-*/

# 4. Check for DeepSpeed checkpoints
!find grpo-start-ckpts/ -name "*.pt" -o -name "trainer_state.json"

# 5. Check the latest checkpoint step
!cat grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-2025/trainer_state.json | grep global_step
```

**What you should see:**
- ✅ Directory exists: `grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-2025/`
- ✅ Checkpoint folders: `checkpoint-50/`, `checkpoint-100/`, etc.
- ✅ Files inside: `trainer_state.json`, model files, optimizer files

**If you DON'T see checkpoints:**
- Training probably crashed before first `save_steps` (every 50 steps)
- Or checkpoints failed to save

---

## ✅ Solution 1: Explicitly Specify Checkpoint Path

Instead of `resume_from_checkpoint=True`, use the **exact checkpoint path**.

### Step 1: Find Your Latest Checkpoint

```python
# Run this in Colab to find your latest checkpoint
import os
import glob

checkpoint_dir = "grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-2025"

if os.path.exists(checkpoint_dir):
    # Find all checkpoints
    checkpoints = sorted(
        glob.glob(os.path.join(checkpoint_dir, "checkpoint-*")),
        key=lambda x: int(x.split('-')[-1])
    )

    if checkpoints:
        latest_checkpoint = checkpoints[-1]
        print(f"✓ Latest checkpoint: {latest_checkpoint}")

        # Verify it has necessary files
        print(f"\nFiles in checkpoint:")
        !ls -la {latest_checkpoint}
    else:
        print("✗ No checkpoints found!")
else:
    print(f"✗ Directory doesn't exist: {checkpoint_dir}")
```

### Step 2: Modify Your Training Command

**Option A: Command-line argument** (if supported)

```bash
accelerate launch --config_file=deepspeed_zero3.yaml main.py \
  --mode grpo \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025 \
  --subset-size 1000 \
  --num-epochs 1 \
  --num-generations 2 \
  --batch-size 2 \
  --resume-from-checkpoint grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-2025/checkpoint-50 \
  2>&1 | tee grpo_1000_resume.log
```

**Option B: Modify main.py directly**

Change line 721 from:
```python
resume_from_checkpoint=True,
```

To:
```python
resume_from_checkpoint="grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-2025/checkpoint-50",
```

---

## ✅ Solution 2: Fix DeepSpeed Checkpoint Loading

If you're using DeepSpeed ZeRO-3, you need to tell the trainer to load DeepSpeed checkpoints.

### Add to main.py before `trainer.train()`:

```python
# Add this around line 822, right before trainer.train()

# Check for existing checkpoints and resume
checkpoint_dir = output_dir
if os.path.exists(checkpoint_dir):
    import glob
    checkpoints = sorted(
        glob.glob(os.path.join(checkpoint_dir, "checkpoint-*")),
        key=lambda x: int(x.split('-')[-1])
    )

    if checkpoints:
        latest_checkpoint = checkpoints[-1]
        logging.info(f"✓ Found checkpoint: {latest_checkpoint}")
        logging.info(f"✓ Resuming training from step {latest_checkpoint.split('-')[-1]}")

        # Resume training
        trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        logging.info("No checkpoints found. Starting fresh training.")
        trainer.train()
else:
    logging.info("Checkpoint directory doesn't exist. Starting fresh training.")
    trainer.train()
```

---

## ✅ Solution 3: Save More Frequently

If training keeps crashing before checkpoints are saved, reduce `save_steps`:

### Current setting (line 706):
```python
save_steps=50,  # Saves every 50 steps
```

### Change to:
```python
save_steps=10,  # Save every 10 steps (more frequent!)
```

**Trade-off:**
- ✅ More checkpoints = safer (less data loss if crash)
- ❌ More disk I/O = slightly slower training

For 1000 samples:
- `save_steps=50` → Checkpoints at: 50, 100, 150, 200, 250, 300, 350, 400, 450, 500
- `save_steps=10` → Checkpoints at: 10, 20, 30, ..., 490, 500 (50 checkpoints total!)

---

## 🔧 Quick Fix Script

**Run this in Colab to check and resume:**

```python
import os
import glob

# Configuration
mode = "grpo"  # or "nsr", "psr", "w-reinforce"
seed = 2025
checkpoint_base = f"grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-{mode if mode != 'grpo' else ''}{'-' if mode != 'grpo' else ''}{seed}"

print("="*80)
print("CHECKPOINT DIAGNOSIS")
print("="*80)

# Check if directory exists
if not os.path.exists(checkpoint_base):
    print(f"\n✗ Checkpoint directory doesn't exist: {checkpoint_base}")
    print("→ Training will start fresh")
else:
    print(f"\n✓ Checkpoint directory exists: {checkpoint_base}")

    # Find checkpoints
    checkpoints = sorted(
        glob.glob(os.path.join(checkpoint_base, "checkpoint-*")),
        key=lambda x: int(x.split('-')[-1])
    )

    if not checkpoints:
        print("✗ No checkpoints found in directory")
        print("→ Training crashed before first save?")
        print("→ Consider using smaller --save-steps (e.g., 10 instead of 50)")
    else:
        print(f"✓ Found {len(checkpoints)} checkpoint(s):")
        for ckpt in checkpoints:
            step = ckpt.split('-')[-1]
            print(f"  - checkpoint-{step}")

        latest = checkpoints[-1]
        latest_step = latest.split('-')[-1]
        print(f"\n✓ Latest checkpoint: checkpoint-{latest_step}")

        # Check trainer state
        trainer_state_path = os.path.join(latest, "trainer_state.json")
        if os.path.exists(trainer_state_path):
            import json
            with open(trainer_state_path, 'r') as f:
                state = json.load(f)
                print(f"✓ Global step: {state.get('global_step', 'unknown')}")
                print(f"✓ Training completed: {state.get('epoch', 0):.2f} epochs")

        print(f"\n{'='*80}")
        print("TO RESUME TRAINING, USE THIS COMMAND:")
        print(f"{'='*80}")
        print(f"\n--resume-from-checkpoint {latest}")
        print("\nOR modify main.py line 721 to:")
        print(f"  resume_from_checkpoint=\"{latest}\",")
        print(f"\n{'='*80}")

print("\n")
```

---

## 🎯 Recommended Solution for Your Case

Based on your situation (training for 1000 samples), here's what I recommend:

### 1. **Check if checkpoints exist** (run diagnosis script above)

### 2. **If checkpoints exist:**

**Modify your training command to explicitly resume:**

```bash
# Find the latest checkpoint first
!ls -la grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-2025/

# Then resume from it
!accelerate launch --config_file=deepspeed_zero3.yaml main.py \
  --mode grpo \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025 \
  --subset-size 1000 \
  --num-epochs 1 \
  --num-generations 2 \
  --batch-size 2 \
  --resume-from-checkpoint grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-2025/checkpoint-50 \
  2>&1 | tee grpo_1000_resume.log
```

### 3. **If checkpoints DON'T exist:**

Training crashed before first checkpoint. Solutions:
- Reduce `save_steps` from 50 to 10 (edit main.py line 706)
- Check Colab GPU didn't disconnect
- Check logs for errors

---

## 🔍 Common Issues & Fixes

### Issue: "Training starts from epoch 0.0 again"

**Cause:** Checkpoint not loaded

**Fix:** Explicitly specify checkpoint path (see Solution 1)

### Issue: "Checkpoint files not found"

**Cause:** Training crashed before `save_steps` interval

**Fix:**
1. Reduce `save_steps` to 10
2. Check if Colab disconnected
3. Monitor with `tail -f grpo.log`

### Issue: "DeepSpeed checkpoint format mismatch"

**Cause:** DeepSpeed vs regular HuggingFace format

**Fix:** Use Solution 2 (modify trainer.train() call)

### Issue: "Out of memory when resuming"

**Cause:** Loading checkpoint + model exceeds VRAM

**Fix:**
1. Clear GPU memory first: `torch.cuda.empty_cache()`
2. Or restart Colab runtime before resuming

---

## 📋 Quick Checklist

Before you resume training:

- [ ] Checkpoint directory exists
- [ ] At least one `checkpoint-*` folder exists
- [ ] `trainer_state.json` exists in latest checkpoint
- [ ] You know the exact checkpoint path
- [ ] You've specified it in command or code
- [ ] GPU memory is clear
- [ ] Same seed, same mode, same config as original run

---

## 💡 Pro Tip: Test Resume Locally

Before running the full 1000-sample training, test resumption:

```bash
# Run 100 samples to step 50
# Let it save checkpoint
# Stop it manually
# Try to resume
# Verify it continues from step 50, not step 0
```

If resumption works correctly, you'll see:
```
Loading checkpoint from: .../checkpoint-50
Resuming training from step 50
Training: 51/500 [10%] ...
```

If it's NOT working, you'll see:
```
Training: 1/500 [0%] ...  ← Started from 0 again!
```

---

**Need more help? Run the diagnosis script above and share the output!**
