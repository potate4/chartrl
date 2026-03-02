# Checkpoint Behavior Explained

## 🚨 TL;DR

**If you run GRPO 1000 with `--seed 2025`, it will RESUME from your 100-sample checkpoint, not start fresh!**

**Solution:** Use `--seed 2026` for 1000-sample run (already updated in guides).

---

## How Checkpoints Work in Your Code

### Checkpoint Directory is Based on Seed

**From `main.py` line 686-688:**
```python
mode_suffix = f"-{args.mode}" if args.mode != "grpo" else ""
output_dir = f"grpo-start-ckpts/{args.vlm_name}-prm-large-train-v2{mode_suffix}-{str(seed)}"
```

This means:
- `--seed 2025` → saves to `grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-2025/`
- `--seed 2026` → saves to `grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-2026/`
- `--mode nsr --seed 2025` → saves to `grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-nsr-2025/`

### Auto-Resume is Enabled

**From `main.py` line 721:**
```python
resume_from_checkpoint=True,  # Auto-resume if checkpoint exists
```

This means:
- If the output directory already has a checkpoint, training will **automatically resume**
- No prompt, no warning, it just continues from last checkpoint!

### Only Keeps Last 3 Checkpoints

**From `main.py` line 707:**
```python
save_total_limit=3,  # Keep only last 3 checkpoints
```

This means:
- Disk space is saved by keeping only the 3 most recent checkpoints
- Example: If you have checkpoint-100, checkpoint-200, checkpoint-300, checkpoint-400, checkpoint-500
  - Only keeps: checkpoint-300, checkpoint-400, checkpoint-500
  - Deletes: checkpoint-100, checkpoint-200

---

## Your Current Situation

### What You Already Have

```
GRPO 100 samples (seed 2025):
  Directory: grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-2025/
  Checkpoints: checkpoint-50/ (final)
  Status: ✅ Completed

NSR 100 samples (seed 2025):
  Directory: grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-nsr-2025/
  Checkpoints: checkpoint-100/ (final)
  Status: ✅ Completed
```

### What Will Happen with Different Seeds

**Scenario 1: Use `--seed 2025` for 1000 samples** ❌ **DON'T DO THIS!**

```bash
accelerate launch ... --seed 2025 --subset-size 1000
```

**Result:**
- ❌ Finds existing checkpoint-50
- ❌ Resumes from step 50 (not step 0!)
- ❌ Continues training with 1000 samples
- ❌ Overwrites your 100-sample checkpoint
- ❌ You lose your 100-sample results!

**Scenario 2: Use `--seed 2026` for 1000 samples** ✅ **RECOMMENDED!**

```bash
accelerate launch ... --seed 2026 --subset-size 1000
```

**Result:**
- ✅ Creates new directory: `...v2-2026/`
- ✅ Starts fresh from step 0
- ✅ Trains for 500 steps (1000 samples / batch 2)
- ✅ Keeps your 100-sample checkpoint (seed 2025)
- ✅ Can compare both later!

---

## Why You Can't Add --output-dir

Looking at `main.py` lines 48-74 (argument parser):
```python
parser.add_argument('--mode', ...)
parser.add_argument('--vlm-name', ...)
parser.add_argument('--seed', ...)
parser.add_argument('--subset-size', ...)
...
# NO --output-dir argument!
```

**The code does NOT support `--output-dir` or `--logging-dir` arguments.**

Output directory is **hardcoded** based on:
- Model name (`--vlm-name`)
- Training mode (`--mode`)
- Seed (`--seed`)

To change output directory, you must either:
1. Change the seed (easiest!)
2. Modify the code at line 688

---

## Solutions Comparison

| Solution | Pros | Cons | Recommended? |
|----------|------|------|--------------|
| **Use different seed (2026)** | ✅ No code changes<br>✅ Keeps both checkpoints<br>✅ Clean separation | ⚠️ Different random seed (minor) | ✅ **YES** |
| **Delete old checkpoint** | ✅ Keep same seed | ❌ Lose 100-sample results<br>❌ Can't compare scaling | ❌ No |
| **Modify code to add --output-dir** | ✅ Full control | ❌ Code changes<br>❌ May break other code | ❌ No |

---

## Your Updated Training Plan

### 100 Samples (Already Complete)

```
GRPO (seed 2025):
  ✅ grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-2025/

NSR (seed 2025):
  ✅ grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-nsr-2025/

PSR (seed 2025):
  ⏳ grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-psr-2025/

W-REINFORCE (seed 2025):
  ⏳ grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-w-reinforce-2025/
```

### 1000 Samples (Using seed 2026)

```
GRPO (seed 2026):  ← Note different seed!
  ⏳ grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-2026/

NSR (seed 2026):   ← Note different seed!
  ⏳ grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-nsr-2026/
```

**This gives you:**
- ✅ Clean 100-sample baselines (all seed 2025)
- ✅ Clean 1000-sample baselines (all seed 2026)
- ✅ Can compare scaling effects (100 vs 1000)
- ✅ No risk of accidentally resuming/overwriting

---

## Quick Reference

### Check Existing Checkpoints

```bash
# In Colab
!ls -la grpo-start-ckpts/

# Should show:
# qwen2-5-3b-prm-large-train-v2-2025/           (GRPO 100)
# qwen2-5-3b-prm-large-train-v2-nsr-2025/       (NSR 100)
# qwen2-5-3b-prm-large-train-v2-psr-2025/       (PSR 100, if complete)
# qwen2-5-3b-prm-large-train-v2-w-reinforce-2025/ (W-REINFORCE 100, if complete)
```

### Delete a Checkpoint (If Needed)

```bash
# CAREFUL! This permanently deletes the checkpoint
!rm -rf grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-2025/
```

### Check if Training Will Resume

```bash
# Before running, check if checkpoint exists
!ls grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-2026/

# If it shows checkpoints → Training will RESUME
# If it shows "No such file" → Training will start FRESH
```

---

## Files Updated

I've already updated these files to use `--seed 2026` for 1000-sample runs:

✅ `RUN_GRPO_1000.md` - Updated to seed 2026
✅ `GRPO_1000_COMMAND.txt` - Updated to seed 2026
✅ Added warning about checkpoint behavior

---

## Bottom Line

**Always check what seed you used before!**

- 100-sample runs → seed 2025
- 1000-sample runs → seed 2026

This keeps everything organized and prevents accidental checkpoint conflicts.

**Ready to run GRPO 1000 with seed 2026!** 🚀
