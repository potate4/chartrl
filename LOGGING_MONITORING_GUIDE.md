## 📊 Logging, Checkpointing & Monitoring Guide

**Complete guide for tracking training progress, resuming across machines, and monitoring convergence.**

---

## 🎯 What's Saved During Training

### Automatic Checkpointing

Training now saves checkpoints every 50 steps:

```
grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-nsr-2025/
├── checkpoint-50/          # Checkpoint at step 50
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   ├── optimizer.pt
│   ├── scheduler.pt
│   ├── rng_state.pth
│   └── trainer_state.json
├── checkpoint-100/         # Checkpoint at step 100
├── checkpoint-150/         # Checkpoint at step 150
├── logs/                   # Training logs
│   ├── events.out.tfevents.*  # Tensorboard logs
│   └── debug.log
├── trainer_state.json      # Final state
├── training_args.bin       # Training configuration
├── adapter_model.safetensors  # Final model
└── adapter_config.json     # Final config
```

**Key features:**
- ✅ Saves every 50 steps (configurable)
- ✅ Keeps only last 3 checkpoints (saves disk space)
- ✅ Auto-resumes if training interrupted
- ✅ All logs in checkpoint directory
- ✅ Portable across machines

---

## 📝 Enhanced Logging

### Console Logging (Every Step)

```
PSR [Step 0]: 3/4 positive samples | Avg reward: 5.234
NSR [Step 0]: 2/4 negative samples | Avg reward: -2.145
W-REINFORCE [Step 0]: 2 pos (λ=0.1) + 2 neg (λ=1.0) | Avg reward: 3.421

Format rewards: [2.0, 0.0, 2.0, 0.0]
Rewards Accuracy: [1.0, 0.0, 1.0, 1.0]
Length Rewards: [1.5, 0.8, 1.2, 0.0]
...

Training progress: 10% 50/500 [15:30<2:18:45, 18.47s/it]
```

### Wandb Logging (Every 10 Steps)

Automatically logs to Weights & Biases:

**For PSR mode:**
- `psr/positive_samples` - Count of positive samples
- `psr/positive_ratio` - Ratio of positive samples
- `psr/avg_reward` - Average reward

**For NSR mode:**
- `nsr/negative_samples` - Count of negative samples
- `nsr/negative_ratio` - Ratio of negative samples
- `nsr/avg_reward` - Average reward

**For W-REINFORCE mode:**
- `w_reinforce/positive_samples` - Count of positive samples
- `w_reinforce/negative_samples` - Count of negative samples
- `w_reinforce/pos_neg_ratio` - Ratio of positive to negative
- `w_reinforce/avg_reward` - Average reward

**Standard metrics (all modes):**
- `train/loss` - Training loss
- `train/learning_rate` - Current learning rate
- `train/grad_norm` - Gradient norm
- `train/epoch` - Current epoch

---

## 🔄 Resuming Training Across Machines

### Scenario: Training Interrupted on Colab

**Step 1: Save checkpoint to Google Drive** (automated)

Checkpoints are automatically saved every 50 steps to:
```
grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-nsr-2025/checkpoint-XXX/
```

**Step 2: Download checkpoint**

```bash
# On Colab
!zip -r checkpoint.zip grpo-start-ckpts/
from google.colab import files
files.download('checkpoint.zip')
```

**Step 3: Resume on different machine**

```bash
# On new machine (Kaggle, local, etc.)
unzip checkpoint.zip

# Re-run EXACT same command - will auto-resume!
accelerate launch --config_file=deepspeed_zero3.yaml main.py \
  --mode nsr \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025 \
  --subset-size 10000 \
  --num-epochs 1 \
  --num-generations 4
```

**Auto-resume enabled:**
```python
resume_from_checkpoint=True  # Already set in training_args
```

Training will automatically detect the latest checkpoint and resume!

---

## 📊 Monitoring Training Convergence

### Method 1: Weights & Biases Dashboard

**Access W&B:**
1. Go to https://wandb.ai/chartrl/chartrl-nsr
2. Find your run: `nsr-qwen2-5-3b-evochart-10000-2025`
3. View real-time charts:
   - Training loss
   - Learning rate
   - Gradient norm
   - NSR-specific metrics

**Example dashboard view:**
```
Loss: 3.2 → 1.5 (↓53%)
LR: 1e-5 → 9.5e-6
Grad Norm: 2.3 → 0.8
NSR negative ratio: 0.48 → 0.52
```

---

### Method 2: Monitor Training Script

**Generate convergence plots:**

```bash
# Single run
python monitor_training.py \
  --checkpoint-dir grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-nsr-2025 \
  --output-dir nsr_plots

# Compare multiple runs
python monitor_training.py \
  --compare \
  --checkpoints grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-grpo-2025 \
              grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-psr-2025 \
              grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-nsr-2025 \
              grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-w-reinforce-2025 \
  --labels GRPO PSR NSR W-REINFORCE \
  --output-dir comparison_plots
```

**Generates 4 plots:**

1. **training_loss.png** - Loss curves over time
   - Should decrease smoothly
   - Flat = converged
   - Oscillating = not converged

2. **learning_rate.png** - LR schedule
   - Should decrease gradually (if using scheduler)

3. **gradient_norm.png** - Gradient magnitude
   - Should decrease and stabilize
   - High spikes = unstable training

4. **rewards.png** - Average reward progression
   - Should increase over time
   - NSR rewards may be negative (correct!)

**Plus NSR-specific plots:**
5. **nsr_metrics.png** - 4 subplots:
   - PSR positive sample ratio
   - NSR negative sample ratio
   - W-REINFORCE pos/neg ratio
   - Average rewards comparison

---

### Method 3: Check Logs Manually

```bash
# View latest logs
tail -f grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-nsr-2025/logs/debug.log

# Search for specific metrics
grep "NSR \[Step" grpo-start-ckpts/*/logs/debug.log
grep "Avg reward" grpo-start-ckpts/*/logs/debug.log
```

---

## 🔍 Convergence Indicators

### ✅ Training is Converging Well

**Signs:**
1. **Loss decreasing smoothly**
   ```
   Step 0: Loss = 3.2
   Step 100: Loss = 2.1
   Step 200: Loss = 1.6
   Step 300: Loss = 1.5
   ```

2. **Gradient norm stabilizing**
   ```
   Step 0: Grad norm = 2.5
   Step 100: Grad norm = 1.2
   Step 200: Grad norm = 0.8
   Step 300: Grad norm = 0.7  ← Stable
   ```

3. **Reward increasing**
   ```
   Step 0: Avg reward = 5.2
   Step 100: Avg reward = 6.8
   Step 200: Avg reward = 7.5
   Step 300: Avg reward = 7.8  ← Increasing
   ```

4. **Sample ratios stable** (for NSR/PSR/W-REINFORCE)
   ```
   NSR negative ratio: 0.48 → 0.52 → 0.51 → 0.50  ← Stable
   ```

---

### ⚠️ Training May Have Issues

**Warning signs:**

1. **Loss not decreasing**
   ```
   Step 0: Loss = 3.2
   Step 100: Loss = 3.1
   Step 200: Loss = 3.0  ← Too slow
   ```
   **Fix:** Increase learning rate or check reward functions

2. **Loss oscillating wildly**
   ```
   Step 100: Loss = 2.0
   Step 101: Loss = 5.3
   Step 102: Loss = 1.8
   Step 103: Loss = 4.2  ← Unstable
   ```
   **Fix:** Reduce learning rate, increase batch size, or enable gradient clipping

3. **Gradient norm exploding**
   ```
   Step 100: Grad norm = 1.2
   Step 101: Grad norm = 15.8  ← Explosion!
   ```
   **Fix:** Enable gradient clipping, reduce learning rate

4. **No negative samples (NSR mode)**
   ```
   NSR [Step 50]: 0/4 negative samples  ← Problem!
   ```
   **Fix:** Increase `--num-generations 8` or lower `--reward-threshold 0.3`

5. **All samples filtered out**
   ```
   PSR [Step 50]: 0/4 positive samples  ← Problem!
   ```
   **Fix:** Lower `--reward-threshold` or check reward computation

---

## 📁 Complete File Structure After Training

```
chartrl/
├── grpo-start-ckpts/
│   ├── qwen2-5-3b-prm-large-train-v2-2025/          # GRPO
│   │   ├── checkpoint-50/
│   │   ├── checkpoint-100/
│   │   ├── checkpoint-150/
│   │   ├── logs/
│   │   │   ├── events.out.tfevents.*
│   │   │   └── debug.log
│   │   ├── trainer_state.json
│   │   ├── training_args.bin
│   │   ├── adapter_model.safetensors
│   │   └── adapter_config.json
│   ├── qwen2-5-3b-prm-large-train-v2-psr-2025/      # PSR
│   ├── qwen2-5-3b-prm-large-train-v2-nsr-2025/      # NSR
│   └── qwen2-5-3b-prm-large-train-v2-w-reinforce-2025/  # W-REINFORCE
├── wandb/                                            # W&B logs
│   ├── run-20251227_123456-nsr/
│   ├── run-20251227_123457-psr/
│   ├── run-20251227_123458-grpo/
│   └── run-20251227_123459-w-reinforce/
├── training_plots/                                   # Monitoring plots
│   ├── training_loss.png
│   ├── learning_rate.png
│   ├── gradient_norm.png
│   ├── rewards.png
│   └── nsr_metrics.png
└── comparison_plots/                                 # Comparison plots
    ├── training_loss.png
    ├── learning_rate.png
    └── nsr_metrics.png
```

---

## 🚀 POC Test Logging Example

**100-sample POC test:**

```bash
# Run POC
accelerate launch --config_file=deepspeed_zero3.yaml main.py \
  --mode nsr \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025 \
  --subset-size 100 \
  --num-epochs 1 \
  --num-generations 4 2>&1 | tee poc_nsr.log

# After training, check convergence
python monitor_training.py \
  --checkpoint-dir grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-nsr-2025 \
  --output-dir poc_plots

# View summary
cat poc_plots/summary.txt
```

**Expected POC output:**
```
NSR TRAINING CONFIGURATION:
  Training mode: nsr
  Training samples: 100
  Epochs: 1
  Reward threshold: 0.5

✓ NSR reward filtering enabled
  → Training only on NEGATIVE samples

Training:
NSR [Step 0]: 2/4 negative samples | Avg reward: -2.145
NSR [Step 10]: 3/4 negative samples | Avg reward: -1.892
NSR [Step 20]: 2/4 negative samples | Avg reward: -1.654
...
NSR [Step 50]: 2/4 negative samples | Avg reward: -0.845

Training progress: 100% 50/50 [30:00<00:00, 36.00s/it]

✓ Training completed successfully
✓ Model saved to grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-nsr-2025/

TRAINING SUMMARY:
  Total steps: 50
  Final loss: 1.523
  Loss reduction: 52.1% (3.165 → 1.523)
  Training time: 0.50 hours
  ✓ Converged (loss std/mean = 0.0082)
```

---

## 📊 Visualizing Convergence

### Training Loss Plot

**Good convergence:**
```
Loss
3.0 ┤╮
2.5 ┤ ╰╮
2.0 ┤  ╰╮
1.5 ┤   ╰─────────  ← Flat = converged
1.0 ┤
    └─────────────────> Steps
    0  100  200  300
```

**Not converged:**
```
Loss
3.0 ┤╮
2.5 ┤ ╰╮
2.0 ┤  ╰╮
1.5 ┤   ╰╮           ← Still decreasing
1.0 ┤    ╰
    └─────────────────> Steps
    0  100  200  300
```

**Unstable:**
```
Loss
3.0 ┤ ╮╭╮╭╮
2.5 ┤╮╯╰╯╰╯╮           ← Oscillating = bad!
2.0 ┤╯    ╰╮
1.5 ┤      ╰
1.0 ┤
    └─────────────────> Steps
    0  100  200  300
```

---

## 💾 Backing Up Training Data

### For Long Training Runs

**Recommended: Auto-sync to cloud**

```python
# Add to training script (Colab)
import time
import shutil
import threading

def backup_checkpoints():
    while True:
        time.sleep(3600)  # Every hour
        print("Backing up checkpoints...")
        shutil.copytree(
            'grpo-start-ckpts',
            '/content/drive/MyDrive/chartrl_backup/grpo-start-ckpts',
            dirs_exist_ok=True
        )
        print("✓ Backup complete")

# Start backup thread
backup_thread = threading.Thread(target=backup_checkpoints, daemon=True)
backup_thread.start()
```

### Manual Backup

```bash
# Compress and download
tar -czf checkpoints.tar.gz grpo-start-ckpts/
# Download via files.download() in Colab

# Upload to new machine
tar -xzf checkpoints.tar.gz
# Resume training
```

---

## 🔧 Troubleshooting Logs

### Issue: No logs appearing

**Check:**
```bash
# Verify logging directory exists
ls -la grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-nsr-2025/logs/

# Check wandb initialization
grep "wandb" poc_nsr.log
```

**Fix:**
```python
# In main.py, ensure wandb is initialized
import wandb
wandb.init(project="chartrl-nsr", ...)
```

---

### Issue: Checkpoint not resuming

**Check:**
```bash
# Verify checkpoint directory exists
ls -la grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-nsr-2025/checkpoint-*/

# Check for trainer_state.json
cat grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-nsr-2025/checkpoint-150/trainer_state.json
```

**Fix:**
```python
# Ensure output_dir matches
output_dir = f"grpo-start-ckpts/{args.vlm_name}-prm-large-train-v2-{args.mode}-{seed}"
```

---

## Summary

**Logging features:**
- ✅ Console logging every step
- ✅ Wandb logging every 10 steps
- ✅ Checkpoint saving every 50 steps
- ✅ Auto-resume enabled
- ✅ All logs in checkpoint directory

**Monitoring:**
- ✅ W&B dashboard (real-time)
- ✅ `monitor_training.py` (generate plots)
- ✅ Manual log inspection

**Convergence indicators:**
- ✅ Loss decreasing smoothly
- ✅ Gradient norm stabilizing
- ✅ Reward increasing
- ✅ Sample ratios stable

**Everything is saved and portable across machines!**
