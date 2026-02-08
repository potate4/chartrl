# NSR POC Test - 100 Samples

**Proof of Concept test to verify NSR integration works seamlessly before full training.**

---

## 🎯 Purpose

Test all 4 training modes (GRPO, PSR, NSR, W-REINFORCE) on 100 samples to ensure:
- ✅ NSR integration doesn't break existing GRPO
- ✅ Reward filtering works correctly
- ✅ Checkpointing and logging work
- ✅ Training converges properly
- ✅ All modes complete without errors

**Time:** ~30 minutes per mode (~2 hours total)
**Cost:** ~$0.30 (Colab Pro)

---

## 🚀 Quick POC Commands

### Test 1: GRPO Baseline (Verify existing code still works)

```bash
accelerate launch --config_file=deepspeed_zero3.yaml main.py \
  --mode grpo \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025 \
  --subset-size 100 \
  --num-epochs 1 \
  --num-generations 2 \
  --batch-size 2 \
  --disable-gradient-checkpointing
```

**Expected:**
- ✅ Training starts without errors
- ✅ Checkpoint saved to `grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-2025/`
- ✅ Logs show reward computation
- ✅ Training completes in ~30 minutes

**Success criteria:**
```
Format rewards: [2.0, 0.0, 2.0, 0.0]
Rewards Accuracy: [1.0, 0.0, 1.0, 1.0]
...
Training progress: 100% 50/50 [30:00<00:00]
✓ Model saved to grpo-start-ckpts/...
```

---

### Test 2: PSR Mode (Verify positive sample filtering)

```bash
accelerate launch --config_file=deepspeed_zero3.yaml main.py \
  --mode psr \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025 \
  --subset-size 100 \
  --num-epochs 1 \
  --num-generations 2 \
  --batch-size 2 \
  --reward-threshold 0.5 \
  --disable-gradient-checkpointing
```

**Expected:**
- ✅ PSR reward filtering logs appear
- ✅ Only positive samples get non-zero rewards
- ✅ Checkpoint saved to `grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-psr-2025/`

**Success criteria:**
```
✓ PSR TRAINING CONFIGURATION:
  Training mode: psr
  Reward threshold: 0.5

✓ PSR reward filtering enabled
  → Training only on POSITIVE samples (correct responses)

PSR: Filtered 3/4 positive samples  ← Should see this in logs
```

---

### Test 3: NSR Mode (Verify negative sample filtering)

```bash
accelerate launch --config_file=deepspeed_zero3.yaml main.py \
  --mode nsr \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025 \
  --subset-size 100 \
  --num-epochs 1 \
  --num-generations 4 \
  --batch-size 2 \
  --reward-threshold 0.5 \
  --disable-gradient-checkpointing
```

**CRITICAL:** NSR needs `--num-generations 4`!

**Expected:**
- ✅ NSR reward filtering logs appear
- ✅ Only negative samples get non-zero (inverted) rewards
- ✅ Checkpoint saved to `grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-nsr-2025/`

**Success criteria:**
```
✓ NSR TRAINING CONFIGURATION:
  Training mode: nsr
  Reward threshold: 0.5

✓ NSR reward filtering enabled
  → Training only on NEGATIVE samples (incorrect responses)

NSR: Filtered 3/4 negative samples  ← Should see this in logs
```

---

### Test 4: W-REINFORCE Mode (Verify weighted filtering)

```bash
accelerate launch --config_file=deepspeed_zero3.yaml main.py \
  --mode w-reinforce \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025 \
  --subset-size 100 \
  --num-epochs 1 \
  --num-generations 4 \
  --batch-size 2 \
  --lambda-psr 0.1 \
  --reward-threshold 0.5 \
  --disable-gradient-checkpointing
```

**Expected:**
- ✅ W-REINFORCE weighted filtering logs appear
- ✅ Positive samples weighted by λ=0.1, negative by 1.0
- ✅ Checkpoint saved to `grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-w-reinforce-2025/`

**Success criteria:**
```
✓ W-REINFORCE TRAINING CONFIGURATION:
  Training mode: w-reinforce
  Lambda PSR: 0.1
  Reward threshold: 0.5

✓ W-REINFORCE reward filtering enabled
  → Weighted training: 0.1·PSR + NSR

W-REINFORCE: 2 positive (λ=0.1), 2 negative (λ=1.0)  ← Should see this
```

---

## 📊 Verification Checklist

After running all 4 tests, verify:

### Files Created:
```bash
# Check checkpoints exist
ls -la grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-2025/          # GRPO
ls -la grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-psr-2025/      # PSR
ls -la grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-nsr-2025/      # NSR
ls -la grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-w-reinforce-2025/  # W-REINFORCE

# Each should contain:
# - adapter_config.json
# - adapter_model.safetensors
# - checkpoint-*/
# - trainer_state.json
# - training_args.bin
```

### Logs Created:
```bash
# Check W&B logs
ls -la wandb/

# Should show 4 runs:
# - run-YYYYMMDD_HHMMSS-grpo
# - run-YYYYMMDD_HHMMSS-psr
# - run-YYYYMMDD_HHMMSS-nsr
# - run-YYYYMMDD_HHMMSS-w-reinforce
```

### Training Metrics:
```bash
# Check training logs show rewards
grep "Format rewards" <log_file>
grep "Rewards Accuracy" <log_file>
grep "PSR: Filtered" <log_file>      # For PSR mode
grep "NSR: Filtered" <log_file>      # For NSR mode
grep "W-REINFORCE:" <log_file>       # For W-REINFORCE mode
```

---

## 🐛 Common Issues & Fixes

### Issue 1: NSR shows "Filtered 0/4 negative samples"

**Cause:** Model generating mostly correct samples (good problem!)

**Fix:**
```bash
# Option 1: Increase generations
--num-generations 8

# Option 2: Lower threshold
--reward-threshold 0.3

# Option 3: Use earlier checkpoint (less trained = more errors)
```

---

### Issue 2: Import error for trainers

**Error:**
```
ModuleNotFoundError: No module named 'trainers.psr_trainer'
```

**Fix:**
```bash
# Create __init__.py files
touch trainers/__init__.py
touch evaluation/__init__.py

# Or add to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

---

### Issue 3: Reward filtering not appearing in logs

**Cause:** Logging level too high

**Fix:**
```python
# In main.py, add before training:
import logging
logging.basicConfig(level=logging.DEBUG)  # Change from INFO to DEBUG
```

---

### Issue 4: Checkpoint not saving

**Cause:** Output directory permissions or disk space

**Fix:**
```bash
# Check disk space
df -h

# Check write permissions
ls -la grpo-start-ckpts/

# Create directory manually if needed
mkdir -p grpo-start-ckpts
chmod 777 grpo-start-ckpts
```

---

## 📈 Expected Training Curves

After POC test (100 samples), you should see:

**GRPO:**
- Loss: Decreases from ~3.0 to ~1.5
- Reward: Increases from ~5.0 to ~7.0

**PSR:**
- Loss: Decreases faster (only learning from correct samples)
- Reward: Higher early on (only positive rewards)

**NSR:**
- Loss: May increase slightly (minimizing bad samples)
- Reward: Lower (negative rewards dominate)

**W-REINFORCE:**
- Loss: Between GRPO and NSR
- Reward: Balanced (weighted positive + negative)

---

## ✅ Success Criteria

POC is successful if:
1. ✅ All 4 modes complete without errors
2. ✅ Checkpoints saved for all 4 modes
3. ✅ Logs show correct reward filtering
4. ✅ Training loss decreases (convergence)
5. ✅ File structure matches expectations

If all pass → Proceed to 10K training
If any fail → Debug before scaling up

---

## 🔍 Detailed Validation

### Validate Reward Filtering

Run this after training to check reward distribution:

```python
# validate_rewards.py
import json
import re

def validate_mode_rewards(log_file, mode):
    """Check that rewards are filtered correctly"""
    with open(log_file, 'r') as f:
        logs = f.read()

    # Extract reward lines
    reward_lines = re.findall(r'Rewards Accuracy: \[(.*?)\]', logs)

    positive_count = 0
    negative_count = 0
    zero_count = 0

    for line in reward_lines:
        rewards = [float(r) for r in line.split(',')]
        positive_count += sum(1 for r in rewards if r > 0.5)
        negative_count += sum(1 for r in rewards if r < -0.1)
        zero_count += sum(1 for r in rewards if abs(r) < 0.1)

    print(f"\n{mode.upper()} Reward Distribution:")
    print(f"  Positive: {positive_count}")
    print(f"  Negative: {negative_count}")
    print(f"  Zero: {zero_count}")

    # Validate mode-specific expectations
    if mode == 'psr':
        assert negative_count == 0, "PSR should have no negative rewards!"
        print("✓ PSR validation passed")
    elif mode == 'nsr':
        assert positive_count == 0, "NSR should have no positive rewards!"
        print("✓ NSR validation passed")
    elif mode == 'w-reinforce':
        assert positive_count > 0 and negative_count > 0, "W-REINFORCE should have both!"
        print("✓ W-REINFORCE validation passed")
    else:  # grpo
        assert positive_count > 0, "GRPO should have positive rewards!"
        print("✓ GRPO validation passed")

# Run validation
validate_mode_rewards('grpo_log.txt', 'grpo')
validate_mode_rewards('psr_log.txt', 'psr')
validate_mode_rewards('nsr_log.txt', 'nsr')
validate_mode_rewards('w_reinforce_log.txt', 'w-reinforce')
```

---

## 🎯 Next Steps After POC

If POC passes:

1. **Quick Test (1K samples, ~2 hours):**
   ```bash
   --subset-size 1000
   ```

2. **Fast Training (10K samples, ~20 hours):**
   ```bash
   --subset-size 10000
   ```

3. **Full Training (34K samples, ~3 days):**
   ```bash
   # Remove --subset-size flag
   ```

If POC fails:
- Debug specific mode that failed
- Check logs for errors
- Verify file structure
- Re-run POC after fixes

---

## 📝 POC Test Script

Complete script to run all 4 POC tests:

```bash
#!/bin/bash
# poc_test_nsr.sh

echo "=========================================="
echo "NSR POC Test - 100 Samples"
echo "=========================================="

# Array of modes to test
MODES=("grpo" "psr" "nsr" "w-reinforce")

for MODE in "${MODES[@]}"; do
    echo ""
    echo "=========================================="
    echo "Testing: $MODE"
    echo "=========================================="

    # Set generations based on mode
    if [[ "$MODE" == "grpo" ]] || [[ "$MODE" == "psr" ]]; then
        GENERATIONS=2
    else
        GENERATIONS=4
    fi

    # Build command
    CMD="accelerate launch --config_file=deepspeed_zero3.yaml main.py \
      --mode $MODE \
      --vlm-name qwen2-5-3b \
      --dataset-name evochart \
      --seed 2025 \
      --subset-size 100 \
      --num-epochs 1 \
      --num-generations $GENERATIONS \
      --batch-size 2 \
      --disable-gradient-checkpointing"

    # Add mode-specific args
    if [[ "$MODE" == "w-reinforce" ]]; then
        CMD="$CMD --lambda-psr 0.1"
    fi

    if [[ "$MODE" != "grpo" ]]; then
        CMD="$CMD --reward-threshold 0.5"
    fi

    # Run training
    echo "Running: $CMD"
    eval "$CMD" 2>&1 | tee "poc_${MODE}.log"

    # Check exit code
    if [ $? -eq 0 ]; then
        echo "✓ $MODE completed successfully"
    else
        echo "✗ $MODE failed!"
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "All POC tests completed!"
echo "=========================================="
echo ""
echo "Checkpoints saved to:"
ls -d grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-*-2025/

echo ""
echo "Logs saved to:"
ls -1 poc_*.log
```

**Usage:**
```bash
chmod +x poc_test_nsr.sh
./poc_test_nsr.sh
```

---

## Summary

**POC Test:** 100 samples, 4 modes
**Time:** ~2 hours total
**Cost:** ~$0.30 (Colab Pro)

**Success = All pass** → Proceed to 10K training
**Failure = Any fail** → Debug and re-run POC

This ensures NSR integration works seamlessly before investing time/money in full training!
