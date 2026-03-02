# NSR Training - Quick Reference Card

**Keep this handy while running experiments!**

---

## 🎯 Current Prompt (Chart-RVR)

**File:** `prompts.py` line 111 (template #4)
**Selection:** `main.py` line 522: `blocks = 4`

**Expected output format:**
```xml
<think>
<type>bar</type>
<table>{"columns": [...], "rows": [...]}</table>
<step-1>: ...
<step-2>: ...
</think>
<answer>42</answer>
```

**Rewards evaluated:**
1. format_reward (2.0) - Correct format
2. chart_type_reward (1.0) - Chart type correct
3. table_style_reward (~1.25) - Table JSON correct
4. accuracy_reward (1.0) - Answer correct
5. length_think_reward (~2.5) - Reasoning length/steps
6. num_token_reward (2.0) - All tags present
7. process_style_reward (~1.0) - Reasoning similarity
**Total max: ~10.75**

---

## 🚀 POC Test Commands (100 samples, ~30 min each)

```bash
# NSR (RECOMMENDED FOR FIRST TEST)
accelerate launch --config_file=deepspeed_zero3.yaml main.py \
  --mode nsr \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025 \
  --subset-size 100 \
  --num-epochs 1 \
  --num-generations 4 \
  --batch-size 2 \
  --disable-gradient-checkpointing \
  2>&1 | tee poc_nsr.log

# Check logs
grep "NSR \[Step" poc_nsr.log | head -10

# Monitor convergence
python monitor_training.py \
  --checkpoint-dir grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-nsr-2025
```

---

## 📊 What to Check During Training

**✅ Good signs:**
```
NSR [Step 0]: 2/4 negative samples | Avg reward: -2.145  ← Has negative samples
NSR [Step 50]: 2/4 negative samples | Avg reward: -0.945  ← Reward increasing (less negative)
Training progress: 100% 50/50 [30:00<00:00]  ← Completes
✓ Model saved to grpo-start-ckpts/...  ← Checkpoint saved
```

**⚠️ Warning signs:**
```
NSR [Step 0]: 0/4 negative samples  ← NO NEGATIVES! Increase --num-generations 8
Loss: 3.2 → 3.2 → 3.2  ← NOT DECREASING! Check learning rate
ModuleNotFoundError: trainers  ← Missing __init__.py! Run: touch trainers/__init__.py
```

---

## 📁 Files to Check After Training

```bash
# Checkpoints exist?
ls grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-nsr-2025/

# Expected:
# checkpoint-50/
# checkpoint-100/
# logs/
# trainer_state.json
# adapter_model.safetensors

# Logs exist?
ls grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-nsr-2025/logs/

# Expected:
# events.out.tfevents.*
# debug.log

# Wandb run?
ls wandb/

# Expected:
# run-YYYYMMDD_HHMMSS-nsr/
```

---

## 🔧 Common Quick Fixes

| Issue | Quick Fix |
|-------|-----------|
| **No negative samples** | `--num-generations 8` |
| **Import error** | `touch trainers/__init__.py evaluation/__init__.py` |
| **OOM** | `--batch-size 1` or enable gradient checkpointing |
| **Loss not decreasing** | Check logs, may need more steps (100 is small) |
| **No checkpoints** | Check disk space: `df -h` |
| **Wandb not logging** | `wandb login` |

---

## 📊 Monitoring Commands

```bash
# View live logs
tail -f grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-nsr-2025/logs/debug.log

# Check reward filtering
grep "NSR \[Step" poc_nsr.log

# Generate convergence plots
python monitor_training.py --checkpoint-dir <path>

# Compare all modes
python monitor_training.py --compare \
  --checkpoints <grpo-path> <psr-path> <nsr-path> <w-reinforce-path> \
  --labels GRPO PSR NSR W-REINFORCE
```

---

## 🎓 Training Modes Quick Reference

| Mode | Trains on | Expected Behavior | Use --num-generations |
|------|-----------|-------------------|----------------------|
| **GRPO** | Both +/- | Baseline | 2 |
| **PSR** | Only + | Best Pass@1, worse Pass@256 | 2 |
| **NSR** | Only - | Same Pass@1, best Pass@256 | **4+** |
| **W-REINFORCE** | λ·+ + - | Best overall | **4+** |

---

## ⏱️ Time & Cost Estimates (Colab Pro A100)

| Config | Samples | Time | Cost |
|--------|---------|------|------|
| **POC** | 100 | 30 min | $0.10 |
| **Quick** | 1K | 2 hours | $0.60 |
| **Fast** | 10K | 20 hours | $3.50 |
| **Full** | 34K | 3 days | $10.00 |

---

## 📍 Key File Locations

| What | Where |
|------|-------|
| **Prompt template** | `prompts.py` line 111 |
| **Template selection** | `main.py` line 522 |
| **Reward filtering** | `main.py` lines 732-793 |
| **Logging config** | `main.py` lines 677-701 |
| **Wandb init** | `main.py` lines 511-531 |
| **PSR trainer** | `trainers/psr_trainer.py` |
| **NSR trainer** | `trainers/nsr_trainer.py` |
| **W-REINFORCE trainer** | `trainers/weighted_reinforce_trainer.py` |
| **Pass@k eval** | `evaluation/pass_at_k.py` |
| **Plotting** | `evaluation/plotting.py` |
| **Monitoring** | `monitor_training.py` |

---

## 🆘 Emergency Contacts (Documentation)

| Issue | See |
|-------|-----|
| **How to run POC?** | `POC_TEST_100_SAMPLES.md` |
| **Logging not working?** | `LOGGING_MONITORING_GUIDE.md` |
| **Colab setup?** | `COLAB_NSR_GUIDE.md` |
| **What changed?** | `FINAL_INTEGRATION_SUMMARY.md` |
| **Quick commands?** | `QUICK_START_NSR.md` |
| **Technical details?** | `NSR_IMPLEMENTATION_GUIDE.md` |

---

## ✅ Pre-Flight Checklist

Before starting POC:
- [ ] `touch trainers/__init__.py`
- [ ] `touch evaluation/__init__.py`
- [ ] `wandb login` (optional)
- [ ] Check GPU: `nvidia-smi` (should show A100)
- [ ] Check disk space: `df -h` (need ~5GB)

---

## 🎯 Success Criteria for POC

POC passes if ALL true:
1. ✅ Training completes without errors
2. ✅ Checkpoint saved to `grpo-start-ckpts/.../`
3. ✅ Logs show reward filtering (e.g., "NSR: X/4 negative")
4. ✅ Loss decreases (>30% reduction)
5. ✅ `monitor_training.py` generates plots

If all pass → Scale to 10K
If any fail → Debug before scaling

---

## 📝 Quick Copy-Paste

**Setup:**
```bash
touch trainers/__init__.py evaluation/__init__.py
```

**Run POC:**
```bash
accelerate launch --config_file=deepspeed_zero3.yaml main.py --mode nsr --vlm-name qwen2-5-3b --dataset-name evochart --seed 2025 --subset-size 100 --num-epochs 1 --num-generations 4 --batch-size 2 --disable-gradient-checkpointing 2>&1 | tee poc_nsr.log
```

**Check results:**
```bash
grep "NSR \[Step" poc_nsr.log
python monitor_training.py --checkpoint-dir grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-nsr-2025
```

---

**Everything you need on one page!**
