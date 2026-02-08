# Quick Fix for Current Error

## Your Error:
```
ModuleNotFoundError: No module named 'undecorated'
```

## Root Cause:
You didn't install from `requirements.txt`. The package `undecorated` is required but wasn't installed.

---

## Fix (Run This Now):

```bash
# In your Colab cell:
!pip install -r requirements.txt
```

This will install ALL 224 dependencies including:
- `undecorated==0.3.0` (the missing package)
- All other required packages

**Time:** ~10-15 minutes

---

## After Installation, Re-run Training:

```bash
!accelerate launch --config_file=deepspeed_zero3.yaml main.py \
  --mode grpo \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025
```

---

## About `--dataset-name evochart` (Your Question):

### This is confusing in the original codebase!

**What it DOESN'T mean:**
- ❌ "Train on EvoChart dataset"

**What it ACTUALLY means:**
- ✅ "Use EvoChart as the validation dataset during training"

**The flow:**
1. **Training data**: Always from `sanchit97/chart-rvr-grpo-train` (34K samples from ChartQA/PlotQA/ChartFC)
2. **Validation data**: From EvoChart (200 samples, OOD dataset)
3. Every 500 training steps, model is evaluated on those 200 EvoChart samples
4. This helps monitor OOD performance during training

**Why use EvoChart for validation?**
- It's an out-of-distribution (OOD) dataset
- Different visual style from training data
- Good indicator of generalization

**Could you use a different validation dataset?**
Yes! You could change it to:
- `chartqa-src` (in-distribution)
- `chartfc` (in-distribution)
- `chartqapro` (OOD)
- etc.

But **evochart is recommended** because it's a good OOD benchmark.

---

## Summary:

| Argument | What It Controls | Value |
|----------|-----------------|-------|
| `--mode grpo` | Training mode | GRPO (vs SFT/DPO/eval) |
| `--vlm-name qwen2-5-3b` | Base model | Qwen2.5-VL-3B |
| `--dataset-name evochart` | **Validation dataset** | EvoChart (200 samples) |
| `--seed 2025` | Random seed | 2025 (for reproducibility) |

**Training dataset** is NOT controlled by command-line args - it's hardcoded to load from HuggingFace in the modified `main.py`.

---

## Verification After Fix:

```bash
# Should show no errors
!python -c "from undecorated import undecorated; print('✓ undecorated installed')"

# Should show all imports work
!python -c "from trl import GRPOTrainer; import flash_attn; print('✓ All dependencies OK')"
```

---

## If `pip install -r requirements.txt` Fails:

Some packages might conflict. Use this minimal install instead:

```bash
# Core dependencies only
!pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
!pip install flash-attn==2.7.4.post1 --no-build-isolation
!pip install trl==0.12.0 transformers==4.53.1 datasets==3.6.0 accelerate==1.7.0 deepspeed==0.15.3 peft==0.15.2
!pip install qwen-vl-utils wandb sentence-transformers sacrebleu scipy scikit-image Pillow undecorated
```

Then try training again.

---

**After this fix, you should be able to start training successfully!**
