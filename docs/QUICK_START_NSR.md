# NSR Training - Quick Start Guide

**TL;DR:** Copy-paste commands for PSR, NSR, and W-REINFORCE training modes.

Based on: "Decomposed Reinforcement Learning from Verifiable Feedback (RLVR)" paper

---

## What is NSR?

**NSR (Negative Sample Reinforcement)** trains only on incorrect responses, teaching the model what NOT to do.

**Key Finding:** NSR-only training matches or beats full GRPO while preserving diversity!

| Method | What it trains on | Expected Result |
|--------|------------------|-----------------|
| **GRPO** | Both correct & incorrect | Baseline performance |
| **PSR** | Only correct responses | Best Pass@1, worse Pass@256 |
| **NSR** | Only incorrect responses | Same Pass@1 as GRPO, best Pass@256 |
| **W-REINFORCE** | 10% correct + 100% incorrect | Best overall across all Pass@k |

---

## 🚀 Quick Commands

### 1. GRPO Baseline (Standard)

```bash
# Standard GRPO training
accelerate launch --config_file=deepspeed_zero3.yaml main.py \
  --mode grpo \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025 \
  --subset-size 10000 \
  --num-epochs 1 \
  --num-generations 2
```

**Time:** ~21 hours on A100 40GB

---

### 2. PSR-Only Training (Positive Samples Only)

```bash
# Train only on correct responses
accelerate launch --config_file=deepspeed_zero3.yaml main.py \
  --mode psr \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025 \
  --subset-size 10000 \
  --num-epochs 1 \
  --num-generations 2 \
  --reward-threshold 0.5
```

**What it does:**
- Filters to keep only high-reward samples (reward ≥ 0.5)
- Maximizes likelihood of correct responses
- Reduces diversity, sharpens distribution

**Expected results:**
- ✓ Best Pass@1 (best single response)
- ✗ Lower Pass@k at large k (reduced diversity)

**Checkpoint saved to:** `grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-psr-2025/`

---

### 3. NSR-Only Training (Negative Samples Only) ⭐

```bash
# Train only on incorrect responses (RECOMMENDED FOR DIVERSITY)
accelerate launch --config_file=deepspeed_zero3.yaml main.py \
  --mode nsr \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025 \
  --subset-size 10000 \
  --num-epochs 1 \
  --num-generations 4 \
  --reward-threshold 0.5
```

**What it does:**
- Filters to keep only low-reward samples (reward < 0.5)
- Minimizes likelihood of incorrect responses
- Redistributes probability mass away from known errors

**Expected results:**
- ✓ Similar Pass@1 as GRPO (surprising!)
- ✓ Best Pass@256 (maximum diversity)
- ✓ Preserves multiple valid reasoning paths

**Important:** Use `--num-generations 4` or higher to get sufficient negative samples!

**Checkpoint saved to:** `grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-nsr-2025/`

---

### 4. Weighted-REINFORCE Training (BEST OVERALL) 🏆

```bash
# Combine PSR and NSR with 10% PSR weight (paper recommendation)
accelerate launch --config_file=deepspeed_zero3.yaml main.py \
  --mode w-reinforce \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025 \
  --subset-size 10000 \
  --num-epochs 1 \
  --num-generations 4 \
  --lambda-psr 0.1 \
  --reward-threshold 0.5
```

**What it does:**
- Applies λ=0.1 weight to correct samples (PSR)
- Applies λ=1.0 weight to incorrect samples (NSR)
- Objective: 0.1·L_PSR + L_NSR

**Expected results:**
- ✓ Best Pass@1 (from 10% PSR)
- ✓ Best Pass@256 (from 100% NSR)
- ✓ Best overall performance across all k

**Lambda tuning:** Try λ ∈ {0.05, 0.1, 0.15, 0.2} for hyperparameter search.

**Checkpoint saved to:** `grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-w-reinforce-2025/`

---

## 📊 Evaluation: Pass@k Metrics

### Step 1: Generate Samples

```bash
# Generate 256 samples per problem for Pass@k evaluation
python generate_samples.py \
  --checkpoint-path grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-nsr-2025 \
  --dataset-name evochart \
  --output-path samples_nsr.json \
  --num-samples 256
```

**Time:** ~2-4 hours for 500 problems

### Step 2: Compute Pass@k

```bash
# Evaluate Pass@k from generated samples
python evaluation/pass_at_k.py \
  --samples-path samples_nsr.json \
  --output-path results_nsr.json \
  --method-name NSR
```

**Output:**
```
Pass@k Results:
  Pass@1: 65.14%
  Pass@2: 71.23%
  Pass@4: 75.89%
  Pass@8: 79.12%
  Pass@16: 81.45%
  Pass@32: 83.21%
  Pass@64: 84.67%
  Pass@128: 85.89%
  Pass@256: 86.73%
```

### Step 3: Plot Comparison

```bash
# Plot Pass@k curves for all methods
python evaluation/plotting.py \
  --results results_grpo.json results_psr.json results_nsr.json results_w_reinforce.json \
  --labels "GRPO" "PSR" "NSR" "W-REINFORCE" \
  --output comparison_pass_at_k.png
```

---

## ⏱️ Training Time Estimates

| Configuration | Samples | GPUs | Time | Cost |
|---------------|---------|------|------|------|
| **Fast (10K)** | 10,000 | 1x A100 | ~21 hours | Colab Pro |
| **Fast (10K)** | 10,000 | 2x T4 | ~40-50 hours | Kaggle FREE |
| **Moderate (34K)** | 34,194 | 1x A100 | ~3 days | Colab Pro |
| **Moderate (34K)** | 34,194 | 2x T4 | ~7-10 days | Kaggle FREE |

**Recommendation:** Start with 10K samples to validate, then scale to 34K.

---

## 🔧 Important Parameters

### --mode

Training mode selection:
- `grpo`: Standard GRPO (baseline)
- `psr`: PSR-only (positive samples)
- `nsr`: NSR-only (negative samples)
- `w-reinforce`: Weighted-REINFORCE (λ·PSR + NSR)

### --lambda-psr (W-REINFORCE only)

Weight for PSR component (default: 0.1):
- `0.05`: More NSR weight → Better diversity
- `0.1`: Balanced (paper recommendation) → Best overall
- `0.2`: More PSR weight → Better Pass@1

### --reward-threshold

Threshold to separate positive/negative samples (default: 0.5):
- Higher threshold (0.7): Stricter positive filtering
- Lower threshold (0.3): More lenient

### --num-generations

Number of samples generated per problem:
- **PSR:** Can use 2 (only need correct samples)
- **NSR:** Need ≥4 (need sufficient incorrect samples)
- **W-REINFORCE:** Need ≥4 (need mix of both)

---

## 🎯 Recommended Workflow

### Stage 1: Quick Validation (1 day)

Train all 4 methods on 10K samples:

```bash
# GRPO baseline
accelerate launch --config_file=deepspeed_zero3.yaml main.py --mode grpo --vlm-name qwen2-5-3b --dataset-name evochart --seed 2025 --subset-size 10000 --num-epochs 1 --num-generations 2

# PSR
accelerate launch --config_file=deepspeed_zero3.yaml main.py --mode psr --vlm-name qwen2-5-3b --dataset-name evochart --seed 2025 --subset-size 10000 --num-epochs 1 --num-generations 2

# NSR
accelerate launch --config_file=deepspeed_zero3.yaml main.py --mode nsr --vlm-name qwen2-5-3b --dataset-name evochart --seed 2025 --subset-size 10000 --num-epochs 1 --num-generations 4

# W-REINFORCE
accelerate launch --config_file=deepspeed_zero3.yaml main.py --mode w-reinforce --vlm-name qwen2-5-3b --dataset-name evochart --seed 2025 --subset-size 10000 --num-epochs 1 --num-generations 4 --lambda-psr 0.1
```

**Run in parallel if you have multiple GPUs!**

### Stage 2: Evaluate Pass@k (1 day)

Generate samples and compute Pass@k for all methods:

```bash
# Generate samples for each method
for method in grpo psr nsr w-reinforce; do
  python generate_samples.py \
    --checkpoint-path grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-${method}-2025 \
    --dataset-name evochart \
    --output-path samples_${method}.json \
    --num-samples 256
done

# Evaluate Pass@k for each
for method in grpo psr nsr w-reinforce; do
  python evaluation/pass_at_k.py \
    --samples-path samples_${method}.json \
    --output-path results_${method}.json \
    --method-name ${method}
done

# Plot comparison
python evaluation/plotting.py \
  --results results_grpo.json results_psr.json results_nsr.json results_w_reinforce.json \
  --labels "GRPO" "PSR" "NSR" "W-REINFORCE" \
  --output comparison_pass_at_k.png
```

### Stage 3: Scale Up (3-7 days)

Train best method on full 34K dataset:

```bash
# If W-REINFORCE performed best
accelerate launch --config_file=deepspeed_zero3.yaml main.py \
  --mode w-reinforce \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025 \
  --num-epochs 1 \
  --num-generations 4 \
  --lambda-psr 0.1
```

---

## 📈 Expected Pass@k Results

Based on MATH dataset results from paper (chart reasoning may differ):

| Method | Pass@1 | Pass@8 | Pass@64 | Pass@256 |
|--------|--------|--------|---------|----------|
| **GRPO** | 65% | 72% | 77% | 79% |
| **PSR** | **67%** | 71% | 75% | 76% |
| **NSR** | 65% | 74% | 78% | **81%** |
| **W-REINFORCE** | **68%** | **75%** | **80%** | **82%** |

**Key Observations:**
- PSR: Best at Pass@1, worst at Pass@256
- NSR: Same Pass@1 as GRPO, best at Pass@256
- W-REINFORCE: Best overall

---

## 🆘 Troubleshooting

### Issue: NSR training crashes - "No negative samples found"

**Cause:** Not enough generations to get incorrect samples

**Solution:**
```bash
# Increase num_generations to at least 4
--num-generations 4
```

### Issue: W-REINFORCE underperforming

**Cause:** Lambda may need tuning for chart reasoning

**Solution:** Try different λ values:
```bash
# Try λ ∈ {0.05, 0.1, 0.15, 0.2}
--lambda-psr 0.05   # More diversity
--lambda-psr 0.2    # More accuracy
```

### Issue: All methods have similar Pass@k

**Possible causes:**
1. Test set too easy → Try harder benchmark
2. Not enough diversity → Increase temperature in generation
3. Insufficient training → Train longer

---

## 📚 Documentation

- **NSR_IMPLEMENTATION_GUIDE.md** - Comprehensive technical guide
- **GRPO_TRAINING_GUIDE.md** - Original GRPO guide
- **QUICK_START.md** - GRPO quick reference
- **KAGGLE_2xT4_GUIDE.md** - Free training on Kaggle

---

## 🎓 Key Concepts

### Why NSR Works

Traditional RL (GRPO/PPO) maximizes:
```
L = Σ r·log(π(y|x))  for all samples
```

NSR decomposes this into:
```
L_PSR = Σ r·log(π(y|x))   for r > 0  (correct)
L_NSR = Σ r·log(π(y|x))   for r < 0  (incorrect)
```

**Key Insight:** L_NSR alone preserves diversity by redistributing probability mass away from known errors without forcing it onto a single correct answer.

### When to Use Each Mode

| Use Case | Recommended Mode | Why |
|----------|-----------------|-----|
| **Need best single answer** | PSR | Maximizes Pass@1 |
| **Need diverse reasoning** | NSR | Maximizes Pass@256 |
| **Production deployment** | W-REINFORCE | Best overall balance |
| **Baseline comparison** | GRPO | Standard approach |

---

## 💡 Pro Tips

1. **Always use ≥4 generations for NSR and W-REINFORCE** to get sufficient negative samples

2. **Start with 10K samples** to validate approach before scaling to 34K

3. **Monitor reward distribution** during training:
   - PSR: Should see mostly positive rewards
   - NSR: Should see mostly negative rewards
   - W-REINFORCE: Should see mix of both

4. **Lambda search for W-REINFORCE:** If you have compute, try λ ∈ {0.05, 0.1, 0.15, 0.2}

5. **Temperature for diversity:** Use temperature=1.0 for Pass@k evaluation (more diversity)

---

## 🎯 One-Command Quick Test

Want to quickly test NSR on a small subset?

```bash
# NSR training on 1K samples (2 hours)
accelerate launch --config_file=deepspeed_zero3.yaml main.py \
  --mode nsr \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025 \
  --subset-size 1000 \
  --num-epochs 1 \
  --num-generations 4
```

---

## Summary

**What you implemented:**
- ✅ PSR trainer (positive samples only)
- ✅ NSR trainer (negative samples only)
- ✅ W-REINFORCE trainer (weighted combination)
- ✅ Pass@k evaluation metrics
- ✅ Plotting utilities
- ✅ Modular architecture

**Next steps:**
1. Run fast validation (10K samples, all 4 methods)
2. Evaluate Pass@k curves
3. Scale up best method to full 34K
4. Publish results!

**Expected outcome:** NSR and W-REINFORCE will show better Pass@k diversity than GRPO while maintaining accuracy.
