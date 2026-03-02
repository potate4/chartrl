# How to Visualize Your Training

You have **2 tools** to view your training metrics:

## Option 1: Text View (No Installation Needed) ✅

**Quick, simple, works immediately!**

```bash
# View single run
python view_training_metrics.py grpo.log

# Compare multiple runs
python view_training_metrics.py grpo.log nsr.log
```

**What you get:**
- ✅ Table showing all metrics per step
- ✅ Summary statistics
- ✅ Comparison table (if multiple runs)
- ✅ Works on any system, no dependencies!

**Example output:**
```
Step       Loss   Reward    Acc Format  Chart    GradN         LR   Length
--------------------------------------------------------------------------------
1       -0.0344     6.84   0.0%    1.5 100.0%   0.2761   1.00e-05    257.0
2       -0.0228     5.83  41.7%    0.8  63.9%   0.2099   8.20e-06    369.6
...

[ACCURACY] (Most Important!):
   Initial:       0.0%
   Peak:         45.0%  (step 3)
   Final:        37.5%
   Improvement:  37.5%
```

---

## Option 2: Visual Plots (Requires matplotlib) 📊

**Beautiful publication-quality plots for thesis!**

### Step 1: Install Dependencies

```bash
pip install matplotlib numpy
```

### Step 2: Generate Plots

```bash
# Single run
python plot_training_metrics.py grpo.log

# Compare multiple runs
python plot_training_metrics.py grpo.log nsr.log
```

**What you get:**
- 📊 **Detailed metrics plot** (9 subplots) - All metrics visualized
- 📊 **Key metrics plot** (4 subplots) - Main results for thesis
- 📊 **Comparison plot** - Side-by-side method comparison
- 📊 **Bar chart** - Final accuracy comparison

**Files saved to:**
- `plots/grpo_metrics.png` (detailed, 9 subplots)
- `plots/grpo_key_metrics.png` (simplified, 4 subplots)
- `plots/nsr_metrics.png` (detailed)
- `plots/nsr_key_metrics.png` (simplified)
- `plots/comparison_all_methods.png` (multi-method comparison)
- `plots/final_accuracy_comparison.png` (bar chart)

---

## Quick Commands

### For GRPO Only
```bash
python view_training_metrics.py grpo.log
```

### For GRPO vs NSR Comparison
```bash
python view_training_metrics.py grpo.log nsr.log
```

### For All 4 Baselines (after PSR/W-REINFORCE complete)
```bash
python view_training_metrics.py grpo.log nsr.log psr_100.log wreinforce_100.log
```

---

## What Metrics to Monitor

### 🎯 Most Important: Accuracy Reward
- Should increase over training
- Your GRPO: 0% → 37.5% ✅
- Your NSR: 0% → 38% ✅

### 📈 Total Reward
- Can fluctuate
- Decreasing is OK if accuracy increases!

### 🔍 Gradient Norm
- Should be > 0.1 (healthy training)
- If 0 → No learning happening!
- Your values: 0.19-0.28 ✅

### 📉 Loss
- Can be **negative** in GRPO (normal!)
- Can **increase** (normal in RL!)
- **Ignore loss**, watch accuracy instead

---

## Understanding Your Results

### GRPO (6 steps, 50 min)

```
Final Accuracy:  37.5%
Total Reward:    5.33
Grad Norm:       0.19 (healthy)
Status:          ✅ Training worked!
```

### NSR (100 steps, 96 min)

```
Final Accuracy:  38.0% (+0.5pp over GRPO)
Total Reward:    5.xx
Grad Norm:       ~0.2 (healthy)
Status:          ✅ Training worked!
```

---

## For Your Thesis

**Use text view for:**
- Quick checks during training
- Getting exact numbers for tables
- Terminal/notebook output

**Use plot view for:**
- Thesis figures (publication quality)
- Presentations
- Method comparisons
- Visualizing training dynamics

**Recommended thesis figures:**
1. **Key metrics plot** (simplified, 4 subplots) → Main results section
2. **Comparison plot** → Method comparison section
3. **Bar chart** → Abstract/conclusion

---

## Installation Instructions

### If matplotlib not installed:

```bash
# On Colab
!pip install matplotlib numpy

# On local machine
pip install matplotlib numpy

# Or with conda
conda install matplotlib numpy
```

### If you get import errors:

```bash
# Verify installation
python -c "import matplotlib; print(matplotlib.__version__)"
python -c "import numpy; print(numpy.__version__)"
```

---

## Files Created

| File | Purpose |
|------|---------|
| `view_training_metrics.py` | Text-based viewer (no dependencies) |
| `plot_training_metrics.py` | Visual plotter (requires matplotlib) |
| `PLOTTING_GUIDE.md` | Detailed plotting documentation |
| `VISUALIZE_TRAINING.md` | This file (quick reference) |

---

## Quick Reference

```bash
# Text view (works immediately)
python view_training_metrics.py grpo.log nsr.log

# Visual plots (after installing matplotlib)
pip install matplotlib numpy
python plot_training_metrics.py grpo.log nsr.log

# Plots saved to: plots/
# All figures are 150 DPI, publication quality
```

---

## Examples for Different Scenarios

### During Training (Quick Check)
```bash
python view_training_metrics.py grpo.log
```

### After Training Completes (Compare Methods)
```bash
python view_training_metrics.py grpo.log nsr.log
```

### For Thesis Writing (Generate All Plots)
```bash
pip install matplotlib numpy
python plot_training_metrics.py grpo.log nsr.log psr_100.log wreinforce_100.log
```

### In Colab Notebook
```python
# Install first
!pip install matplotlib numpy

# Then generate plots
!python plot_training_metrics.py grpo.log nsr.log

# Display in notebook
from IPython.display import Image, display
display(Image('plots/comparison_all_methods.png'))
```

---

## Troubleshooting

**Q: "No metrics found in log"**
- Make sure training actually started and logged metrics
- Check that log file contains `{'loss': ...}` dictionaries

**Q: "ModuleNotFoundError: No module named 'matplotlib'"**
- Run: `pip install matplotlib numpy`
- Or use text view instead: `view_training_metrics.py`

**Q: "Plots are empty/wrong"**
- Verify training completed successfully
- Check log file has metric dictionaries
- Try text view first to see if metrics are parsed correctly

---

**Ready to visualize!** 🎨

**Quick start:** `python view_training_metrics.py grpo.log`
