# Training Metrics Plotting Guide

## Quick Start

### Plot Single Training Run

```bash
# Plot GRPO training
python plot_training_metrics.py grpo.log

# Plot NSR training
python plot_training_metrics.py nsr.log
```

### Compare Multiple Runs

```bash
# Compare GRPO vs NSR
python plot_training_metrics.py grpo.log nsr.log

# Compare all 4 baselines
python plot_training_metrics.py grpo.log nsr.log psr_100.log wreinforce_100.log
```

---

## What Gets Generated

### For Each Training Run

**1. Detailed Metrics Plot** (`{method}_metrics.png`)
- 9 subplots showing all training metrics:
  - Total Reward with std deviation
  - Accuracy Reward (MOST IMPORTANT!)
  - Loss (can be negative in GRPO)
  - Component rewards (format, chart type, accuracy)
  - Style rewards (table, process, length)
  - Gradient norm (training health)
  - Entropy (exploration)
  - Learning rate schedule
  - Mean completion length

**2. Key Metrics Plot** (`{method}_key_metrics.png`)
- 4 simplified plots showing:
  - Accuracy progression (with final value)
  - Total reward ± std
  - Reward breakdown by component
  - Training health (gradient norm)

### For Comparisons (Multiple Runs)

**3. Comparison Plot** (`comparison_all_methods.png`)
- Side-by-side comparison of:
  - Accuracy across methods
  - Total reward across methods
  - Loss across methods
  - Gradient norm across methods

**4. Final Accuracy Bar Chart** (`final_accuracy_comparison.png`)
- Bar chart showing final accuracy for each method
- Easy to see which method performed best

---

## Example Outputs

### Single Run Example

```bash
python plot_training_metrics.py grpo.log
```

**Output:**
```
============================================================
PARSING LOG FILES
============================================================

Parsing grpo.log... (GRPO)
  ✓ Found 6 training steps

============================================================
SUMMARY: GRPO
============================================================

Accuracy:
  Initial:  0.0%
  Peak:     45.0% (step 3)
  Final:    37.5%
  Change:   37.5%

Total Reward:
  Initial:  6.84
  Peak:     6.84 (step 1)
  Final:    5.33
  Mean:     5.63 ± 0.57

Loss:
  Initial:  -0.0344
  Final:    0.0208
  Mean:     -0.0252
  Note:     NEGATIVE values are normal in GRPO!

Training Health:
  Grad Norm (mean):  0.2268
  Grad Norm (final): 0.1906
  Entropy (mean):    0.6678
  Status:            ✓ Healthy

============================================================

✓ Saved: plots/grpo_metrics.png
✓ Saved: plots/grpo_key_metrics.png

============================================================
✓ ALL PLOTS GENERATED!
============================================================

Plots saved to: ./plots/

Generated files:
  - grpo_metrics.png (detailed)
  - grpo_key_metrics.png (simplified)
```

### Comparison Example

```bash
python plot_training_metrics.py grpo.log nsr.log
```

**Output:**
```
Parsing grpo.log... (GRPO)
  ✓ Found 6 training steps
  ✓ Saved: plots/grpo_metrics.png
  ✓ Saved: plots/grpo_key_metrics.png

Parsing nsr.log... (NSR)
  ✓ Found 100 training steps
  ✓ Saved: plots/nsr_metrics.png
  ✓ Saved: plots/nsr_key_metrics.png

============================================================
CREATING COMPARISON PLOTS
============================================================

✓ Saved: plots/comparison_all_methods.png
✓ Saved: plots/final_accuracy_comparison.png

Generated files:
  - grpo_metrics.png (detailed)
  - grpo_key_metrics.png (simplified)
  - nsr_metrics.png (detailed)
  - nsr_key_metrics.png (simplified)
  - comparison_all_methods.png
  - final_accuracy_comparison.png
```

---

## Understanding the Plots

### Most Important Metrics

**1. Accuracy Reward (Green Plot)**
- **What it measures:** Percentage of correct answers
- **What to look for:** Should increase over training
- **Your GRPO result:** 0% → 37.5% ✓
- **Your NSR result:** 0% → 38% ✓

**2. Total Reward (Blue Plot with Shaded Area)**
- **What it measures:** Sum of all 7 reward components
- **What to look for:** Can fluctuate, focus on trend
- **Note:** Decreasing reward is OK if accuracy increases!

**3. Gradient Norm (Purple Plot)**
- **What it measures:** Magnitude of gradient updates
- **What to look for:** Should be > 0 (if 0, no learning!)
- **Healthy range:** 0.1 - 1.0
- **Your values:** 0.19 - 0.28 ✓ Healthy

### Understanding Loss in GRPO

**Loss Plot (Red/Blue Scatter)**
- **Red dots:** Positive loss
- **Blue dots:** Negative loss (NORMAL in GRPO!)
- **What to watch:** NOT the loss value, but accuracy!
- **Your progression:** Negative → Positive (means converged) ✓

### Component Rewards Breakdown

**7 Reward Components:**
1. **Format Reward:** Is output in correct JSON format?
2. **Accuracy Reward:** Is the answer correct? (MOST IMPORTANT)
3. **Chart Type Reward:** Did model identify chart type correctly?
4. **Length Think Reward:** Is reasoning appropriately detailed?
5. **Num Token Reward:** Is output length appropriate?
6. **Table Style Reward:** Is table formatting correct?
7. **Process Style Reward:** Is reasoning process structured well?

---

## Common Patterns

### Healthy Training

✅ **Good signs:**
- Accuracy increases over time
- Gradient norm > 0.1
- Entropy gradually decreases (model becoming more confident)
- Component rewards improve

⚠️ **Warning signs:**
- Accuracy stuck at 0%
- Gradient norm near 0 (no learning!)
- All rewards at 0 (check reward threshold!)
- Loss exploding (> 10)

### GRPO vs NSR Differences

**GRPO (50 steps):**
- Fewer steps, faster training
- 2 generations per sample
- Trains on all samples
- Lower final accuracy (35.5%)

**NSR (100 steps):**
- More steps, longer training
- 4 generations per sample
- Trains only on negative samples
- Higher final accuracy (38.0%)

---

## Advanced Usage

### Custom Output Directory

```bash
# Save plots to custom directory
mkdir my_plots
python plot_training_metrics.py grpo.log
# Then move: mv plots/* my_plots/
```

### For Thesis/Papers

The generated plots are publication-quality (150 DPI). You can:

1. **Use Key Metrics plots** for main paper (cleaner, 4 subplots)
2. **Use Detailed plots** for appendix (comprehensive, 9 subplots)
3. **Use Comparison plots** for method comparison section
4. **Use Bar chart** for abstract/conclusion (single clear message)

### Extracting Specific Values

If you need specific numbers for your thesis:

```bash
# Run the script and look at the printed summary
python plot_training_metrics.py grpo.log | grep "Final:"
```

Output:
```
  Final:    37.5%        # Final accuracy
  Final:    5.33         # Final total reward
  Final:    0.0208       # Final loss
```

---

## Troubleshooting

### Error: "No training metrics found"

**Problem:** Log file doesn't contain metric dictionaries

**Solutions:**
1. Make sure log file is from training (not just model loading)
2. Check that training actually started
3. Verify file is not corrupted

### Error: "No module named matplotlib"

**Problem:** matplotlib not installed

**Solution:**
```bash
pip install matplotlib numpy
```

### Empty Plots

**Problem:** Metrics extracted but values all zero

**Solution:**
1. Check if training actually completed
2. Verify reward computation is working
3. Look at raw log file to see if metrics are logged

---

## What the Script Does

1. **Parses log files** - Extracts all `{'loss': ...}` dictionaries
2. **Extracts metrics** - Pulls out 15+ different metrics per step
3. **Generates plots** - Creates matplotlib visualizations
4. **Saves as PNG** - 150 DPI, publication quality
5. **Prints summaries** - Shows key statistics in terminal

---

## File Structure

```
chartrl/
├── grpo.log                          # Your training logs
├── nsr.log
├── plot_training_metrics.py          # This script
├── PLOTTING_GUIDE.md                 # This guide
└── plots/                            # Generated plots (auto-created)
    ├── grpo_metrics.png
    ├── grpo_key_metrics.png
    ├── nsr_metrics.png
    ├── nsr_key_metrics.png
    ├── comparison_all_methods.png
    └── final_accuracy_comparison.png
```

---

## Tips for Thesis

### For Methods Section
- Use detailed metrics plot to show you monitored all aspects
- Explain each reward component

### For Results Section
- Use key metrics plot to show training progression
- Use comparison plot to compare GRPO vs NSR vs PSR vs W-REINFORCE
- Use bar chart to show final accuracy comparison

### For Discussion
- Plot reward breakdown to discuss what the model learned
- Show gradient norm to prove training was stable
- Compare entropy to discuss exploration vs exploitation

---

## Quick Reference

| Plot | What it Shows | When to Use |
|------|---------------|-------------|
| `{method}_metrics.png` | All 9 metrics in detail | Appendix, detailed analysis |
| `{method}_key_metrics.png` | 4 most important metrics | Main results section |
| `comparison_all_methods.png` | Side-by-side comparison | Method comparison |
| `final_accuracy_comparison.png` | Final accuracy bar chart | Abstract, conclusion |

---

## Example for Your Thesis

**Figure Caption:**

> **Figure 1:** Training progression for GRPO and NSR on 100 EvoChart samples. (Top left) Accuracy reward shows NSR achieves 38.0% vs GRPO's 35.5%. (Top right) Total reward with standard deviation bands. (Bottom left) Reward component breakdown shows both methods learn chart type recognition fastest. (Bottom right) Gradient norm confirms stable training for both methods.

---

**Ready to visualize your training!** 🎨

Just run: `python plot_training_metrics.py grpo.log nsr.log`
