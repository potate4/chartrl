# Thesis Proposal: Hierarchical Correct-Path Consistency for Robust Chart Reasoning

## Research Question

Can Hierarchical Correct-Path Consistency (HCPC) improve chart reasoning robustness while maintaining output diversity, especially on out-of-distribution charts?

---

## Background

### What the Papers Did

**Chart-RVR Paper:**
- Fine-tuned Qwen2.5-VL-3B on chart reasoning using GRPO
- Added verifiable rewards: chart type, table reconstruction, process conformity
- Results: 84.56% ChartQA, 53.36% EvoChart (OOD)
- Data: 6K samples from ChartQA/PlotQA/ChartFC with CoT rationales

**NSR Paper:**
- Decomposed RLVR into PSR (reinforce correct) and NSR (penalize incorrect)
- Key finding: NSR alone matches/beats GRPO on math reasoning
- NSR maintains diversity (high Pass@k), PSR improves Pass@1 but hurts Pass@k
- W-REINFORCE (λ=0.1): downweight PSR, keep full NSR → best results

---

## Our Contribution: Hierarchical Correct-Path Consistency (HCPC)

### Core Innovation

Unlike NSR which passively preserves diversity, HCPC **actively enforces level-specific objectives**:

**Chart reasoning has hierarchical structure:**
```
Chart Image → Type Identification → Table Extraction → Reasoning → Answer
```

**Different levels need different properties:**
- **Visual extraction (Type, Table):** Should be CONSISTENT (one correct interpretation)
- **Reasoning strategies:** Should be DIVERSE (multiple valid solution paths)

### Mathematical Formulation

```python
# Step 1: Strict filtering - only rollouts matching ALL dataset labels
fully_correct = [r for r in rollouts 
                 if r.type == GT.type and
                    r.answer == GT.answer and
                    table_similarity(r.table, GT.table) > 0.8]

# Step 2: Measure level-specific properties (only among fully correct)
if len(fully_correct) >= 2:
    # Level 1: Type consistency (want HIGH)
    C_type = fraction_matching_mode([r.type for r in fully_correct])
    
    # Level 2: Table consistency (want HIGH)
    C_table = avg_pairwise_similarity([r.table for r in fully_correct])
    
    # Level 3: Reasoning DIVERSITY (want HIGH)
    D_reasoning = 1 - avg_pairwise_similarity([r.reasoning for r in fully_correct])
    
    # Step 3: Combine with correctness rate
    correctness_rate = len(fully_correct) / K
    R_HCPC = correctness_rate × (1.0×C_type + 2.0×C_table + 1.5×D_reasoning)
else:
    R_HCPC = 0

# Step 4: Total reward
R_total[i] = R_base_ChartRVR[i] + R_HCPC  # R_HCPC shared across all rollouts
```

### Why Novel

**Comparison with existing work:**

| Method | Levels | Measurement | Objective | Filtering |
|--------|--------|-------------|-----------|-----------|
| Self-Consistency | Single (answer) | Cross-rollout | Uniform agreement | Majority vote |
| Chart-RVR | Multi | Per-rollout | Independent | None |
| Generic Diversity | All | Cross-rollout | Uniform diversity | None |
| **HCPC (Ours)** | Multi | Cross-rollout | **Level-specific** | **Dataset labels** |

**Novel aspects:**
1. ✅ Hierarchical structure: Different objectives at different levels
2. ✅ Cross-rollout measurement: Properties measured across rollouts
3. ✅ Level-specific: Consistency for extraction, diversity for reasoning
4. ✅ Dataset-anchored: Only among ground-truth-correct rollouts (prevents lucky guesses)

---

## Implementation Pipeline

### Architecture (9 Steps)

```
1. Input: Chart image + question + ground truth (type, table, answer)
2. Generate K=8 rollouts (temperature=0.8)
3. Parse components: Extract type, table, reasoning, answer
4. Filter by dataset labels: Keep only fully correct rollouts
5. Compute HCPC reward: Measure C_type, C_table, D_reasoning
6. Compute base rewards: Chart-RVR per-rollout rewards
7. Combine: R_total[i] = R_base[i] + R_HCPC
8. Policy update: GRPO/NSR/W-REINFORCE
9. Gradient update
```

### Policy Update Methods

| Method | Correct Samples | Wrong Samples | Diversity | Accuracy |
|--------|-----------------|---------------|-----------|----------|
| GRPO | Strong boost | Penalize | Low (collapses) | High in-domain |
| NSR | Skip | Penalize | High (preserved) | Slightly lower |
| W-REINFORCE | Weak boost (λ=0.1) | Strong penalize | Medium | Balanced |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | Qwen2.5-VL-3B-Instruct |
| Data | 1K samples (ChartQA + PlotQA) |
| K rollouts | 8 |
| Temperature | 0.8 |
| Learning rate | 5e-7 |
| Epochs | 3 |
| HCPC weights | w1=1.0, w2=2.0, w3=1.5 |
| W-REINFORCE λ | 0.1 |
| Table similarity threshold | 0.8 |
| Compute | 24GB VRAM, 3 months |

---

## Experimental Design

### Experiment Matrix (6 experiments)

| Exp | Method | HCPC | Purpose |
|-----|--------|------|---------|
| 1 | GRPO | No | Baseline (Chart-RVR replication) |
| 2 | GRPO | Yes | Does HCPC help GRPO? |
| 3 | NSR | No | Passive diversity preservation |
| 4 | NSR | Yes | NSR + active structured diversity |
| 5 | W-REINFORCE | No | W-REINFORCE on charts |
| 6 | W-REINFORCE | Yes | **Full pipeline (best expected)** |

### Evaluation

**Datasets:**
- **ChartQA** (in-domain): Similar to training
- **EvoChart** (out-of-distribution): Diverse chart styles

**Metrics:**
- **Primary:** Accuracy, OOD Gap (ID - OOD)
- **Secondary:** C_table, D_reasoning, Entropy
- **Analysis:** Correlation(C_table, Accuracy), Correlation(D_reasoning, OOD)

### Ablations

1. **Weight sensitivity:** Test w2∈{1.0,1.5,2.0,2.5}, w3∈{1.0,1.5,2.0}
2. **Filtering strategy:** Strict vs permissive
3. **Component contribution:** HCPC without C_table, HCPC without D_reasoning

---

## Expected Results

### Main Results

| Method | ChartQA | EvoChart | OOD Gap | C_table | D_reasoning |
|--------|---------|----------|---------|---------|-------------|
| GRPO | 84.6% | 53.4% | 31.2% | 0.65 | 0.30 |
| GRPO + HCPC | 85.0% | 55.5% | 29.5% | 0.78 | 0.52 |
| NSR | 84.0% | 56.0% | 28.0% | 0.70 | 0.60 |
| NSR + HCPC | 84.5% | 58.0% | 26.5% | 0.82 | 0.72 |
| W-REINFORCE | 85.5% | 57.0% | 28.5% | 0.75 | 0.55 |
| **W-REINFORCE + HCPC** | **86.0%** | **59.0%** | **27.0%** | **0.85** | **0.75** |

### Key Findings

1. ✅ **HCPC improves OOD performance** +1.5-2.5% across all methods
2. ✅ **Higher C_table correlates with accuracy** (r = +0.72)
3. ✅ **Higher D_reasoning correlates with OOD robustness** (r = +0.68)
4. ✅ **W-REINFORCE + HCPC achieves best balance**

### Example Walkthrough

**Question:** "Total sales for 2020 and 2021?"

**Ground Truth:** type='bar', table={2020:100, 2021:150}, answer=250

**After Training (W-REINFORCE + HCPC):**

**Rollout 1 (Table-based):**
```xml
<type>bar</type>
<table>{"columns": ["Year", "Sales"], "rows": [[2020, 100], [2021, 150]]}</table>
Extract table → Sum values: 100 + 150 = 250
<answer>250</answer>
```

**Rollout 2 (Visual estimation):**
```xml
<type>bar</type>
<table>{"columns": ["Year", "Sales"], "rows": [[2020, 100], [2021, 150]]}</table>
Observe bars → Estimate ~100 and ~150 → Total ~250
<answer>250</answer>
```

**Rollout 3 (Direct calculation):**
```xml
<type>bar</type>
<table>{"columns": ["Year", "Sales"], "rows": [[2020, 100], [2021, 150]]}</table>
Identify years → Calculate 100 + 150 = 250
<answer>250</answer>
```

**Analysis:**
- C_type = 1.0 (all identified "bar")
- C_table = 1.0 (all extracted same table)
- D_reasoning = 0.72 (three different strategies)
- correctness_rate = 3/8
- **R_HCPC = 0.375 × (1.0 + 2.0 + 1.08) = 1.53** (high reward!)

---

## Timeline (12 Weeks)

| Week | Task |
|------|------|
| 1-2 | Setup + Data Preparation |
| 3-4 | GRPO baseline training |
| 5-6 | HCPC implementation + testing |
| 7-8 | NSR & W-REINFORCE training (Exp 3-6) |
| 9 | Ablation studies |
| 10 | Evaluation + correlation analysis |
| 11 | Writing |
| 12 | Revision + defense prep |

**Compute:** ~80 GPU-hours (feasible with 24GB VRAM, LoRA, 1K subset)

---

## Success Criteria

**Minimum (Thesis completion):**
- HCPC improves OOD performance >1% on any method
- Higher entropy than baseline
- Complete comparison of GRPO/NSR/W-REINFORCE on charts

**Strong (Publication potential):**
- W-REINFORCE + HCPC best overall Pass@k
- OOD improvement >3%
- Strong correlations: C_table ↔ accuracy, D_reasoning ↔ OOD

---

## Contributions

1. **Hierarchical Correct-Path Consistency (HCPC):** First reward enforcing level-specific objectives (consistent extraction, diverse reasoning) across rollouts in vision-language reasoning

2. **Systematic comparison:** First application of NSR and W-REINFORCE to vision-language chart reasoning

3. **Empirical insights:** Table consistency predicts correctness, reasoning diversity predicts OOD robustness

4. **Diagnostic framework:** HCPC provides interpretable signals about model behavior at each reasoning level

---

## Key References

1. Chart-RVR (Sinha et al., 2025) - arxiv.org/abs/2510.10973
2. NSR (Zhu et al., 2025) - arxiv.org/abs/2506.01347
3. Self-Consistency (Wang et al., 2022) - ICLR 2023
4. Qwen2.5-VL - huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct

---

## Code Structure

```
SCR-RLVR/
├── data/chartqa/         # Training data
├── src/
│   ├── rewards/
│   │   ├── chart_rvr.py  # Base rewards
│   │   └── hcpc.py       # HCPC reward
│   ├── training/
│   │   ├── grpo.py
│   │   ├── nsr.py
│   │   └── wreinforce.py
│   └── utils/
│       ├── parsing.py    # Extract components
│       └── similarity.py # Embeddings
├── scripts/
│   ├── train.py
│   └── evaluate.py
└── configs/              # YAML configs
```

---

**END OF PROPOSAL**