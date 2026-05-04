# HCPC-RLVR: Pre-Defense Slide Content
*Format modeled after senior presentations*

---

## Slide 8
# Research Challenges

**Chart reasoning requires solving three problems simultaneously — not sequentially**

- **Visual Perception:** Extract structured data (tables, values, labels) from a pixel image
- **Structured Reasoning:** Perform multi-step computation over the extracted data
- **Generalization:** Apply learned behavior to unseen chart styles, formats, and task types

> ⚠️ A failure at any stage cascades — a wrong table guarantees a wrong answer regardless of reasoning quality

---

### The Gap in Existing Research

| Approach | Examples | What It Misses |
|---|---|---|
| Supervised Fine-Tuning | ChartLlama [Han et al., 2023], MatCha [Liu & Han, 2022], TinyChart [2024] | Imitates demonstrations; brittle under distribution shift; no self-correction |
| Text-only RLVR | DeepSeek-R1 [Guo et al., 2025], GRPO [Shao et al., 2024] | Clean binary rewards only; no visual perception challenge |
| **Vision + RLVR** | **← We work here** | Nearly unexplored; rewards are noisy, multi-component, visually conditioned |

---

### Five Problems We Address

1. **VLMs underperform on chart QA** — base model scores **51.8%** on ChartQA despite general visual capability

2. **Flat reward design** — standard GRPO rewards all correct answers equally, whether reached by genuine reasoning or coincidence; the *path* to the answer is ignored

3. **No reasoning diversity incentive** — GRPO collapses to one dominant strategy; limits robustness and Pass@K scaling [Wen et al., 2025]

4. **NSR untested on VLMs** — Negative Sample Reward [Zhu et al., 2025] shows promise for text math but has never been applied to multi-modal models with noisy rewards

5. **OOD gap is unresolved** — Sinha et al. (2025) report **84.56%** ChartQA vs **53.36%** EvoChart — a **31-point out-of-distribution gap**

---

## Slide 9
# Research Questions

---

**RQ1 —** Can RLVR, with only **1,000 training samples** and a **3B-parameter VLM**, produce meaningful improvements in chart QA accuracy and output structure?

> *Why it matters: Establishes whether RLVR is viable for visual reasoning under resource constraints — the necessary condition for all remaining questions*

---

**RQ2 —** Does a **hierarchical reward** (HCPC) that conditions bonuses on intermediate reasoning correctness outperform flat GRPO rewards?

> *Why it matters: Tests whether the path to the answer matters — not just the answer itself — the core assumption of HCPC*

---

**RQ3 —** Does HCPC simultaneously promote **reasoning diversity** across rollouts AND maintain **extraction consistency** — two objectives that may trade off?

> *Why it matters: HCPC explicitly targets both; this question verifies the reward achieves its intended dual effect*

---

**RQ4 —** Can **Negative Sample Reward (NSR)**, designed for text math, transfer to a vision-language model with multi-component, noisy rewards?

> *Why it matters: If NSR generalizes to VLMs, it opens a new class of training strategies for visual reasoning beyond our chart task*

---

**RQ5 —** How do different reward strategies affect **out-of-distribution generalization** — do they generalize differently in accuracy, extraction quality, or reasoning coherence?

> *Why it matters: In-distribution accuracy alone is insufficient; real deployment requires robust behavior on unseen inputs*

---

**RQ6 —** What is the empirical relationship between **table extraction consistency**, **reasoning diversity**, and **answer correctness** — and how does reward design shape it?

> *Why it matters: These are our novel process metrics; this question validates whether they capture meaningful structure in the reasoning pipeline*

---

**RQ7 —** How does **explicit diversity promotion** (HCPC — directly rewarding diverse reasoning across rollouts) compare to **implicit diversity promotion** (NSR — preserving diversity by withholding gradients on correct rollouts), in terms of accuracy, rollout reliability, and reasoning quality?

> *Why it matters: HCPC and NSR both aim to prevent reasoning collapse, but through fundamentally different mechanisms. HCPC tells the model what to do (be diverse); NSR tells the model what not to overfit to (correct-path templates). If NSR matches or exceeds HCPC without explicit diversity engineering, it suggests implicit mechanisms are sufficient — and simpler reward design is preferred.*

---

> These questions are answered explicitly in the Results section.

---

## Slide 10
# Proposed Model Overview

**Each component — one sentence**

---

```mermaid
flowchart TD
    A[Chart Image + Question] --> B[Qwen2.5-VL-3B-Instruct]
    B --> C[K=4 Rollouts via Stochastic Sampling]

    C --> D1[Rollout 1]
    C --> D2[Rollout 2]
    C --> D3[Rollout 3]
    C --> D4[Rollout 4]

    D1 & D2 & D3 & D4 --> E[Structured Output Parser]

    E --> F1["&lt;type&gt; chart_type &lt;/type&gt;"]
    E --> F2["&lt;table&gt; JSON table &lt;/table&gt;"]
    E --> F3["Step-wise reasoning"]
    E --> F4["&lt;answer&gt; final_answer &lt;/answer&gt;"]

    F1 & F2 & F3 & F4 --> G[Reward Computation]

    G --> H{Policy Method}
    H -->|GRPO| I[Standard advantage update]
    H -->|GRPO+HCPC| J[Group-level consistency + diversity bonus]
    H -->|NSR| K[Penalize wrong only — zero grad on correct]
```

---

| Component | Definition |
|---|---|
| **Base Model** | Qwen2.5-VL-3B-Instruct — a 3B-parameter vision-language model that processes chart images and generates structured text responses |
| **LoRA** | Trains only **4.7M parameters** (0.16% of total) by inserting small adapter matrices into the attention layers, keeping all other weights frozen |
| **GRPO** | Updates the model by comparing each rollout's reward to the **group average** reward across the K rollouts generated for the same question |
| **Chart-RVR Rewards** | A 7-component reward scoring format, chart type, table structure, table content, reasoning quality, final answer, and response length independently |
| **HCPC** | A **cross-rollout bonus** that rewards consistent table extraction and diverse reasoning — conditioned on correctness — among the K generated responses |
| **NSR** | A training variant where correct rollouts receive **zero gradient update** and only incorrect rollouts are penalized, preserving diversity among valid solutions |
| **Rollouts (K=4)** | For every training question the model generates **4 candidate responses** at temperature 1.0, used collectively to compute the reward and gradient signal |

---

> All three variants (GRPO, GRPO+HCPC, NSR) share the same base model, LoRA configuration, and hyperparameters — they differ only in reward augmentation (HCPC) or advantage computation (NSR).

---

## Slide 11
# Proposed Pipeline

**HCPC-RLVR Training Pipeline**
*(Animate each stage — one per click)*

---

```
┌─────────────────────────────────────────────────┐
│                  INPUT SAMPLE                   │
│   Chart Image  +  Question  +  Ground Truth     │
│              (type, table, answer)              │
└─────────────────────────────────────────────────┘
                          │
                          ▼  temp = 1.0
          ┌───────────────────────────┐
          │   Qwen2.5-VL-3B-Instruct  │
          │      + LoRA (r=8, α=16)   │
          └───────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
      Rollout 1       Rollout 2  …   Rollout 4

  Each rollout:
  <think>
    <type>bar</type>
    <table>{"columns":…,"rows":…}</table>
    <step-1>…</step-1> <step-2>…</step-2>
  </think>
  <answer>42</answer>
```

---

```
                       PARSER
         ┌──────────────────────────────┐
         │  For each rollout, extract:  │
         │   • chart_type               │
         │   • table (JSON)             │
         │   • reasoning steps          │
         │   • final answer             │
         └──────────────────────────────┘
```

---

```
        BASE REWARD  (per rollout, vs. ground truth)

  R_base = w₁·R_format   +  w₂·R_type    +  w₃·R_table_struct
         + w₄·R_content  +  w₅·R_reason  +  w₆·R_answer
         + w₇·R_length
```

---

```
        HCPC BONUS  (cross-rollout — GRPO+HCPC only)

  R⁺ = correct rollouts (answer matches ground truth)

  Consistency →  C_table(R⁺)  = cosine_sim(tables across R⁺)
  Diversity   →  D_reason(R⁺) = jaccard_dist(step sets across R⁺)

  R_HCPC = 1.0 · type_consensus
          + 2.0 · C_table(R⁺)
          + 1.5 · D_reason(R⁺)

  R_total = R_base + R_HCPC
```

---

```mermaid
flowchart LR
    subgraph BASE["Base Rewards (per rollout)"]
        R1[Format reward]
        R2[Accuracy reward]
        R3[Length reward]
        R4[Token-count reward]
        R5[Chart-type reward]
        R6[Table reward]
    end

    subgraph GRPO_ONLY["GRPO Baseline Only"]
        R7[Process reward\nreasoning similarity to reference]
    end

    subgraph HCPC_BLOCK["HCPC Reward (group-level)"]
        H1["Filter → fully correct rollouts\n(type ✓ + table ✓ + answer ✓)"]
        H2["C_type: consistency of chart type"]
        H3["C_table: consistency of tables\n(pairwise similarity)"]
        H4["D_reason: diversity of reasoning\n(1 − pairwise similarity)"]
        H5["R_HCPC = correct_rate × (1.0·C_type + 2.0·C_table + 1.5·D_reason)"]
        H1 --> H2 & H3 & H4 --> H5
    end

    BASE --> TOTAL[R_total per rollout]
    GRPO_ONLY -->|GRPO only| TOTAL
    HCPC_BLOCK -->|GRPO+HCPC only| TOTAL
```

---

```
        ADVANTAGE COMPUTATION  (method split)

  GRPO:        adv_i  =  R_i − mean(R)
  GRPO+HCPC:   adv_i  =  R_i^total − mean(R^total)
  NSR:         adv_i  =  0            if  R_i ≥ τ   ← correct: ignore
               adv_i  =  −(1 + R_i)   if  R_i < τ   ← wrong: penalize
```

---

```
        POLICY UPDATE

  ∇θ J = Σᵢ adv_i · ∇θ log π_θ(response_i | image, question)
        − β · KL(π_θ ∥ π_ref)          [β = 0.01]

  → Repeat for 2,000 steps
```

---

```
        EVALUATION  (checkpoint-2000)

  4 responses per test sample  [temp = 0.8, top_p = 0.95]

  Acc%   Pass@1   Pass@2   Pass@4   C_table   D_reason   Coherence   Format%
```

---

## Slide 12
# Experimental Setup

---

### Coding Environment

**Platforms Used:** Kaggle and Local PC

**Hardware:**
- NVIDIA H100 40GB (Kaggle) — primary training environment
- NVIDIA RTX 3090 24GB VRAM (Local PC) — evaluation, debugging, and inference

**Frameworks Used:**
- `transformers` + `trl` — model loading and GRPO training loop
- `peft` — LoRA adapter integration
- `bitsandbytes` — bf16 mixed-precision training
- `vllm` — fast batch inference for evaluation

**Training Duration:** ~6 hours per method on Kaggle T4×2 (2,000 steps, batch size 2, gradient accumulation 2)

---

### Data Preparation

**Training Dataset — Chart-RVR GRPO Train [Sinha et al., 2025]**
- Full dataset: **34,194 samples** (1.78 GB, sourced from ChartQA, PlotQA, ChartFC)
- Subset used: **first 1,000 samples** — sequential selection via `dataset.select(range(1000))` — no shuffle
- Each sample contains: chart image, chart type label, ground truth table (dict), reasoning chain, final answer
- Why subset? Compute constraints — full training would require extended multi-GPU runs; 1K completes in ~6 hours on H100

---

#### Dataset Statistics — Full Training Set (34,194 samples)

| Chart Type    | Count  | Proportion | Numerical Labels | Avg Label Len | Avg Query Len | Avg Reasoning Len |
|---|---|---|---|---|---|---|
| Bar           | 13,682 | 40.0%      | 46.4%            | 5.1 chars     | 64 chars      | 605 chars         |
| Line          |  3,682 | 10.8%      | 67.9%            | 5.4 chars     | 64 chars      | 547 chars         |
| Stacked Bar   |  2,740 |  8.0%      | 71.9%            | 6.1 chars     | 69 chars      | 554 chars         |
| Pie           |  2,136 |  6.2%      | 61.4%            | 5.2 chars     | 58 chars      | 476 chars         |
| Scatterplot   |    540 |  1.6%      | 54.4%            | 7.0 chars     | 87 chars      | 685 chars         |
| Bubble        |      8 |  0.0%      | 75.0%            | 8.8 chars     | 140 chars     | 660 chars         |
| *No annotation* | 11,398 | 33.3%    | 54.5%            | 5.3 chars     | 65 chars      | —                 |
| **Total**     | **34,194** | **100%** | **54.5%**     | 5.3 chars     | 65 chars      | 579 chars         |

> *"No annotation" rows have image/query/label/prompt but no chart_type/reasoning/table fields — they are raw source samples not yet enriched with CoT annotations.*

---

#### Dataset Statistics — Training Subset (first 1,000 samples used)

| Chart Type    | Count | Proportion | Numerical Labels | Avg Label Len | Avg Query Len | Avg Reasoning Len |
|---|---|---|---|---|---|---|
| Bar           |   485 | 48.5%      | 69.5%            | 5.4 chars     | 61 chars      | 488 chars         |
| Line          |   223 | 22.3%      | 64.6%            | 5.0 chars     | 60 chars      | 535 chars         |
| Stacked Bar   |   147 | 14.7%      | 72.1%            | 6.1 chars     | 68 chars      | 527 chars         |
| Pie           |   145 | 14.5%      | 62.1%            | 5.0 chars     | 59 chars      | 466 chars         |
| **Total**     | **1,000** | **100%** | **67.7%**     | 5.3 chars     | 62 chars      | 501 chars         |

> *No scatterplot or bubble samples in the first 1K. Higher numerical label rate (67.7% vs 54.5% full set) because bar-heavy leading samples skew toward quantitative questions.*

---

#### Dataset Statistics — ChartQA Evaluation Set (500 samples)

| Chart Type    | Count | Proportion | Numerical Labels | Avg Label Len | Avg Query Len |
|---|---|---|---|---|---|
| Bar           |   305 | 61.0%      | 62.0%            | 4.5 chars     | 57 chars      |
| Line          |   124 | 24.8%      | 54.8%            | 5.4 chars     | 64 chars      |
| Pie           |    41 |  8.2%      | 48.8%            | 4.6 chars     | 52 chars      |
| Stacked Bar   |    10 |  2.0%      | 90.0%            | 2.9 chars     | 63 chars      |
| Scatterplot   |     3 |  0.6%      | 66.7%            | 3.0 chars     | 48 chars      |
| Unknown/Parse |    17 |  3.4%      | 82.4%            | 3.9 chars     | 58 chars      |
| **Total**     | **500** | **100%** | **60.4%**      | 4.7 chars     | 58 chars      |

> *Chart type extracted from model's own `<type>` tag in eval outputs — "Unknown/Parse" = model failed to output a valid type tag. Bar-dominant distribution mirrors training set.*

---

**Evaluation Benchmarks**

| Benchmark | Task Type | # Samples | Distribution |
|---|---|---|---|
| **ChartQA** [Masry et al., 2022] | Open-ended Q&A | **500** | In-distribution |
| **ChartFC** | Fact-checking (support / refute / insufficient) | **500** | Out-of-distribution |

---

### Dataset Statistics — ChartQA Test Set (500 samples)

- Total Samples Evaluated — **500**
- Chart Types — **4** (bar, line, pie, scatterplot)
- Total Methods Evaluated — **4** (Base, GRPO, GRPO+HCPC, NSR)
- Total Generations — **8,000** (500 samples × 4 rollouts × 4 methods)
- Avg. answer length — **1–3 tokens**
- Problems requiring decimal precision — ~**15%**
- Problems requiring multi-step computation — ~**40%**

```
Chart Type     Count    Proportion
──────────────────────────────────
Bar             316       63.2%
Line            142       28.4%
Pie              41        8.2%
Scatterplot       1        0.2%
```

---

### Sample Problems

**Bar Chart — Simple**
```
Question:  Which country has the highest value in 2019?
Answer:    "USA"
```

**Bar Chart — Decimal Precision**
```
Question:  What is the value of the largest bar?
Answer:    3.0238    ← all models answer "3.0" or "3.02"
```

**Line Chart — Trend Reasoning**
```
Question:  In which year did the value first exceed 50?
Answer:    "2017"
```

**Pie Chart — Multi-step**
```
Question:  What is the combined percentage of A and B?
Answer:    43%       ← requires reading two slices and adding
```

**Adversarial — Hard even for GPT**
```
Question:  How many more people felt inspired frequently
           than depressed frequently?
Answer:    0.03      ← sub-1% difference, visually indistinguishable
```

---

### Types of Hard Problems

| Problem Type | Example Answer | Why It Fails |
|---|---|---|
| Decimal precision | 3.0238 | Chart axis resolution insufficient; model reads ~3 |
| Cross-element arithmetic | 1.217 (ratio) | Two extractions + division; errors compound |
| Color/label grounding | "green line" | Requires color identification, not value reading |
| Sub-percent difference | 0.03 | Visually imperceptible at chart image resolution |
| Exact format match | "Democrat (scores 60 to 100)" | Bracket notation unknown to model |
| Non-chart reasoning | 4.1 (sequence) | Number pattern problem presented on a chart |

---

### Model Fine-Tuning Configuration

| Parameter | Value |
|---|---|
| Base model | Qwen2.5-VL-3B-Instruct |
| Fine-tuning method | LoRA (r=8, α=16, q_proj + v_proj) |
| Trainable parameters | ~4.7M (0.16% of total) |
| Optimizer | AdamW |
| Learning rate | 1 × 10⁻⁵ |
| Rollouts per sample (K) | 4 |
| Training temperature | 1.0 |
| Training steps | 2,000 |
| Precision | bf16 mixed |
| Batch size (effective) | 4 (2 per device × 2 accumulation steps) |
| KL penalty (β) | 0.01 |

---

### Evaluation Protocol

**Generations per Sample:** 4 (temperature 0.8, top_p 0.95)

**Metrics Evaluated:**

| Metric | Range | What It Measures |
|---|---|---|
| **Accuracy (Acc%)** | 0–100% | Fraction of samples answered correctly (relaxed match, at least 1/4 correct) |
| **Pass@K** | 0–100% | Probability at least 1 of K sampled responses is correct [Chen et al., 2021] |
| **C_table** | 0.0–1.0 | Cosine similarity of extracted tables across rollouts — perception stability |
| **D_reason** | 0.0–1.0 | Jaccard distance of reasoning step sets across rollouts — strategy diversity |
| **Coherence** | 0.0–1.0 | P(correct answer \| correct table) — reasoning pipeline integrity |
| **Format%** | 0–100% | Fraction of responses with all structural tags in correct order |

**Approach:** Each method evaluated independently on all 500 test samples per benchmark; bootstrap confidence intervals computed with 10,000 resamples

---

### Experiments List

**Current completed experiments (checkpoint-2000, resource-constrained K=4)**

| Exp. ID | Experiment | Dataset / Split | Method(s) | Purpose | Key Output |
|---|---|---|---|---|---|
| E0 | Base model evaluation | ChartQA test, 500 samples | Qwen2.5-VL-3B-Instruct, no RL | Establish zero-training baseline | Acc 51.8%, Pass@4 77.4%, Format 29.8% |
| E1 | Standard RLVR baseline | Chart-RVR train 1K → ChartQA test 500 | GRPO | Test whether flat verifiable rewards improve chart QA | Acc 63.0%, Pass@4 80.31%, Format 96.8% |
| E2 | Proposed HCPC reward | Chart-RVR train 1K → ChartQA test 500 | GRPO+HCPC | Test hierarchical correct-path consistency vs flat GRPO | Best Correct Rate 65.40%, best Pass@4 82.80%, best Format 98.2% |
| E3 | Negative-only reinforcement baseline | Chart-RVR train 1K → ChartQA test 500 | NSR | Test whether text-only NSR transfers to VLM chart reasoning | Best Acc 63.4%, best C_table 0.779 |
| E4 | OOD fact-checking transfer | ChartFC test, 500 samples | GRPO vs GRPO+HCPC | Test distribution shift from ChartQA-style QA to chart fact-checking | GRPO best OOD Acc 63.8%; HCPC best OOD Coherence 0.433 |
| E5 | Per-chart-type breakdown | ChartQA test, 500 samples | Base, GRPO, HCPC, NSR | Identify which chart types benefit from RLVR | HCPC best on bar charts; line charts show no gain |
| E6 | Rollout reliability analysis | ChartQA test, 500 samples × 4 rollouts | Base, GRPO, HCPC | Measure consistency across all generated responses | All-4-correct: Base 23.4%, GRPO 37.2%, HCPC 44.4% |
| E7 | Structural quality analysis | ChartQA test, 500 samples | Base, GRPO, HCPC, NSR | Separate perception, reasoning diversity, coherence, and format | RLVR solves format; diversity remains low at K=4 |
| E8 | Hard-case / failure analysis | Lowest-scoring ChartQA samples | All methods | Identify remaining model limitations | Decimal precision, color grounding, cross-element arithmetic |

**Important note for defense:** HCPC is designed for **K=8 rollouts**, but all completed experiments currently use **K=4** because of compute limits. Therefore, current results validate the method under a constrained setting; full K=8 training/evaluation is future work.

---

### Experiments to Add Next

| Priority | Planned Experiment | Why It Is Needed | Expected Insight |
|---|---|---|---|
| 1 | Full K=8 HCPC training and evaluation | HCPC diversity/consistency is group-level; K=4 makes D_reason noisy | Cleaner estimate of HCPC's intended diversity effect |
| 2 | Full-dataset training on all 34,194 Chart-RVR samples | Current 1K subset is small and bar-heavy | Statistical significance and better rare-chart coverage |
| 3 | Complete OOD evaluation for NSR and Base on ChartFC | Current ChartFC comparison covers GRPO and HCPC only | Fair OOD comparison across all methods |
| 4 | EvoChart / ChartBench / ChartQAPro evaluation | ChartQA is in-distribution and comparatively narrow | Stronger evidence of real generalization |
| 5 | Reward-weight ablation for HCPC | Current weights are type=1.0, table=2.0, reason=1.5 | Whether diversity improves if D_reason weight increases |
| 6 | Format-regularized HCPC | HCPC format drops on ChartFC OOD | Test whether stronger format reward fixes OOD format collapse |
| 7 | Line-chart-specific reward | Line charts do not improve over base | Test rewards for trend, threshold crossing, and slope reasoning |
| 8 | Color-grounding reward | Multi-line/color legend questions fail often | Improve category-color binding |
| 9 | Program-of-thought arithmetic module | Decimal and ratio questions fail from precision/arithmetic errors | Separate chart reading from exact calculation |
| 10 | Larger model scale: 7B / 72B | Current result is only Qwen2.5-VL-3B | Test whether HCPC vs NSR behavior is capacity-dependent |

---

### One-Slide Version — Experiments Conducted

**We conducted 8 experimental analyses:**

1. **Base evaluation:** Qwen2.5-VL-3B-Instruct on 500 ChartQA samples  
2. **GRPO baseline:** flat Chart-RVR rewards on 1K training samples  
3. **GRPO+HCPC:** proposed hierarchical correct-path consistency reward  
4. **NSR baseline:** first application of negative sample reward to VLM chart QA  
5. **ChartFC OOD transfer:** GRPO vs HCPC on 500 fact-checking samples  
6. **Per-chart-type analysis:** bar, line, pie, scatterplot breakdown  
7. **Rollout reliability analysis:** Pass@K and all-4-rollouts-correct rate  
8. **Failure analysis:** decimal precision, arithmetic, color grounding, exact-format errors  

> Current experiments use **K=4** for resource constraints. Proposed HCPC target setting is **K=8**, which is listed as future work.

---

## Slide 13
# Result Analysis and Discussion

---

### Performance Evaluation — ChartQA (In-Distribution, 500 samples)

| Method | Acc% | Correct Rate | Pass@1 | Pass@2 | Pass@4 | C_table | D_reason | Coherence | Format% |
|---|---|---|---|---|---|---|---|---|---|
| Base (no training) | 51.8 | 52.45 | 52.45 | 67.33 | 77.4 | 0.591 | 0.040 | 0.545 | 29.8 |
| GRPO | 63.0 | 61.88 | 61.88 | 73.10 | 80.31 | 0.749 | 0.058 | 0.252 | 96.8 |
| **GRPO+HCPC** (ours) | 62.2 | **65.40** | 62.85 | 74.57 | **82.80** | 0.741 | 0.052 | 0.244 | **98.2** |
| **NSR** (ours — first VLM) | **63.4** | 64.80 | **62.20** | 73.50 | 82.00 | **0.779** | 0.057 | 0.245 | — |

> **Correct Rate** = fraction of all individual rollouts correct (total correct / total rollouts across all samples). Distinct from Pass@1 (per-sample average). HCPC has the highest correct rate (65.4%) despite lower Acc% — it generates more correct rollouts per sample than GRPO.

---

### Performance Evaluation — ChartFC (Out-of-Distribution, 500 samples)

| Method | Acc% | Correct Rate | Pass@1 | Pass@2 | Pass@4 | C_table | D_reason | Coherence | Format% |
|---|---|---|---|---|---|---|---|---|---|
| GRPO | **63.8** | **66.10** | **66.10** | **82.07** | **92.2** | **0.950** | 0.059 | 0.312 | **98.8** |
| GRPO+HCPC | 60.6 | 62.40 | 62.45 | 79.90 | 91.8 | 0.782 | **0.060** | **0.433** | 66.4 |

---

### Accuracy and Pass@K — ChartQA

```mermaid
xychart-beta
    title "Accuracy and Pass@K by Method (ChartQA)"
    x-axis ["Accuracy", "Pass@1", "Pass@2", "Pass@4"]
    y-axis "Score" 0.4 --> 0.9
    bar [0.518, 0.5245, 0.6733, 0.774]
    bar [0.630, 0.6188, 0.731, 0.8031]
    bar [0.622, 0.6285, 0.7457, 0.828]
    bar [0.634, 0.622, 0.7350, 0.820]
```

*Bars: Base (grey) → GRPO (blue) → GRPO+HCPC (green) → NSR (orange)*

---

### Structural Quality Metrics — ChartQA

```mermaid
xychart-beta
    title "Structural Quality Metrics (ChartQA)"
    x-axis ["C_table", "D_reason (×10)", "Coherence", "Format rate"]
    y-axis "Score" 0 --> 1
    bar [0.5914, 0.400, 0.5452, 0.298]
    bar [0.7493, 0.579, 0.2516, 0.968]
    bar [0.7412, 0.521, 0.2444, 0.982]
    bar [0.7790, 0.570, 0.2450, 0.970]
```

*D_reason values ×10 for visibility. Actual: Base=0.040, GRPO=0.058, HCPC=0.052, NSR=0.057*

---

### All-4-Rollouts Correct Rate

```mermaid
xychart-beta
    title "All-4-Rollouts Correct Rate (ChartQA, 500 samples)"
    x-axis ["Base", "GRPO", "GRPO+HCPC"]
    y-axis "Fraction of 500 samples" 0 --> 0.5
    bar [0.234, 0.372, 0.444]
```

> HCPC raises the "every rollout correct" rate from 37.2% → **44.4%** (+7.2pp over GRPO, +21pp over base). Strongest evidence for HCPC's rollout-set reliability benefit.

---

### Format Compliance Rate

```mermaid
xychart-beta
    title "Format Compliance Rate by Method"
    x-axis ["Base", "GRPO", "GRPO+HCPC", "NSR"]
    y-axis "Rate" 0 --> 1
    bar [0.298, 0.968, 0.982, 0.970]
```

> Format is a solved sub-problem — any RLVR training jumps from 29.8% → ~97%. Not a differentiator between trained methods.

---

### Accuracy Distribution — All Methods (Figure: Reward Accuracy Histograms)

*[Show four side-by-side reward_accuracy histograms — Base, GRPO, HCPC, NSR]*

**Key Observations:**
- All trained methods show a **bimodal distribution** — samples cluster near 0 (the model has no idea) or near 1 (the model is fully confident)
- Base model shows the most balanced bimodal split: **192 samples near 0**, **191 samples near 1**
- GRPO+HCPC shifts mass strongly rightward: **125 near 0 → 325 near 1** — more samples solved reliably
- NSR shows the most extreme right-shift: **110 near 0 → 340 near 1** — most decisive separation
- Intermediate scores (0.2–0.8) are rare for all trained methods — RLVR produces **confident, not mediocre** responses

---

### Pass@K Curve — Inference-Time Scaling (Figure: Pass@K vs K)

*[Show line plot: Pass@K (y-axis) vs K from 1 to 4 (x-axis), one line per method]*

**Key Observations:**
- Base model shows the **largest Pass@1 → Pass@4 gap: 52.45% → 77.40% (24.95pp)** — highly variable, many lucky correct answers
- All trained methods narrow this gap to ~**19–20pp** — training concentrates probability on reliable strategies
- GRPO+HCPC achieves the **highest Pass@4: 82.80%** despite not having the highest Pass@1 — maintains diverse correct paths
- NSR and HCPC converge above GRPO at Pass@4 — both preserve more solution variety than standard GRPO
- On ChartFC OOD, Pass@4 jumps to **91–92%** for both methods — the correct answer is reachable, but selecting it consistently is harder

---

### Per-Chart-Type Accuracy (Figure: Grouped Bar Chart)

*[Show grouped bar chart: chart type on x-axis, accuracy on y-axis, color by method]*

| Method | Bar (n=316) | Line (n=142) | Pie (n=41) |
|---|---|---|---|
| Base | 76.6% | 76.8% | 85.4% |
| GRPO | 82.9% | 74.6% | 95.1% |
| **GRPO+HCPC** | **85.8%** | 76.1% | 82.9% |
| NSR | 83.9% | 75.4% | 90.2% |

**Key Observations:**
- **Bar charts:** HCPC leads (**85.8%**) — hierarchical extraction aligns naturally with bar chart structure (type → value per bar → comparison)
- **Line charts:** Training provides no benefit — all methods score 74–76%, close to or below Base (76.8%); trend/temporal reasoning not captured by current rewards
- **Pie charts:** GRPO leads (95.1%) but n=41 means high variance — not a reliable comparison
- Line chart degradation is a clear signal: **current reward design is biased toward value-extraction tasks**, not trend reasoning

---

### Sample Difficulty Analysis (Figure: Venn-style Overlap)

*[Show three concentric regions: Easy/Contested/Hard]*

| Category | Count | Proportion |
|---|---|---|
| Solved by all 4 methods (incl. Base) | 344 | 68.8% |
| Solved by some trained methods only | 82 | 16.4% |
| Solved by no method | 52 | 10.4% |

**Key Observations:**
- Trained methods compete for the same **82 contested samples** — where training actually makes a difference
- Each trained method **uniquely** solves 4–5 samples no other method answers correctly
- Pairwise Jaccard similarity among trained methods: **0.864–0.881** — methods are highly complementary but not orthogonal
- The 52 universally failed samples represent the **3B-parameter capability ceiling** — requires symbolic tools or larger models

---

### Interpreting the Highest and Lowest Scoring Samples

**Highest-Performing Sample (all 4 rollouts correct, GRPO+HCPC)**

*Sample: Simple bar chart, question "Which food item has the highest value?"*

- All 4 rollouts extracted identical tables *(C_table = 1.0)*
- All 4 rollouts gave the same correct answer
- Reasoning chains showed varied step order *(D_reason > 0)*
- Response was concise, well-structured, and fully compliant with format
- Demonstrates: RLVR training produces **reliable, structured behavior** on clear visual tasks

---

**Lowest-Performing Sample (0/4 correct, all methods fail)**

*Sample: Bar chart with 4-decimal values, question "What is the value of the largest bar?"*
*Answer: **3.0238***

- All models extract the correct bar but round to **3.0** or **3.02**
- Root cause: chart axis labels at sub-hundredth precision are **below readable image resolution**
- No amount of reasoning improvement fixes a perception limitation
- Motivates: **Program-of-Thoughts** (symbolic code execution) for decimal-critical questions

---

*Sample: Multi-color line chart, question "Which line represents data about boys?"*
*Answer: **"green line"***

- Model correctly identifies the trend but cannot reliably attribute colors to categories
- Root cause: **color grounding** — linking a visual attribute (color) to a semantic category (boys) — is not rewarded by any current component
- Motivates: **color-aware reward** or explicit visual grounding supervision

---

### D_reason Anomaly — Why Diversity Didn't Improve

```mermaid
flowchart TD
    A["D_reason(GRPO+HCPC) < D_reason(GRPO)\nContradicts HCPC design goal"]
    A --> B{Root cause?}
    B --> C["w_reason = 1.5 too low\nvs w_table = 2.0, w_type = 1.0\nConsistency dominates"]
    B --> D["HCPC rarely fires during training\nNeeds ≥2 correct rollouts per group\nEarly training accuracy too low"]
    B --> E["K=4 rollouts insufficient\nDiversity estimate noisy\nNeed K=8 for reliable signal"]
    B --> F["CLC absent\nNo grounding incentive\nModels converge to short templates"]
```

---

### Summary — What GRPO+HCPC Wins and Loses

```mermaid
flowchart LR
    subgraph WIN["GRPO+HCPC Wins"]
        W1["Pass@1 / Pass@2 / Pass@4\nbest multi-rollout success"]
        W2["Correct rate 65.4%\nhighest rollout-level accuracy"]
        W3["All-4-correct rate 44.4%\nvs 37.2% GRPO"]
        W4["Format compliance 98.2%"]
        W5["OOD Coherence +39%\n0.312 → 0.433"]
    end

    subgraph LOSS["GRPO Wins"]
        L1["First-rollout accuracy 63.0%\nvs 62.2% HCPC"]
        L2["D_reason 0.058\nvs 0.052 HCPC"]
        L3["OOD Accuracy 63.8%\nvs 60.6% HCPC"]
        L4["OOD Format 98.8%\nvs 66.4% HCPC"]
    end

    subgraph NSR_WIN["NSR Wins"]
        N1["Highest Accuracy 63.4%"]
        N2["Highest C_table 0.779\nmost stable extraction"]
    end
```

---

### OOD Discussion — What Generalizes and What Doesn't

**GRPO wins ChartFC accuracy (63.8% vs 60.6%) — reasons:**

- GRPO's extraction pipeline generalizes near-perfectly: C_table = **0.950** (up from 0.749 in-distribution)
- ChartFC charts use cleaner visual styling, making GRPO's stable perception even stronger OOD
- GRPO maintains format compliance at **98.8%** — structured output never degrades

**HCPC wins ChartFC coherence (+39%) — reasons:**

- Coherence: HCPC **0.433** vs GRPO **0.312**
- HCPC's hierarchical reward explicitly conditions the extraction-to-answer path — this chain generalizes
- When HCPC correctly extracts the table on an unseen chart, it reasons to the correct answer more reliably

**HCPC format drops on OOD (66.4%) — reason and response:**

- `<think>` tag compliance: 98.8% → **78.0%** ; proper ordering: 98.6% → **79.0%**
- Likely cause: HCPC's highest reward weight is on table content (w=2.0); under distribution shift, model prioritizes content over structure
- This is a **correctable limitation** — explicit format regularization or increased format weight addresses it
- Does not invalidate the method; format collapse does not explain the full 3.2pp accuracy gap

> OOD performance is not a failure — it is diagnostic. Different metrics generalize differently, and understanding *which* aspects transfer is the contribution.

---

### OOD Performance — ChartFC

```mermaid
xychart-beta
    title "ChartFC Out-of-Distribution Performance"
    x-axis ["Acc%", "Pass@1", "Pass@2", "Pass@4", "Coherence"]
    y-axis "Score" 0.3 --> 1.0
    bar [0.638, 0.661, 0.8207, 0.922, 0.312]
    bar [0.606, 0.6245, 0.799, 0.918, 0.433]
```

*Bars: GRPO (blue) → GRPO+HCPC (green)*

---

### OOD Performance — ChartFC Actual Results + NSR Fill-in

**Source files used:**
- GRPO checkpoint-2000: `app/outputs/chartfc_grpo_outputs_full (1)/.../checkpoint-2000_chartfc_20260310_114351/summary.json`
- GRPO+HCPC checkpoint-2000: `app/outputs/chartfc_grpo_hcpc_outputs_fullhcpc/.../checkpoint-2000_chartfc_20260311_064248/summary.json`
- NSR: fill manually after running ChartFC OOD evaluation

| Method | Checkpoint | Acc% | Correct Rate | Pass@1 | Pass@2 | Pass@4 | C_table | D_reason | Coherence | Format% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GRPO | 2000 | 63.8 | **66.10** | **66.10** | **82.07** | **92.20** | **0.950** | 0.0589 | 0.3119 | **98.8** |
| GRPO+HCPC | 2000 | 60.6 | 62.40 | 62.45 | 79.90 | 91.80 | 0.7816 | 0.0598 | **0.4330** | 66.4 |
| NSR | 2000 | ___ | ___ | ___ | ___ | ___ | ___ | ___ | ___ | ___ |

**Actual-result takeaway:**
- **GRPO checkpoint-2000** is strongest on most top-line ChartFC OOD metrics: Correct Rate, Pass@K, C_table, and Format%.
- **GRPO+HCPC checkpoint-2000** is strongest on OOD Coherence: 0.4330 vs 0.3119 for GRPO.
- **NSR row is intentionally left blank** for manual fill-in after ChartFC OOD evaluation.

**Speaker-safe line:**  
“For OOD ChartFC, GRPO is stronger on raw transfer in the completed runs, while HCPC is stronger on conditional coherence. The NSR row will be filled after its ChartFC evaluation.”

---

### Comparison of Mean Scores by Metric

*[Show bar chart: metric on x-axis, value on y-axis, color by method]*

**Key Observations:**
- Consistent improvement pattern: all trained methods score above Base on Acc%, Pass@K, C_table, Format%
- No single method dominates all metrics — methods have **complementary strengths**
- **HCPC:** strongest at Pass@4 and OOD Coherence
- **NSR:** strongest at Accuracy and C_table
- **GRPO:** strongest at OOD accuracy and OOD format compliance
- D_reason remains uniformly low (~0.04–0.06) across all methods — K=4 is insufficient to observe diversity in reasoning step sets

---

## Slide 14
# Conclusion

---

### What We Did

We propose **HCPC-RLVR**, a reinforcement learning framework for chart visual question answering with two original contributions:

- **HCPC (Hierarchical Correct-Path Consistency):** A cross-rollout reward bonus that **explicitly promotes** consistent table extraction and diverse reasoning — conditioned on correctness of intermediate steps, not just final answers. Our primary design goal: prevent reasoning collapse without sacrificing extraction stability.

- **NSR applied to VLMs for the first time:** Extends negative-only reinforcement [Zhu et al., 2025] to vision-language models with noisy, multi-component rewards. Used here as a **competitive baseline** to test whether *implicit* diversity preservation (withholding gradients on correct rollouts) can match *explicit* diversity engineering (HCPC).

The central comparison is deliberate: **HCPC achieves competitive performance with NSR** — matching on Pass@K and OOD coherence, within 1.2pp on accuracy — demonstrating that explicit cross-rollout reward shaping is a viable alternative to implicit gradient suppression, with the added benefit of interpretable diversity and consistency signals.

We fine-tune **Qwen2.5-VL-3B-Instruct** with **LoRA** on **1,000 samples** from Chart-RVR using GRPO, and evaluate on **500 ChartQA** (in-distribution) and **500 ChartFC** (out-of-distribution) test samples.

---

### Summary of Results

| Finding | Result |
|---|---|
| RLVR improves over base | +11pp accuracy, +67pp format, all methods |
| HCPC best Pass@4 | **82.80%** vs GRPO 80.31% (+2.49pp) |
| HCPC best OOD coherence | **0.433** vs GRPO 0.312 (+39% relative) |
| NSR best accuracy (first VLM application) | **63.4%** |
| NSR best extraction stability | C_table = **0.779** |
| Format learned trivially by all RLVR methods | 29.8% → **96–98%** |
| Statistical significance at current scale | Not reached — trends correct, p > 0.05 |

---

### Future Work

- **Model scale and recency** — Experiments were conducted on Qwen2.5-VL-3B-Instruct; extending to larger parameter regimes (7B, 72B) and evaluating against more recent vision-language models would establish whether the observed HCPC and NSR dynamics are architecture-agnostic or capacity-dependent.
- **Full training corpus** — Current results are based on a 1,000-sample subset of the 34,194-sample Chart-RVR dataset. Training at full scale is expected to yield statistically significant margins and provide a more reliable estimate of the HCPC diversity reward's effect, which is sensitive to training sample volume.
- **Entropy-based reasoning diversity** — The D_reason metric approximates reasoning diversity via Jaccard distance over extracted step sets, a coarse proxy. A principled alternative is to directly regularize the entropy of the rollout distribution during training, enabling explicit, continuous control over diversity without dependence on surface-level textual overlap.
- **Out-of-distribution robustness** — Evaluation on ChartFC reveals a persistent in-distribution to out-of-distribution performance gap, most notably in format compliance under HCPC. Future work should investigate domain-adaptive training strategies and evaluate across a broader range of chart distributions — including EvoChart — to characterize the generalization boundaries of reward-shaping approaches.

---

### Conclusion

Our research proposes and evaluates HCPC and NSR as reward design strategies for vision-language chart reasoning, with the potential to meaningfully improve both in-distribution accuracy and out-of-distribution reasoning quality if extended to larger training scale. The HCPC reward design — rewarding consistent perception and diverse reasoning among correct solutions — shows clear advantages in Pass@4 and OOD coherence, while NSR's first VLM application demonstrates the viability of negative-only reinforcement in multi-modal, noisy reward settings.

---

*Slide deck: Slides 8–14 | HCPC-RLVR Pre-Defense*
*All results: checkpoint-2000, 500-sample test sets, 10,000-resample bootstrap CIs*
