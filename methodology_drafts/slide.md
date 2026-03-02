# HCPC-RLVR: Hierarchical Correct-Path Consistency for Robust Chart Reasoning
## Slide Deck

---

## Slide 12: Step-by-Step Training Flow

### Training Pipeline Overview

```mermaid
flowchart LR
    A[📊 Input<br/>Chart + Question + GT] --> B[🎲 Generate<br/>K Rollouts]
    B --> C[🔍 Parse<br/>Components]
    C --> D[🎯 Compute<br/>Rewards]
    D --> E[📈 Policy<br/>Update]

    style A fill:#e1f5fe
    style B fill:#fff3e0
    style C fill:#f3e5f5
    style D fill:#e8f5e9
    style E fill:#fce4ec
```

### Detailed Steps

| Step | Action | Output |
|------|--------|--------|
| **1. Input** | Chart image + Question + Ground truth (type, table, answer) | Training sample |
| **2. Generation** | Generate K rollouts with temperature sampling | K candidate responses |
| **3. Parsing** | Extract type, table, reasoning, answer from each rollout | Structured components |
| **4. Filtering** | Identify fully correct rollouts (match GT type, table, answer) | Correct set for HCPC |
| **5. Rewards** | Compute R_base + R_HCPC | Per-sample total reward |
| **6. Advantages** | Method-dependent (GRPO/NSR/W-REINFORCE) | Gradient weights |
| **7. Update** | Policy gradient step | Updated model θ |

### Visual Flow (Mermaid)

```mermaid
flowchart TB
    subgraph Input["1️⃣ Input"]
        A["📊 Chart Image<br/>❓ Question<br/>✅ Ground Truth"]
    end

    subgraph Generation["2️⃣ Generation"]
        A --> B["🤖 VLM<br/>K rollouts, temp sampling"]
        B --> R1["Rollout 1"]
        B --> R2["Rollout 2"]
        B --> R3["..."]
        B --> RK["Rollout K"]
    end

    subgraph Parsing["3️⃣ Parsing"]
        R1 & R2 & R3 & RK --> P["Extract:<br/>• Type<br/>• Table<br/>• Reasoning<br/>• Answer"]
    end

    subgraph Filtering["4️⃣ GT Filtering"]
        P --> F{"Match GT?<br/>type ✓<br/>table ✓<br/>answer ✓"}
        F -->|Yes| FC["✅ Fully Correct Set"]
        F -->|No| IC["❌ Incorrect Set"]
    end

    subgraph Rewards["5️⃣ Reward Computation"]
        FC --> HCPC["R_HCPC<br/>(cross-rollout, novel)"]
        P --> BASE["R_base<br/>(per-rollout, Chart-RVR)"]
        HCPC & BASE --> TOTAL["R_total = R_base + R_HCPC"]
    end

    subgraph Update["6️⃣ Policy Update"]
        TOTAL --> ADV{"Advantage<br/>Computation"}
        ADV --> GRPO["GRPO"]
        ADV --> NSR["NSR"]
        ADV --> WR["W-REINFORCE"]
        GRPO & NSR & WR --> GRAD["∇θ Update"]
    end

    style Input fill:#e3f2fd
    style Generation fill:#fff8e1
    style Parsing fill:#f3e5f5
    style Filtering fill:#fff3e0
    style Rewards fill:#e8f5e9
    style Update fill:#fce4ec
```

---

## Slide 13: Reward Computation

### Two Reward Components

#### 1. Base Reward (R_base) — Per-Rollout (from Chart-RVR)
```
R_base = R_format + R_type + R_table + R_process + R_accuracy
```

#### 2. HCPC Reward (R_HCPC) — Cross-Rollout (Novel, Ours)
**Computed only among ground-truth-correct rollouts:**

```
R_HCPC = (|fully_correct| / K) × (w₁·C_type + w₂·C_table + w₃·D_reason)
```

| Component | What it measures | Goal |
|-----------|------------------|------|
| C_type | Fraction of correct rollouts with same chart type | HIGH (consistency) |
| C_table | Avg pairwise similarity of extracted tables | HIGH (consistency) |
| D_reason | 1 - avg pairwise similarity of reasoning traces | HIGH (diversity) |

### Total Reward
```
R_total[i] = R_base[i] + R_HCPC
```

### Reward Flow Diagram

```mermaid
flowchart TB
    subgraph PerRollout["Per-Rollout (Chart-RVR)"]
        R1["Rollout 1"] --> B1["R_base[1]"]
        R2["Rollout 2"] --> B2["R_base[2]"]
        RK["Rollout K"] --> BK["R_base[K]"]
    end

    subgraph CrossRollout["Cross-Rollout (HCPC - Novel)"]
        FC["Fully Correct<br/>Rollouts Only"] --> CT["C_type<br/>consistency"]
        FC --> CTB["C_table<br/>consistency"]
        FC --> DR["D_reason<br/>diversity"]

        CT & CTB & DR --> HCPC["R_HCPC"]
    end

    subgraph Total["Total Reward"]
        B1 --> T1["R_total[1]"]
        B2 --> T2["R_total[2]"]
        BK --> TK["R_total[K]"]
        HCPC --> T1 & T2 & TK
    end

    style PerRollout fill:#e3f2fd
    style CrossRollout fill:#e8f5e9
    style Total fill:#fce4ec
```

---

## Slide 14: Why Level-Specific Objectives?

### The Key Insight

| Level | Task | Desired Property | Rationale |
|-------|------|------------------|-----------|
| **Perception** | Identify chart type | CONSISTENT | One correct type exists |
| **Extraction** | Parse data table | CONSISTENT | One correct table exists |
| **Reasoning** | Compute/analyze | DIVERSE | Multiple valid strategies |
| **Answer** | Final response | CONSISTENT | One correct answer |

### HCPC Reward Interpretation

| C_table | D_reason | Meaning |
|---------|----------|---------|
| High | High | Same data, different strategies → **Robust!** |
| High | Low | Same data, same strategy → Fragile |
| Low | High | Different data → Lucky guesses |
| Low | Low | Confused model |

### Level-Specific Objectives Diagram

```mermaid
flowchart LR
    subgraph Pipeline["Chart Reasoning Pipeline"]
        direction LR
        I["🖼️ Chart"] --> T["📋 Type ID"]
        T --> E["📊 Table<br/>Extraction"]
        E --> R["🧠 Reasoning"]
        R --> A["✅ Answer"]
    end

    subgraph Objectives["Desired Properties"]
        T -.-> OT["CONSISTENT"]
        E -.-> OE["CONSISTENT"]
        R -.-> OR["DIVERSE"]
        A -.-> OA["CONSISTENT"]
    end

    style OT fill:#c8e6c9
    style OE fill:#c8e6c9
    style OR fill:#fff9c4
    style OA fill:#c8e6c9
```

---

## Slide 15: Policy Update Methods

### Three Methods We Will Compare

#### GRPO (Group Relative Policy Optimization) - Baseline
```
advantage[i] = R_total[i] - mean(R_total)
```
- Boosts above-average rollouts
- Penalizes below-average rollouts
- Known issue: causes collapse to single strategy

#### NSR (Negative Sample Reinforcement)
```
If correct: advantage[i] = 0  (skip)
If wrong:   advantage[i] = -(1 - R_total[i])
```
- Preserves diversity by not reinforcing any single path
- Trade-off: may have lower peak accuracy

#### W-REINFORCE (Weighted REINFORCE)
```
If correct: advantage[i] = λ × R_total[i]     (λ = 0.1, weak)
If wrong:   advantage[i] = -(1 - R_total[i])  (strong)
```
- Balanced approach
- Weak positive + strong negative

### Policy Methods Comparison

```mermaid
flowchart TB
    subgraph Input["Rollouts with Rewards"]
        R["R_total for each rollout"]
    end

    R --> GRPO_box
    R --> NSR_box
    R --> WR_box

    subgraph GRPO_box["GRPO (Baseline)"]
        G1["adv = R - mean(R)"]
        G2["Strong positive & negative"]
    end

    subgraph NSR_box["NSR"]
        N1["Correct: skip"]
        N2["Wrong: penalize"]
    end

    subgraph WR_box["W-REINFORCE"]
        W1["Correct: weak boost"]
        W2["Wrong: strong penalize"]
    end

    GRPO_box --> OUT["Policy Gradient Update"]
    NSR_box --> OUT
    WR_box --> OUT

    style GRPO_box fill:#ffcdd2
    style NSR_box fill:#fff9c4
    style WR_box fill:#c8e6c9
```

---

## Slide 16: Experimental Setup

### What We Will Use

#### Model
- **Qwen2.5-VL-3B-Instruct** (same as Chart-RVR for fair comparison)

#### Datasets (from Chart-RVR)

| Dataset | Split | Size | Purpose |
|---------|-------|------|---------|
| **ChartQA** | Train | ~18K | Training |
| **ChartQA** | Test | ~2.5K | In-domain evaluation |
| **PlotQA** | Train | Subset | Additional training data |

#### Hyperparameters (To Be Tuned)

| Parameter | Planned Range |
|-----------|---------------|
| K (rollouts) | 4, 8 |
| Temperature | 0.7, 0.8, 1.0 |
| Learning Rate | 1e-6, 5e-7 |
| HCPC weights (w₁, w₂, w₃) | To be determined |
| λ (W-REINFORCE) | 0.1 (from paper) |

---

## Slide 17: Experiment Matrix

### What We Plan to Run

#### Main Experiments: 3 Policies × 2 Reward Settings = 6 Experiments

| Exp | Policy | Reward | Purpose |
|-----|--------|--------|---------|
| **1** | SFT | - | Baseline (no RL) |
| **2** | GRPO | Chart-RVR only | Replicate Chart-RVR baseline |
| **3** | GRPO | Chart-RVR + HCPC | Effect of HCPC on GRPO |
| **4** | NSR | Chart-RVR only | NSR on chart reasoning |
| **5** | NSR | Chart-RVR + HCPC | NSR + HCPC |
| **6** | W-REINFORCE | Chart-RVR only | W-REINFORCE on chart reasoning |
| **7** | W-REINFORCE | Chart-RVR + HCPC | Full method |

### Experiment Matrix Diagram

```mermaid
flowchart TB
    subgraph Baselines["Baselines"]
        SFT["Exp 1: SFT<br/>(no RL)"]
        GRPO_base["Exp 2: GRPO<br/>(Chart-RVR)"]
    end

    subgraph WithHCPC["+ HCPC (Novel)"]
        GRPO_hcpc["Exp 3: GRPO + HCPC"]
        NSR_hcpc["Exp 5: NSR + HCPC"]
        WR_hcpc["Exp 7: W-R + HCPC"]
    end

    subgraph PolicyOnly["Policy Methods Only"]
        NSR_base["Exp 4: NSR"]
        WR_base["Exp 6: W-REINFORCE"]
    end

    GRPO_base -->|"+ HCPC"| GRPO_hcpc
    NSR_base -->|"+ HCPC"| NSR_hcpc
    WR_base -->|"+ HCPC"| WR_hcpc

    style Baselines fill:#ffcdd2
    style WithHCPC fill:#c8e6c9
    style PolicyOnly fill:#fff9c4
```

---

## Slide 18: Research Questions

### What We Want to Answer

| RQ | Question | How We'll Answer |
|----|----------|------------------|
| **RQ1** | Does HCPC improve over base Chart-RVR rewards? | Compare Exp 2 vs 3, 4 vs 5, 6 vs 7 |
| **RQ2** | Do NSR/W-REINFORCE preserve diversity better than GRPO? | Measure D_reason across methods |
| **RQ3** | Does diversity (D_reason) correlate with OOD performance? | Correlation analysis |
| **RQ4** | Does extraction consistency (C_table) correlate with accuracy? | Correlation analysis |
| **RQ5** | Which combination works best? | Compare all 7 experiments |

### What We Expect to Observe

1. **HCPC should help**: Adding cross-rollout consistency/diversity signals should improve over per-rollout rewards alone
2. **NSR/W-REINFORCE should preserve diversity**: Less collapse than GRPO
3. **Trade-offs**: Higher diversity might mean slightly lower in-domain accuracy but better generalization

---

## Slide 19: Planned Ablations

### Ablation Studies We Will Run

| Ablation | What We Remove | Question |
|----------|----------------|----------|
| **A1** | Remove C_table from HCPC | Is extraction consistency necessary? |
| **A2** | Remove D_reason from HCPC | Is reasoning diversity necessary? |
| **A3** | Remove C_type from HCPC | Is type consistency necessary? |
| **A4** | Use majority vote instead of GT filtering | Does GT-anchoring matter? |
| **A5** | Vary K (4 vs 8 rollouts) | How many rollouts are needed? |
| **A6** | Vary HCPC weights | Optimal w₁, w₂, w₃? |

### Ablation Diagram

```mermaid
flowchart TB
    FULL["Full HCPC<br/>(C_type + C_table + D_reason)"]

    FULL --> A1["- C_table"]
    FULL --> A2["- D_reason"]
    FULL --> A3["- C_type"]
    FULL --> A4["Majority vote<br/>(no GT filter)"]

    A1 --> Q1["Is table consistency<br/>necessary?"]
    A2 --> Q2["Is reasoning diversity<br/>necessary?"]
    A3 --> Q3["Is type consistency<br/>necessary?"]
    A4 --> Q4["Does GT anchoring<br/>matter?"]

    style FULL fill:#c8e6c9
    style A1 fill:#fff9c4
    style A2 fill:#fff9c4
    style A3 fill:#fff9c4
    style A4 fill:#fff9c4
```

---

## Slide 20: Metrics We Will Report

### Primary Metrics

| Metric | What It Measures |
|--------|------------------|
| **ChartQA Accuracy** | In-domain performance |
| **Relaxed Accuracy** | Standard chart QA metric (allows small numeric errors) |

### Secondary Metrics (for Analysis)

| Metric | What It Measures |
|--------|------------------|
| **C_table** | Extraction consistency among correct rollouts |
| **D_reason** | Reasoning diversity among correct rollouts |
| **C_type** | Type identification consistency |
| **Entropy** | Policy entropy (diversity measure) |

### What We Hope to Show

```mermaid
flowchart LR
    subgraph Hypothesis["Expected Correlations"]
        H1["High C_table → High Accuracy"]
        H2["High D_reason → Better Generalization"]
        H3["HCPC → Improves both"]
    end

    style Hypothesis fill:#e8f5e9
```

---

## Slide 21: Expected Outcomes

### What We Expect to Find

#### Scenario 1: HCPC Works
- GRPO + HCPC > GRPO alone
- NSR/W-REINFORCE + HCPC shows best balance
- C_table correlates with accuracy
- D_reason correlates with robustness

#### Scenario 2: Policy Method Matters More
- NSR/W-REINFORCE > GRPO regardless of HCPC
- HCPC provides marginal additional gains

#### Scenario 3: Neither Helps Much
- All methods similar to baseline
- Need to rethink approach

### We Will Report Honestly

Whatever results we get, we will analyze:
- What worked and what didn't
- Why certain combinations succeeded/failed
- Limitations of the approach

---

## Slide 22: Summary

### What We're Planning

1. **Dataset**: ChartQA (from Chart-RVR)
2. **Model**: Qwen2.5-VL-3B-Instruct
3. **Baselines**: SFT, GRPO with Chart-RVR rewards
4. **Novel Contribution**: HCPC reward (cross-rollout consistency + diversity)
5. **Policy Methods**: GRPO, NSR, W-REINFORCE
6. **Experiments**: 7 main experiments + ablations

### Key Questions We'll Answer

1. Does HCPC improve chart reasoning?
2. Do NSR/W-REINFORCE preserve diversity?
3. Which combination works best?
4. What are the trade-offs?

### Timeline

| Phase | Tasks |
|-------|-------|
| **Phase 1** | Implement HCPC reward, run baselines |
| **Phase 2** | Run all 7 experiments |
| **Phase 3** | Ablation studies |
| **Phase 4** | Analysis and writing |

---

*End of Slides*
