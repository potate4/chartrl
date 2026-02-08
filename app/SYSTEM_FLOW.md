# HCPC-RLVR System Flow

This document explains the complete system flow from input to output, including all reward calculations and policy update methods.

---

## Table of Contents

1. [High-Level Overview](#1-high-level-overview)
2. [Input Format](#2-input-format)
3. [Model Output Format](#3-model-output-format)
4. [Rollout Generation](#4-rollout-generation)
5. [Reward Calculation](#5-reward-calculation)
6. [Policy Update Methods](#6-policy-update-methods)
7. [Complete Training Loop](#7-complete-training-loop)
8. [Experiment Configurations](#8-experiment-configurations)

---

## 1. High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           HCPC-RLVR TRAINING FLOW                           │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
    │ Chart Image  │      │   Question   │      │ Ground Truth │
    │    (PNG)     │      │   (Text)     │      │ (label, type,│
    │              │      │              │      │  table, CoT) │
    └──────┬───────┘      └──────┬───────┘      └──────┬───────┘
           │                     │                     │
           └──────────┬──────────┘                     │
                      ▼                                │
           ┌──────────────────────┐                    │
           │   Format as Prompt   │                    │
           │  (System + User msg) │                    │
           └──────────┬───────────┘                    │
                      ▼                                │
           ┌──────────────────────┐                    │
           │  Generate K=8        │                    │
           │  Rollouts            │                    │
           │  (temperature=0.8)   │                    │
           └──────────┬───────────┘                    │
                      │                                │
                      ▼                                ▼
           ┌──────────────────────────────────────────────┐
           │            REWARD COMPUTATION                 │
           │  ┌────────────────────────────────────────┐  │
           │  │  Base Rewards (per rollout):           │  │
           │  │  - Format, Accuracy, Length, etc.      │  │
           │  └────────────────────────────────────────┘  │
           │  ┌────────────────────────────────────────┐  │
           │  │  HCPC Reward (cross-rollout):          │  │
           │  │  - Filter to correct rollouts          │  │
           │  │  - Measure C_type, C_table, D_reason   │  │
           │  └────────────────────────────────────────┘  │
           └──────────────────┬───────────────────────────┘
                              │
                              ▼
           ┌──────────────────────────────────────────────┐
           │         POLICY UPDATE (choose one)           │
           │  ┌─────────┐  ┌─────────┐  ┌─────────────┐  │
           │  │  GRPO   │  │   NSR   │  │ W-REINFORCE │  │
           │  └─────────┘  └─────────┘  └─────────────┘  │
           └──────────────────┬───────────────────────────┘
                              │
                              ▼
           ┌──────────────────────────────────────────────┐
           │              UPDATE MODEL                     │
           │         (gradient descent on LoRA)            │
           └──────────────────────────────────────────────┘
```

---

## 2. Input Format

### 2.1 What We Feed to the LLM

Each training sample consists of:

```python
{
    "image": PIL.Image,           # Chart image (bar, line, pie, etc.)
    "query": str,                 # Question about the chart
    "label": str,                 # Ground truth answer
    "chart_type": str,            # e.g., "bar", "line", "pie"
    "table": dict,                # Underlying data table
    "reasoning": str,             # Ground truth reasoning steps
}
```

### 2.2 Prompt Construction

The prompt is built as a conversation:

```python
conversation = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT  # See below
    },
    {
        "role": "user",
        "content": [
            {"type": "image"},              # The chart image
            {"type": "text", "text": query} # The question
        ]
    }
]
```

### 2.3 System Prompt

```
You are an expert in understanding and reasoning over charts. Given a chart
image and a question, your task is to provide accurate answers with detailed
reasoning.

Your response MUST follow this exact format:

<think>
<type>chart_type</type>
<table>{"columns": [...], "rows": [...]}</table>
Step 1: [First reasoning step]
Step 2: [Second reasoning step]
...
</think>
<answer>your final answer</answer>

Guidelines:
1. First identify the chart type (bar, line, pie, scatter, etc.)
2. Extract the data table from the chart in JSON format
3. Show your reasoning step by step
4. Provide a concise final answer

Important:
- The table must be valid JSON with "columns" and "rows" keys
- Each reasoning step should reference actual values from the table
- The final answer should be precise and match the question format
```

---

## 3. Model Output Format

### 3.1 Expected Output Structure

```xml
<think>
<type>bar</type>
<table>{"columns": ["Year", "Sales"], "rows": [["2020", 100], ["2021", 150], ["2022", 200]]}</table>
Step 1: Looking at the chart, I need to find the total sales for all years.
Step 2: From the extracted table: 2020 = 100, 2021 = 150, 2022 = 200.
Step 3: Total = 100 + 150 + 200 = 450.
</think>
<answer>450</answer>
```

### 3.2 Parsed Components

After parsing with `parse_response()`:

```python
{
    "type": "bar",
    "table": {
        "columns": ["Year", "Sales"],
        "rows": [["2020", 100], ["2021", 150], ["2022", 200]]
    },
    "reasoning": "Step 1: Looking at the chart... Step 2: From the extracted table... Step 3: Total = 100 + 150 + 200 = 450.",
    "answer": "450"
}
```

---

## 4. Rollout Generation

### 4.1 Process

For each training sample, we generate **K = 8 rollouts** (configurable):

```python
rollouts = []
for i in range(K):
    output = model.generate(
        prompt,
        temperature=0.8,      # Encourages diversity
        top_p=0.95,
        max_new_tokens=768,
    )
    rollouts.append(output)
```

### 4.2 Why Multiple Rollouts?

1. **Diversity**: Different rollouts may use different reasoning strategies
2. **GRPO Requirement**: Need multiple samples to compute relative advantages
3. **HCPC Requirement**: Need multiple correct rollouts to measure consistency/diversity

### 4.3 Example: 8 Rollouts for One Question

```
Question: "What is the total sales from 2020 to 2022?"
Ground Truth: 450

Rollout 1: type=bar, table=correct, reasoning="100+150+200=450", answer=450 ✓
Rollout 2: type=bar, table=correct, reasoning="Sum all: 450", answer=450 ✓
Rollout 3: type=bar, table=correct, reasoning="Adding up: 450", answer=450 ✓
Rollout 4: type=bar, table=WRONG,   reasoning="80+160+210=450", answer=450 ✓ (lucky!)
Rollout 5: type=line, table=correct, reasoning="100+150+200=450", answer=450 ✓ (wrong type)
Rollout 6: type=bar, table=correct, reasoning="100+150=250", answer=250 ✗
Rollout 7: type=bar, table=correct, reasoning="Just 2022: 200", answer=200 ✗
Rollout 8: type=bar, table=correct, reasoning="Average: 150", answer=150 ✗
```

---

## 5. Reward Calculation

### 5.1 Overview

```
Total Reward = Base Rewards + HCPC Reward (if enabled)

Base Rewards (per rollout):
├── Format Reward (0-2)
├── Accuracy Reward (0-1)
├── Length Reward (0-2)
├── Token Count Reward (0-2)
├── Chart Type Reward (0-1)
├── Table Reward (0-2)
└── Process Reward (0-1)

HCPC Reward (group-level, same for all rollouts):
└── HCPC Score (0 to ~4.5)
```

### 5.2 Base Rewards (from Chart-RVR)

#### 5.2.1 Format Reward (0 or 2)

Checks if output follows the exact tag structure:

```python
def format_reward(completion):
    pattern = r"<think>.*?<type>.*?</type>.*?<table>.*?</table>.*?</think>.*?<answer>.*?</answer>"
    if re.search(pattern, completion, re.DOTALL):
        return 2.0
    return 0.0
```

**Examples:**
```
✓ "<think><type>bar</type><table>{...}</table>...</think><answer>450</answer>" → 2.0
✗ "<think>bar chart, table: {...}, answer: 450</think>" → 0.0 (missing tags)
✗ "<answer>450</answer><think>...</think>" → 0.0 (wrong order)
```

#### 5.2.2 Accuracy Reward (0 or 1)

Compares predicted answer to ground truth:

```python
def accuracy_reward(completion, label, tolerance=0.05):
    pred_answer = extract_answer(completion)  # Get from <answer> tags

    # Try numeric comparison (5% tolerance)
    pred_num = try_parse_numeric(pred_answer)
    label_num = try_parse_numeric(label)

    if both_numeric:
        if |pred_num - label_num| / |label_num| <= 0.05:
            return 1.0

    # Fall back to string match
    if normalize(pred_answer) == normalize(label):
        return 1.0

    return 0.0
```

**Examples:**
```
Label: "450"
├── Prediction: "450" → 1.0 (exact match)
├── Prediction: "452" → 1.0 (within 5%: |452-450|/450 = 0.4%)
├── Prediction: "500" → 0.0 (outside 5%: |500-450|/450 = 11%)
└── Prediction: "four fifty" → 0.0 (string doesn't match)
```

#### 5.2.3 Length Reward (0 to ~2)

Rewards appropriate reasoning length:

```python
def length_reward(completion, min_length=70, max_length=250):
    reasoning = extract_reasoning(completion)
    length = len(reasoning)

    reward = 0.0

    # Base reward for meeting minimum
    if length >= min_length:
        reward = 1.0
    else:
        reward = length / min_length  # Partial credit

    # Penalty for excessive length
    if length > max_length:
        reward -= 0.5

    # Bonus for step markers
    steps = count_steps(reasoning)  # Count "Step 1:", "Step 2:", etc.
    step_bonus = min(0.25 * steps, 1.0)
    reward += step_bonus

    return max(0.0, reward)
```

#### 5.2.4 Token Count Reward (0 or 2)

Ensures exactly one of each tag:

```python
def token_count_reward(completion):
    tags = ["<think>", "</think>", "<answer>", "</answer>",
            "<type>", "</type>", "<table>", "</table>"]

    for tag in tags:
        if completion.count(tag) != 1:
            return 0.0

    # Check ordering
    if not re.search(r"<think>\s*\n?\s*<type>", completion):
        return 0.0

    return 2.0
```

#### 5.2.5 Chart Type Reward (0 or 1)

```python
def chart_type_reward(completion, ground_truth_type):
    pred_type = extract_type(completion).lower()
    gt_type = ground_truth_type.lower()

    return 1.0 if pred_type == gt_type else 0.0
```

#### 5.2.6 Table Reward (0 to 2)

```python
def table_reward(completion, ground_truth_table):
    pred_table = extract_and_parse_table(completion)

    if not pred_table:
        return 0.0

    reward = 0.5  # Valid JSON

    # Column matching
    gt_cols = set(ground_truth_table["columns"])
    pred_cols = set(pred_table.get("columns", []))
    col_overlap = len(gt_cols & pred_cols) / len(gt_cols)
    reward += col_overlap * 0.5

    # Row matching
    row_score = compare_rows(gt_table["rows"], pred_table["rows"])
    reward += row_score  # 0 to 1

    return min(reward, 2.0)
```

#### 5.2.7 Process Reward (0 to 1)

Uses sentence embeddings to compare reasoning:

```python
def process_reward(completion, ground_truth_reasoning):
    pred_reasoning = extract_reasoning(completion)

    # Compute semantic similarity
    similarity = cosine_similarity(
        embed(pred_reasoning),
        embed(ground_truth_reasoning)
    )

    return similarity  # 0 to 1
```

### 5.3 HCPC Reward (Hierarchical Correct-Path Consistency)

#### 5.3.1 Concept

HCPC measures **cross-rollout properties** among **ground-truth-correct** rollouts:

```
Chart Reasoning Hierarchy:
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Chart Type  │ --> │   Table     │ --> │  Reasoning  │ --> │   Answer    │
│             │     │ Extraction  │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
       ↓                   ↓                   ↓                   ↓
   CONSISTENT          CONSISTENT           DIVERSE           CONSISTENT
   (all same)          (all same)        (different ok)       (all same)
```

#### 5.3.2 Step-by-Step Calculation

**Step 1: Filter to Fully Correct Rollouts**

A rollout is "fully correct" if:
- Chart type matches ground truth
- Table similarity >= 0.8
- Answer matches ground truth

```python
def filter_correct(rollouts, ground_truth):
    correct = []
    for r in rollouts:
        parsed = parse_response(r)

        # Check type
        if parsed["type"] != ground_truth["chart_type"]:
            continue

        # Check table similarity
        if table_similarity(parsed["table"], ground_truth["table"]) < 0.8:
            continue

        # Check answer
        if not answers_match(parsed["answer"], ground_truth["label"]):
            continue

        correct.append(parsed)

    return correct
```

**Example Filtering:**
```
8 rollouts generated
├── Rollout 1: type=bar ✓, table=0.95 ✓, answer=450 ✓ → CORRECT
├── Rollout 2: type=bar ✓, table=0.92 ✓, answer=450 ✓ → CORRECT
├── Rollout 3: type=bar ✓, table=0.88 ✓, answer=450 ✓ → CORRECT
├── Rollout 4: type=bar ✓, table=0.65 ✗, answer=450 ✓ → REJECTED (bad table)
├── Rollout 5: type=line ✗, table=0.95 ✓, answer=450 ✓ → REJECTED (wrong type)
├── Rollout 6: type=bar ✓, table=0.90 ✓, answer=250 ✗ → REJECTED (wrong answer)
├── Rollout 7: type=bar ✓, table=0.85 ✓, answer=200 ✗ → REJECTED (wrong answer)
└── Rollout 8: type=bar ✓, table=0.91 ✓, answer=150 ✗ → REJECTED (wrong answer)

Result: 3 fully correct rollouts
```

**Step 2: Compute C_type (Type Consistency)**

```python
def compute_type_consistency(correct_rollouts):
    types = [r["type"] for r in correct_rollouts]

    # Find most common type
    from collections import Counter
    type_counts = Counter(types)
    modal_type, modal_count = type_counts.most_common(1)[0]

    # Fraction matching modal type
    return modal_count / len(types)
```

**Example:**
```
Correct rollouts: [type=bar, type=bar, type=bar]
Modal type: bar (3 occurrences)
C_type = 3/3 = 1.0 (perfect consistency)
```

**Step 3: Compute C_table (Table Consistency)**

```python
def compute_table_consistency(correct_rollouts):
    tables = [r["table"] for r in correct_rollouts]

    # Compute pairwise similarity
    similarities = []
    for i in range(len(tables)):
        for j in range(i + 1, len(tables)):
            sim = table_similarity(tables[i], tables[j])
            similarities.append(sim)

    return mean(similarities)
```

**Example:**
```
3 correct rollouts with tables T1, T2, T3
Pairwise similarities:
  sim(T1, T2) = 0.95
  sim(T1, T3) = 0.92
  sim(T2, T3) = 0.98

C_table = (0.95 + 0.92 + 0.98) / 3 = 0.95
```

**Step 4: Compute D_reason (Reasoning Diversity)**

```python
def compute_reasoning_diversity(correct_rollouts):
    reasonings = [r["reasoning"] for r in correct_rollouts]

    # Compute pairwise similarity
    similarities = []
    for i in range(len(reasonings)):
        for j in range(i + 1, len(reasonings)):
            sim = sentence_similarity(reasonings[i], reasonings[j])
            similarities.append(sim)

    avg_similarity = mean(similarities)

    # Diversity = 1 - similarity
    return 1.0 - avg_similarity
```

**Example:**
```
3 correct rollouts with reasoning R1, R2, R3:
  R1: "From table: 100+150+200 = 450"
  R2: "Sum of all years: 100, 150, 200 → 450"
  R3: "Adding: 100+150+200 = 450"

Pairwise similarities:
  sim(R1, R2) = 0.75
  sim(R1, R3) = 0.85
  sim(R2, R3) = 0.70

avg_similarity = 0.767
D_reason = 1 - 0.767 = 0.233 (moderate diversity)
```

**Step 5: Combine into HCPC Reward**

```python
def compute_hcpc_reward(rollouts, ground_truth, weights):
    correct = filter_correct(rollouts, ground_truth)

    if len(correct) < 2:
        return 0.0  # Need at least 2 for cross-rollout metrics

    c_type = compute_type_consistency(correct)
    c_table = compute_table_consistency(correct)
    d_reason = compute_reasoning_diversity(correct)

    correct_rate = len(correct) / len(rollouts)

    # Weighted combination
    # w_type=1.0, w_table=2.0, w_reason=1.5 (defaults)
    weighted_sum = (
        weights.w_type * c_type +      # 1.0 * c_type
        weights.w_table * c_table +    # 2.0 * c_table
        weights.w_reason * d_reason    # 1.5 * d_reason
    )

    return correct_rate * weighted_sum
```

**Example Calculation:**
```
correct_rate = 3/8 = 0.375
c_type = 1.0
c_table = 0.95
d_reason = 0.233

weighted_sum = 1.0 * 1.0 + 2.0 * 0.95 + 1.5 * 0.233
            = 1.0 + 1.9 + 0.35
            = 3.25

HCPC_reward = 0.375 * 3.25 = 1.22
```

### 5.4 Total Reward Computation

```python
def compute_total_rewards(rollouts, ground_truth, config):
    total_rewards = []

    # Compute base rewards for each rollout
    for rollout in rollouts:
        base = (
            format_reward(rollout) +           # 0-2
            accuracy_reward(rollout, gt) +     # 0-1
            length_reward(rollout) +           # 0-2
            token_count_reward(rollout) +      # 0-2
            chart_type_reward(rollout, gt) +   # 0-1
            table_reward(rollout, gt) +        # 0-2
            process_reward(rollout, gt)        # 0-1
        )
        total_rewards.append(base)

    # Add HCPC (if enabled) - same value for all rollouts
    if config.use_hcpc:
        hcpc = compute_hcpc_reward(rollouts, ground_truth, config)
        total_rewards = [r + hcpc for r in total_rewards]

    return total_rewards
```

**Example: Full Reward Calculation for 8 Rollouts**

```
Rollout 1 (correct, good reasoning):
  format=2 + acc=1 + len=1.5 + token=2 + type=1 + table=1.8 + process=0.7
  Base = 10.0
  + HCPC = 1.22
  Total = 11.22

Rollout 2 (correct, shorter reasoning):
  format=2 + acc=1 + len=1.0 + token=2 + type=1 + table=1.7 + process=0.6
  Base = 9.3
  + HCPC = 1.22
  Total = 10.52

Rollout 3 (correct):
  Base = 9.8 + HCPC = 1.22 → Total = 11.02

Rollout 4 (correct answer, bad table):
  format=2 + acc=1 + len=1.2 + token=2 + type=1 + table=0.5 + process=0.5
  Base = 8.2
  + HCPC = 1.22
  Total = 9.42

Rollout 5 (wrong type):
  format=2 + acc=1 + len=1.3 + token=2 + type=0 + table=1.6 + process=0.6
  Base = 8.5
  + HCPC = 1.22
  Total = 9.72

Rollout 6 (wrong answer):
  format=2 + acc=0 + len=0.8 + token=2 + type=1 + table=1.5 + process=0.4
  Base = 7.7
  + HCPC = 1.22
  Total = 8.92

Rollout 7 (wrong answer):
  Base = 7.5 + HCPC = 1.22 → Total = 8.72

Rollout 8 (wrong answer):
  Base = 7.3 + HCPC = 1.22 → Total = 8.52

Final rewards: [11.22, 10.52, 11.02, 9.42, 9.72, 8.92, 8.72, 8.52]
```

---

## 6. Policy Update Methods

### 6.1 Overview: From Rewards to Gradient Updates

All methods follow this pattern:
1. Compute rewards for all K rollouts
2. Convert rewards to **advantages**
3. Use advantages to weight the policy gradient

```python
# Policy gradient update (simplified)
for rollout, advantage in zip(rollouts, advantages):
    log_prob = model.log_probability(rollout)
    loss = -advantage * log_prob  # Negative because we maximize
    loss.backward()
```

The key difference between methods is **how advantages are computed**.

### 6.2 GRPO (Group Relative Policy Optimization)

**Concept:** Reinforce above-average rollouts, penalize below-average.

```python
def grpo_advantages(rewards):
    mean_reward = sum(rewards) / len(rewards)
    std_reward = std(rewards)

    advantages = []
    for r in rewards:
        adv = (r - mean_reward) / max(std_reward, 1.0)
        advantages.append(adv)

    return advantages
```

**Example:**
```
Rewards: [11.22, 10.52, 11.02, 9.42, 9.72, 8.92, 8.72, 8.52]
Mean: 9.76
Std: 1.03

Advantages (normalized):
  Rollout 1: (11.22 - 9.76) / 1.03 = +1.42  ← Strong positive
  Rollout 2: (10.52 - 9.76) / 1.03 = +0.74  ← Positive
  Rollout 3: (11.02 - 9.76) / 1.03 = +1.22  ← Positive
  Rollout 4: (9.42 - 9.76) / 1.03 = -0.33   ← Slight negative
  Rollout 5: (9.72 - 9.76) / 1.03 = -0.04   ← Near zero
  Rollout 6: (8.92 - 9.76) / 1.03 = -0.82   ← Negative
  Rollout 7: (8.72 - 9.76) / 1.03 = -1.01   ← Negative
  Rollout 8: (8.52 - 9.76) / 1.03 = -1.20   ← Strong negative
```

**Effect:**
- Rollouts 1, 2, 3 (best) get **increased probability**
- Rollouts 6, 7, 8 (worst) get **decreased probability**
- Model converges toward the "winning" strategies

**Problem:** Over time, this causes **reasoning collapse** - the model learns to produce only one type of correct response, losing diversity.

### 6.3 NSR (Negative Sample Reinforcement)

**Concept:** Only penalize wrong responses, don't reinforce correct ones.

```python
def nsr_advantages(rewards, threshold):
    # threshold = 0.5 * max_possible_reward ≈ 5.5
    advantages = []
    for r in rewards:
        if r >= threshold:
            adv = 0.0  # Skip correct samples (no gradient)
        else:
            adv = -(threshold - r)  # Penalize wrong samples
        advantages.append(adv)

    return advantages
```

**Example:**
```
Rewards: [11.22, 10.52, 11.02, 9.42, 9.72, 8.92, 8.72, 8.52]
Threshold: 5.5 (50% of max ~11)

All rewards > 5.5, so all are "correct":
Advantages: [0, 0, 0, 0, 0, 0, 0, 0]

But if threshold was 9.0:
Advantages:
  Rollout 1: 11.22 >= 9.0 → 0.0 (skip)
  Rollout 2: 10.52 >= 9.0 → 0.0 (skip)
  Rollout 3: 11.02 >= 9.0 → 0.0 (skip)
  Rollout 4: 9.42 >= 9.0 → 0.0 (skip)
  Rollout 5: 9.72 >= 9.0 → 0.0 (skip)
  Rollout 6: 8.92 < 9.0 → -(9.0 - 8.92) = -0.08
  Rollout 7: 8.72 < 9.0 → -(9.0 - 8.72) = -0.28
  Rollout 8: 8.52 < 9.0 → -(9.0 - 8.52) = -0.48
```

**Effect:**
- **Correct rollouts get zero gradient** - their probabilities don't change
- **Wrong rollouts get penalized** - their probabilities decrease
- The probability mass **redistributes naturally** across all correct strategies
- **Preserves diversity** - doesn't push toward any single correct answer

**Why it works (from NSR paper):**
> "By not reinforcing correct answers, NSR doesn't push toward any single strategy. It only pushes AWAY from wrong strategies, allowing probability mass to redistribute naturally across all valid approaches."

### 6.4 W-REINFORCE (Weighted REINFORCE)

**Concept:** Weak positive + strong negative = balanced

```python
def w_reinforce_advantages(rewards, threshold, lambda_psr=0.1):
    advantages = []
    for r in rewards:
        if r >= threshold:
            adv = lambda_psr * r  # WEAK positive (10% weight)
        else:
            adv = -(threshold - r)  # STRONG negative (100% weight)
        advantages.append(adv)

    return advantages
```

**Example (threshold=9.0, lambda=0.1):**
```
Rewards: [11.22, 10.52, 11.02, 9.42, 9.72, 8.92, 8.72, 8.52]

Advantages:
  Rollout 1: 11.22 >= 9.0 → 0.1 * 11.22 = +1.12 (weak positive)
  Rollout 2: 10.52 >= 9.0 → 0.1 * 10.52 = +1.05 (weak positive)
  Rollout 3: 11.02 >= 9.0 → 0.1 * 11.02 = +1.10 (weak positive)
  Rollout 4: 9.42 >= 9.0 → 0.1 * 9.42 = +0.94 (weak positive)
  Rollout 5: 9.72 >= 9.0 → 0.1 * 9.72 = +0.97 (weak positive)
  Rollout 6: 8.92 < 9.0 → -(9.0 - 8.92) = -0.08 (negative)
  Rollout 7: 8.72 < 9.0 → -(9.0 - 8.72) = -0.28 (negative)
  Rollout 8: 8.52 < 9.0 → -(9.0 - 8.52) = -0.48 (negative)
```

**Effect:**
- Correct samples get a **small boost** (maintains accuracy)
- Wrong samples get **full penalty** (like NSR)
- **Best of both worlds**: accuracy + diversity

### 6.5 Comparison Summary

```
┌─────────────────┬───────────────────┬───────────────────┬──────────────────┐
│                 │       GRPO        │        NSR        │   W-REINFORCE    │
├─────────────────┼───────────────────┼───────────────────┼──────────────────┤
│ Correct samples │ Strong positive   │ Zero (skip)       │ Weak positive    │
│                 │ (r - mean)        │                   │ (0.1 × r)        │
├─────────────────┼───────────────────┼───────────────────┼──────────────────┤
│ Wrong samples   │ Negative          │ Negative          │ Negative         │
│                 │ (r - mean < 0)    │ -(threshold - r)  │ -(threshold - r) │
├─────────────────┼───────────────────┼───────────────────┼──────────────────┤
│ Effect          │ Converges to best │ Pushes away from  │ Slight push to   │
│                 │ single strategy   │ wrong, preserves  │ correct + strong │
│                 │                   │ all correct       │ push from wrong  │
├─────────────────┼───────────────────┼───────────────────┼──────────────────┤
│ Diversity       │ LOW (collapses)   │ HIGH (preserved)  │ MEDIUM           │
├─────────────────┼───────────────────┼───────────────────┼──────────────────┤
│ Pass@1          │ High              │ Slightly lower    │ High             │
├─────────────────┼───────────────────┼───────────────────┼──────────────────┤
│ Pass@k (k>1)    │ Degrades          │ Maintains/improves│ Good balance     │
└─────────────────┴───────────────────┴───────────────────┴──────────────────┘
```

---

## 7. Complete Training Loop

```python
def training_loop(model, dataset, config):
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    for epoch in range(config.num_epochs):
        for batch in dataset:
            # 1. Extract inputs
            images = batch["images"]
            questions = batch["queries"]
            ground_truths = {
                "label": batch["labels"],
                "chart_type": batch["chart_types"],
                "table": batch["tables"],
                "reasoning": batch["reasonings"],
            }

            # 2. Generate K rollouts per sample
            all_rollouts = []
            for img, q in zip(images, questions):
                prompt = format_prompt(img, q)
                rollouts = model.generate(prompt, n=config.num_generations)
                all_rollouts.append(rollouts)

            # 3. Compute rewards
            all_rewards = []
            for rollouts, gt in zip(all_rollouts, ground_truths):
                rewards = compute_total_rewards(rollouts, gt, config)
                all_rewards.append(rewards)

            # 4. Compute advantages based on policy method
            all_advantages = []
            for rewards in all_rewards:
                if config.policy_method == "grpo":
                    advantages = grpo_advantages(rewards)
                elif config.policy_method == "nsr":
                    advantages = nsr_advantages(rewards, config.reward_threshold)
                elif config.policy_method == "w_reinforce":
                    advantages = w_reinforce_advantages(
                        rewards, config.reward_threshold, config.lambda_psr
                    )
                all_advantages.append(advantages)

            # 5. Compute policy gradient loss
            loss = 0
            for rollouts, advantages in zip(all_rollouts, all_advantages):
                for rollout, adv in zip(rollouts, advantages):
                    log_prob = model.log_probability(rollout)
                    loss -= adv * log_prob  # Negative for gradient ascent

            # 6. Update model
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # 7. Save checkpoint if needed
            if step % config.checkpoint.save_every_n_steps == 0:
                save_checkpoint(model, optimizer, step)
```

---

## 8. Experiment Configurations

### 8.1 The 6 Experiments

| # | Name | Policy | HCPC | Description |
|---|------|--------|------|-------------|
| 1 | `grpo_baseline` | GRPO | OFF | Chart-RVR reproduction |
| 2 | `grpo_hcpc` | GRPO | ON | GRPO + structured diversity |
| 3 | `nsr_baseline` | NSR | OFF | Pure NSR |
| 4 | `nsr_hcpc` | NSR | ON | NSR + structured diversity |
| 5 | `w_reinforce_baseline` | W-REINFORCE | OFF | Balanced baseline |
| 6 | `w_reinforce_hcpc` | W-REINFORCE | ON | **Full HCPC-RLVR** |

### 8.2 Expected Outcomes

```
                    │ In-Domain (ChartQA) │ Out-of-Domain (EvoChart) │ OOD Gap │
────────────────────┼─────────────────────┼──────────────────────────┼─────────┤
grpo_baseline       │ ~84%                │ ~53%                     │ ~31%    │
grpo_hcpc           │ ~85%                │ ~56%                     │ ~29%    │
nsr_baseline        │ ~84%                │ ~56%                     │ ~28%    │
nsr_hcpc            │ ~85%                │ ~58%                     │ ~27%    │
w_reinforce_baseline│ ~85%                │ ~57%                     │ ~28%    │
w_reinforce_hcpc    │ ~86%                │ ~60%                     │ ~26%    │
```

**Key predictions:**
1. NSR and W-REINFORCE improve OOD over GRPO (diversity preservation)
2. HCPC adds ~2-3% on OOD across all methods (structured diversity)
3. W-REINFORCE + HCPC achieves best balance (your hypothesis)

---

## Appendix: Glossary

| Term | Definition |
|------|------------|
| **Rollout** | One generated response from the model |
| **K** | Number of rollouts per sample (default 8) |
| **Advantage** | Weight for policy gradient (positive = reinforce, negative = penalize) |
| **GRPO** | Group Relative Policy Optimization - standard RL method |
| **NSR** | Negative Sample Reinforcement - only penalize wrong |
| **W-REINFORCE** | Weighted REINFORCE - weak positive + strong negative |
| **HCPC** | Hierarchical Correct-Path Consistency - cross-rollout reward |
| **C_type** | Type consistency among correct rollouts |
| **C_table** | Table consistency among correct rollouts |
| **D_reason** | Reasoning diversity among correct rollouts |
| **Pass@k** | Probability of at least one correct answer in k attempts |
| **OOD Gap** | In-domain accuracy minus out-of-domain accuracy |
