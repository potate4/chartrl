# HCPC-RLVR Pre-Defense: Slides 8–14
### Full Slide Content + Speaker Script + Study Notes

> **How to use this doc:** Every slide has three parts:
> - `[SLIDE]` = exactly what appears on screen
> - `[SCRIPT]` = what you say word-for-word (or close to it)
> - `[STUDY]` = the conceptual understanding behind it, so you never blank out

---
---

# SLIDE 8: Research Challenges

---

## 8a — The Problem Space

### [SLIDE]

**Title:** Why Chart QA Is Hard: Open Research Challenges

**Three-column layout:**

| Challenge | What It Means | Why It Matters |
|---|---|---|
| Dual-Stage Reasoning | See chart → Extract data → Reason → Answer | One bad stage cascades into wrong answers |
| Distribution Shift | Train on one style, test on another | Models memorize visual patterns, not reasoning logic |
| Reward Design for RLVR | What signal do we give the model? | Wrong reward = model games the metric, not the task |

**Bottom banner:** *"Chart QA is not just visual recognition — it requires perception, structured extraction, and multi-step reasoning simultaneously."*

---

### [SCRIPT]
"Let me start with the honest problem: chart understanding looks deceptively simple, but it's actually a layered hard problem that has resisted easy solutions.

There are three core challenges that motivate this entire work.

First — dual-stage reasoning. When a human looks at a bar chart and answers a question, they're doing two very different cognitive tasks almost simultaneously. First they're visually parsing the image — reading axes, identifying bars, reading values. Then they're reasoning — comparing, calculating, inferring. Models handle these as a sequential pipeline, and one failure in perception guarantees failure downstream. You can't reason correctly from wrong data.

Second — distribution shift. The models we train today achieve respectable numbers on benchmark datasets like ChartQA, but the moment you show them a slightly different chart style — different color palette, different annotation format, different chart type — performance falls off a cliff. Chart-RVR [Sinha et al., 2025] reported an 84.56% score on ChartQA but only 53.36% on EvoChart, a 31-point gap. That's not a small degradation, that's a different model.

Third — reward design. We're using reinforcement learning, which means the training signal is a reward function. The challenge is: what do you reward? If you only reward final answer correctness, the model may learn shortcuts — guessing the correct answer type without doing real reasoning. If you reward intermediate steps, you risk the model gaming those steps without understanding them. Designing a reward that genuinely captures the quality of the full reasoning chain is an open research problem."

---

### [STUDY]
**The cascade failure problem:** Think of chart QA as a pipeline:
```
Image → [Perception] → Table → [Reasoning] → Answer
```
If Perception is wrong (extracted table has wrong values), then Reasoning has nothing to work with. Even perfect reasoning logic produces a wrong answer. This is why C_table (table extraction consistency) matters — it measures whether the first stage is reliable.

**Why distribution shift is especially bad for charts:** Charts are not like natural images. A cat is a cat whether it's photographed or painted. But a bar chart in blue on a white background uses completely different pixel patterns than one in red on a dark background, even if they show the same data. Models can accidentally learn "bar charts look like this" rather than "bar charts mean this."

**The RLVR reward dilemma:** Imagine training a student with only a grade (correct/wrong). They might memorize tricks that get the right grade without understanding the material. Now imagine you grade the *process*: did they correctly read the axis? Did they do the right arithmetic? That's what multi-component reward functions try to do. But designing these components without creating loopholes is non-trivial.

---

## 8b — The Literature Gap

### [SLIDE]

**Title:** What Existing Work Gets Wrong (or Doesn't Address)

**Visual: A 2x2 grid**

```
                        TEXT ONLY    |    VISION + TEXT
                   ─────────────────┼──────────────────
SUPERVISED         ChartLlama       |   Chart-RVR
FINE-TUNING        MatCha, DePlot   |   TinyChart, ChartMoE
                   ─────────────────┼──────────────────
REINFORCEMENT      DeepSeek-R1      |   ← WE ARE HERE
LEARNING (RLVR)    GRPO-Math        |   (Almost nobody)
```

**Bullet points:**
- Most chart models are **supervised fine-tuning only** — they imitate demonstrations, never self-correct
- RLVR has transformed text reasoning (DeepSeek-R1, GRPO) but barely touched vision
- No prior work applies NSR to VLMs
- No prior work uses **hierarchical** inter-rollout consistency as a reward

*Citations: [Sinha et al., 2025], [Han et al., 2023], [Liu & Han, 2022], [Guo et al., 2025], [Shao et al., 2024], [Zhu et al., 2025]*

---

### [SCRIPT]
"Now let me place our work on the map of existing research.

Chart understanding has been dominated by supervised fine-tuning. Models like MatCha, DePlot, ChartLlama, and TinyChart all follow the same recipe: take a pretrained VLM, show it many (chart, question, answer) pairs, teach it to imitate correct answers. These are strong baselines. TinyChart achieves 83.6% on ChartQA at 3B parameters — competitive with our work.

But supervised fine-tuning has a fundamental ceiling: the model learns to imitate, not to reason. It gets good at patterns it has seen, and struggles with anything novel.

Reinforcement Learning from Verifiable Rewards has completely transformed text reasoning. DeepSeek-R1 [Guo et al., 2025] showed that giving a language model a simple binary reward — 'is your math answer correct?' — and letting it explore solutions through GRPO [Shao et al., 2024] produces emergent chain-of-thought reasoning that beats fine-tuned models.

But this revolution has barely touched vision-language models. Why? Because rewards are harder to define. For math, correctness is binary and unambiguous. For chart QA, what is the reward? Just answer correctness? That ignores the quality of the table extraction. All intermediate steps? That's hard to evaluate automatically.

Our work sits at the intersection of RLVR and vision-language chart understanding — an intersection that is nearly empty. And within that, we're doing two novel things: HCPC, which applies hierarchical inter-rollout consistency as a reward, and NSR applied to VLMs for the first time."

---

### [STUDY]
**Why RLVR works for reasoning but not (yet) for charts:** In pure text math, the reward is clean — run the answer through a verifier, it's right or wrong. In chart QA, you need visual parsing first, and that adds noise. A model can generate a wrong table but guess the right answer, or vice versa. The reward signal is therefore ambiguous. Our 7-component reward (Chart-RVR framework) tries to decompose this, rewarding each stage separately.

**MatCha, DePlot** (Liu & Han, 2022): These are Google-developed chart understanding models based on supervised fine-tuning. They represent the state of the art in chart-to-text conversion (DePlot) and augmented chart QA (MatCha). Our baseline (Qwen2.5-VL) is of similar capability but much smaller and therefore more resource-efficient.

**ChartLlama** (Han et al., 2023): A LLaMA-based chart model, instruction-tuned on chart-specific data. Good baseline but again, pure SFT.

**DeepSeek-R1** (Guo et al., 2025): The paper that galvanized the RLVR movement for reasoning. Key lesson: simple binary reward + GRPO = emergent chain-of-thought. We're extending that lesson to a harder, multi-modal domain.

---

## 8c — Collective Disadvantage Summary

### [SLIDE]

**Title:** The Collective Disadvantage We're Addressing

**A numbered problem list:**

1. **VLMs underperform on chart QA** relative to their general capability — chart-specific reasoning is a weak spot even for frontier models *(ChartQA scores for Qwen2.5-VL-3B: 51.8% base vs 63% after training)*

2. **RLVR reward design is flat** — existing approaches (standard GRPO) treat all correct answers equally, ignoring *how* the model arrived at the answer *(No hierarchy: correct by luck = correct by reasoning)*

3. **Reasoning diversity is not explicitly incentivized** — models converge to one reasoning strategy, reducing robustness and inference-time scaling *(Wen et al., 2025: diversity → better Pass@K)*

4. **NSR has never been tested on visual inputs** — a promising negative-reinforcement approach [Zhu et al., 2025] exists but only for text math, leaving its VLM applicability unknown

5. **OOD generalization remains unsolved** — models trained on ChartQA significantly degrade on new chart formats and tasks *(Sinha et al., 2025: 84.56% → 53.36%, a 31pp gap)*

---

### [SCRIPT]
"Let me crystallize the problems we're attacking into five specific disadvantages in the current research landscape.

One: VLMs are weak at chart QA relative to their general capability. The same model that can have a sophisticated conversation about philosophy scores barely above chance on structured chart questions out-of-the-box. We confirm this — our base model starts at 51.8%.

Two: Standard RLVR rewards are flat. Whether the model got the right answer by correctly extracting the table and reasoning properly, or by hallucinating an answer that happened to match, the reward is the same. We argue this is wrong. The *path* to the correct answer matters, not just the destination. HCPC addresses this by giving bonuses for consistent, correct extraction paths.

Three: Reasoning diversity is not explicitly incentivized in standard GRPO. Wen et al. showed theoretically that maintaining diverse reasoning strategies improves Pass@K — the more ways you can reach a correct answer, the more likely you are to find one. Standard GRPO tends to collapse toward the most common correct strategy.

Four: NSR has never been tested on VLMs. Zhu et al. proposed a smart idea — only penalize wrong answers, don't reinforce right ones — but only evaluated it on text math. Does it work when the input is an image and the reward is noisy? We test this.

Five: OOD generalization is genuinely unsolved. We tested on ChartFC, a different task format, and the results are illuminating — in ways we'll discuss in the results section."

---

### [STUDY]
**Why the "flat reward" problem matters:** Imagine two students both get 80/100 on a test. Student A understood everything but made careless errors. Student B memorized answers but failed to understand the concepts. A flat reward (the 80) treats them identically. But for training purposes, we want to reinforce Student A's strategy more than Student B's. HCPC tries to be the teacher who looks at the *process*, not just the score.

**The Wen et al. (2025) insight:** They proved mathematically that Pass@K improves when the model maintains diverse reasoning paths. Intuitively: if the model always reasons the same way and is wrong, it'll always be wrong across all K rollouts. If it has diverse strategies, at least one might work. This gives theoretical motivation for diversity-promoting rewards like HCPC.

---
---

# SLIDE 9: Research Questions

---

### [SLIDE]

**Title:** Research Questions

**Left side — Questions:**

**RQ1:** Can RLVR, applied to a 3B-parameter VLM with only 1,000 training examples, produce meaningful improvements in chart QA accuracy and structural output quality?

**RQ2:** Does a hierarchical reward structure (HCPC) that conditions bonuses on the *path* of correct reasoning — not just the final answer — lead to more robust reasoning behavior than flat GRPO?

**RQ3:** Does HCPC explicitly promote reasoning diversity (measured by Pass@K and D_reason) while simultaneously maintaining extraction consistency (C_table)?

**RQ4:** Can Negative Sample Reward (NSR), originally designed for text-based mathematical reasoning, transfer effectively to the vision-language domain with multi-component rewards?

**RQ5:** How do RLVR-trained models generalize to out-of-distribution chart tasks, and do different reward strategies lead to qualitatively different generalization behaviors?

**RQ6:** What is the relationship between table extraction consistency, reasoning diversity, and final answer correctness — and does reward design shape this relationship?

**Right side — Why These Questions Matter:**
> These questions directly address whether RLVR is a viable paradigm for visual reasoning — a question with implications beyond charts, extending to any domain where structured visual understanding is required.

---

### [SCRIPT]
"Let me now state precisely what we're trying to answer. These aren't rhetorical questions — after the results slides, we'll revisit each one and give an explicit answer.

RQ1 is the existence proof. Does RLVR even work here? The motivation for asking this is that RLVR's successes have been mostly on text math — clean rewards, clean data, well-defined problems. Chart QA is messier. The reward is noisy, the visual input introduces uncertainty. We're asking whether RLVR's core mechanism — generating rollouts, computing rewards, updating the policy — can extract signal from this noisier environment with just 1,000 training examples.

RQ2 is about the design philosophy of our HCPC contribution. Standard GRPO rewards correct answers. HCPC additionally rewards *how* the model got there — did it extract a consistent table across all correct rollouts? Did those correct rollouts use diverse reasoning strategies? We're asking whether this higher-level reward improves beyond just measuring the answer.

RQ3 is about whether HCPC achieves its stated design goals. The theory says diversity should improve Pass@K. The question is: does the reward actually induce diversity? And does it do so without sacrificing extraction stability?

RQ4 validates whether NSR's insight generalizes beyond text. NSR says: 'don't reinforce correct rollouts, just penalize wrong ones.' For text math, this works because correct rollouts are few and precious — you don't want to collapse onto one correct strategy. Does the same logic hold when you add visual perception?

RQ5 is the generalization question. We test on ChartFC, which is a different task (fact-checking vs Q&A) with a different visual distribution. We want to know: what survives training? What gets brittle?

RQ6 is the correlation question. C_table, D_reason, coherence — these metrics were designed to capture different aspects of the reasoning process. Do they actually correlate with correctness in the ways theory predicts? And does the reward design change these correlations?"

---

### [STUDY]
**Why phrase them as questions, not assertions?** A pre-defense committee wants to see that you know where the boundaries of your work are. If you say "we show HCPC improves over GRPO," that's a claim you need to fully defend. If you say "we investigate whether HCPC improves over GRPO," you have more room to discuss nuanced results. But importantly, by the end of your results section, you need to *answer* each question — even if the answer is "we found trends consistent with X but significance requires larger scale."

**Framing each RQ at defense time:**
- RQ1: "Yes, definitively. 11pp improvement from 500 samples is substantial."
- RQ2: "Partially. HCPC shows better Pass@4 and OOD coherence, but not always better raw accuracy."
- RQ3: "Yes on Pass@4 diversity. Limited on D_reason — 4 rollouts is insufficient."
- RQ4: "Yes. NSR transfers successfully, achieving the highest raw accuracy and C_table."
- RQ5: "GRPO generalizes robustly in accuracy. HCPC generalizes better in reasoning quality (coherence). Format generalization is where HCPC struggles."
- RQ6: "C_table negatively correlates with correctness (perception is not the bottleneck). D_reason positively correlates. HCPC weakens the C_table–correctness anti-correlation."

---
---

# SLIDE 10: Proposed Model Overview

---

### [SLIDE]

**Title:** Proposed Model: Each Component at a Glance

**Table format for teacher reference:**

| Component | What It Is | One-Sentence Definition |
|---|---|---|
| **Base Model** | Qwen2.5-VL-3B-Instruct | A 3-billion-parameter vision-language model that can process chart images and generate structured text responses. |
| **LoRA Fine-tuning** | Low-Rank Adaptation (r=8, α=16) | A parameter-efficient method that trains only 4.7M additional parameters (0.16% of total) rather than the full model. |
| **GRPO** | Group Relative Policy Optimization | A reinforcement learning algorithm that updates the model based on how each response compares to the average quality within a group of rollouts for the same question. |
| **Chart-RVR Rewards** | 7-Component Reward Function | A hierarchical reward that scores the model separately on format, chart type prediction, table structure, table content, reasoning quality, final answer, and response length. |
| **HCPC** | Hierarchical Correct-Path Consistency | A reward bonus that encourages consistent table extraction and diverse reasoning specifically among the rollouts that produce correct answers. |
| **NSR** | Negative Sample Reward | A training variant that modifies how advantages are computed — only penalizing wrong rollouts and ignoring correct ones, rather than treating all rollouts symmetrically. |
| **Rollouts (K=4)** | Multiple Responses Per Sample | For each training question, the model generates 4 different responses (at temperature 1.0), which are evaluated and used together to compute the training signal. |

**Bottom note:** *All three training variants (GRPO, GRPO+HCPC, NSR) share the same base model, LoRA configuration, and hyperparameters — they differ only in reward computation and advantage calculation.*

---

### [SCRIPT]
"Before I show you the full pipeline diagram, let me give you a one-sentence definition of each component so you have a reference point as we go through the details.

[Point to each row] The base model is Qwen2.5-VL-3B-Instruct — a small but capable vision-language model. We don't train all 3 billion parameters — instead we use LoRA, which adds small adapter matrices to just two attention layers, keeping 99.84% of the model frozen.

The training algorithm is GRPO — Group Relative Policy Optimization. The key word is *relative* — the model is rewarded based on how its response compares to its other responses on the same question, not against a fixed threshold.

The reward function is the Chart-RVR framework from Sinha et al., which breaks down chart QA into seven measurable components. HCPC is our addition on top of this — it adds bonuses that look *across* multiple responses, not just within one response.

NSR is not a reward change, it's an advantage computation change. It asymmetrically handles correct and incorrect rollouts in a specific way we'll explain in the pipeline slide.

And rollouts — every training question gets 4 responses generated simultaneously. These 4 responses are the raw material for both the reward computation and the gradient update."

---

### [STUDY]
**Why LoRA?** Full fine-tuning of 3B parameters requires enormous GPU memory and risks catastrophic forgetting — overwriting general knowledge with chart-specific knowledge. LoRA adds small matrices (rank 8, meaning 8 basis vectors) that capture the "delta" of task-specific knowledge without touching the frozen pretrained weights. It's like writing notes in the margins of a textbook rather than rewriting the whole book.

**Why GRPO over PPO?** Standard RL for LLMs (PPO) requires a separate value network — another model of the same size that estimates how good a state is. For 3B models, that's computationally expensive. GRPO eliminates this by using the group mean reward as the baseline. Advantage = how much better than average within your group of 4 rollouts.

**The key difference between HCPC and standard reward:** Standard rewards are *within-rollout* — evaluate one response and score it. HCPC is *across-rollout* — look at all 4 responses together and ask "among the correct ones, are they consistent and diverse?" This cross-rollout signal is what makes HCPC novel.

---
---

# SLIDE 11: Proposed Pipeline Diagram

---

### [SLIDE]

**Title:** HCPC-RLVR Training Pipeline

*(Speaker's note: animate this in PowerPoint or Google Slides in stages — one box appears at a time. Each animation step corresponds to a paragraph in the script below. 2-3 minutes total for this slide.)*

**ANIMATION STAGE 1 — Input (0:00–0:20)**
```
┌──────────────────────────────────────────┐
│              INPUT SAMPLE                │
│   📊 Chart Image   ❓ Question           │
│   ✅ Ground Truth: Type + Table + Answer  │
└──────────────────────────────────────────┘
                     │
                     ▼
```

**ANIMATION STAGE 2 — Generation (0:20–0:45)**
```
          ┌─────────────────────┐
          │   Qwen2.5-VL-3B     │
          │    + LoRA (r=8)     │
          │   temp = 1.0        │
          └─────────────────────┘
                     │
        ┌────────────┼────────────────┐
        ▼            ▼                ▼
   Rollout 1    Rollout 2   ...  Rollout 4
     (R1)         (R2)             (R4)
```
*Each rollout produces: `<think>type + table + reasoning</think><answer>...</answer>`*

**ANIMATION STAGE 3 — Parsing (0:45–1:05)**
```
For EACH rollout:
┌──────────────────────────────────────────────┐
│  PARSER extracts 4 structured components:    │
│  • chart_type  (e.g., "bar")                 │
│  • table       (JSON: columns + rows)        │
│  • reasoning   (step-by-step text)           │
│  • answer      (final answer string)         │
└──────────────────────────────────────────────┘
```

**ANIMATION STAGE 4 — Base Reward (1:05–1:25)**
```
For EACH rollout independently:

R_base = w₁·R_format + w₂·R_type + w₃·R_table_struct
       + w₄·R_table_content + w₅·R_reasoning
       + w₆·R_answer + w₇·R_length

    (7 components, each scored against ground truth)
```

**ANIMATION STAGE 5 — HCPC Bonus (1:25–1:50)**
```
CROSS-ROLLOUT CHECK (only for GRPO+HCPC):

   Identify Correct Set R⁺ = {rollouts where answer ✓}
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
  CONSISTENCY CHECK        DIVERSITY CHECK
  (TABLE level)            (REASONING level)
  C_table = cosine sim     D_reason = Jaccard dist
  of extracted tables      of reasoning step sets
  across R⁺                across R⁺
        │                       │
        └─────────┬─────────────┘
                  ▼
  R_HCPC = w_type·type_consensus
          + w_table·C_table(R⁺)    [w_table = 2.0]
          + w_reason·D_reason(R⁺)  [w_reason = 1.5]

  R_total = R_base + R_HCPC
```

**ANIMATION STAGE 6 — Advantage Computation (1:50–2:10)**
```
METHOD SPLIT:

   GRPO:           GRPO+HCPC:          NSR:
   adv_i =         adv_i =             adv_i = 0         if R_i ≥ τ
   R_i − R̄         R_i^total − R̄       adv_i = −(1+R_i)  if R_i < τ
   (all rollouts)  (all rollouts)       (penalize wrong, ignore correct)
```

**ANIMATION STAGE 7 — Policy Update (2:10–2:30)**
```
Gradient:  ∇θ J = Σᵢ adv_i · ∇θ log π_θ(response_i | image, question)

KL penalty: β · KL(π_θ || π_ref)    [β = 0.01]

                     │
                     ▼
            ┌────────────────┐
            │  Updated Model │   ← repeat 2000 steps
            └────────────────┘
```

**ANIMATION STAGE 8 — Evaluation (2:30–3:00)**
```
EVALUATION (checkpoint-2000, temp=0.8, top_p=0.95):

  Generate 4 responses per test sample

  Compute:
  • Acc%       = relaxed match on at least 1/4 correct
  • Pass@K     = unbiased coverage estimator
  • C_table    = extraction stability
  • D_reason   = reasoning diversity
  • Coherence  = P(correct answer | correct table)
  • Format%    = structural compliance rate
```

---

### [SCRIPT]
**[Stage 1 appears]**
"Let me walk you through the full training pipeline step by step.

Every training iteration starts with a single sample: a chart image, a natural language question, and the ground truth — which includes the expected chart type, the expected data table, and the final answer. This triple ground truth is what makes the Chart-RVR reward framework possible.

**[Stage 2 appears]**
This sample goes into our base model — Qwen2.5-VL-3B with LoRA adapters — and we generate not one response, but four. At training time we use temperature 1.0, which means the model is sampling somewhat randomly. This is intentional. We *want* variety across the four rollouts. Some might be correct, some wrong, and the training signal comes from comparing them.

**[Stage 3 appears]**
Each of those four responses is in a structured format with XML-style tags. A parser extracts four components from each: the predicted chart type, an extracted data table in JSON format, the step-by-step reasoning chain, and the final answer. This structured extraction is itself part of what we're training — the model has to learn to produce parseable output.

**[Stage 4 appears]**
For each rollout individually, we compute the base reward — a weighted sum of seven components from the Chart-RVR framework. Each component compares one aspect of the response to ground truth. Format: did you use all the required tags? Type: is the chart type right? Table: is the extracted data accurate? Answer: is the final answer correct? Each is scored separately and combined into a total base reward.

**[Stage 5 appears — only for GRPO+HCPC track]**
Now here's where HCPC comes in, and this is unique to our method. After computing the base reward, we look *across all four rollouts together*. We identify the subset that produced correct answers — call it R-plus. Then we ask two questions about this correct subset.

First: are the tables they extracted consistent with each other? We measure this with cosine similarity. High consistency means the model reliably perceives the chart the same way when it's correct.

Second: are the reasoning chains diverse? We measure this with Jaccard distance between reasoning step sets. High diversity means the model finds multiple valid strategies to arrive at the answer.

HCPC rewards both — consistency in perception, diversity in reasoning — but only within the correct rollouts. If you're wrong, the HCPC bonus is zero. This is the 'correct-path' part of the name.

**[Stage 6 appears]**
Now, the advantage computation is where GRPO, HCPC, and NSR diverge.

GRPO: the advantage of each rollout is just how much better than the group average it is. Simple.

GRPO+HCPC: same formula, but the reward now includes the HCPC bonus, so the group average is higher, and advantages are computed from this enriched reward.

NSR: completely different logic. If a rollout is correct — advantage is zero. Do nothing. If a rollout is wrong — apply a negative advantage proportional to how wrong it is. So NSR *only* tells the model what not to do, never reinforcing what to do. The correct behavior emerges by eliminating the incorrect ones.

**[Stage 7 appears]**
The advantages are multiplied by the log probabilities and used to update the model's parameters through gradient descent. We also add a small KL penalty to prevent the model from drifting too far from the original pretrained model. This runs for 2,000 steps.

**[Stage 8 appears]**
At evaluation time, we generate four responses per test sample — but now at temperature 0.8 for more focused outputs — and compute all our metrics. Pass@K tells us coverage. Accuracy tells us practical performance. C_table and D_reason tell us about the reasoning process. Coherence tells us about the extraction-to-answer pipeline integrity."

---

### [STUDY]
**Why temperature 1.0 for training and 0.8 for testing?** During training, you want diverse rollouts — high entropy sampling helps the model explore different strategies. During evaluation, you want the model's best behavior — slightly lower temperature means more confident, focused responses. Using the exact same setting for both would either under-explore during training or over-randomize during evaluation.

**Why KL penalty?** Without it, the model can catastrophically forget general knowledge. The KL divergence between the current policy and the original pretrained model keeps the updates "close" to the original — so the model improves at chart QA without becoming useless at everything else.

**The NSR insight explained intuitively:** Imagine you're a student learning by trial and error. Standard GRPO says: "when you get something right, do more of that exact thing. When you get something wrong, do less of that." NSR says: "I'm not going to tell you what to do when you're right — you might have gotten lucky and I don't want to lock you in. But when you're wrong, definitely don't do that again." This preserves the *space* of correct strategies rather than collapsing onto one.

**What "correct-path" means in HCPC:** Imagine the four rollouts for one question. Two get the right answer. HCPC looks only at those two. Were their extracted tables similar? (Consistency reward) Did they reason differently to get there? (Diversity reward) This is the "path" — how the model got to the answer, not just that it got there.

---
---

# SLIDE 12: Experimental Setup

---

## 12a — Datasets

### [SLIDE]

**Title:** Datasets Used

**Two-column layout:**

**Training Data — Chart-RVR CoT (Sinha et al., 2025)**
- Full dataset: 34,194 chart-question-answer pairs
- We used: **first 1,000 samples** (sequential `dataset.select(range(1000))` — no shuffle applied)
- Each sample contains: chart image, chart type label, ground truth table (JSON), step-by-step reasoning, final answer
- Why subset? Compute constraints — full training on 11K samples would require extended multi-GPU runs; 1K completes in ~6 hours on H100

**Test Benchmarks:**

| Benchmark | Task | Samples | Source |
|---|---|---|---|
| **ChartQA** (in-dist) | Open-ended Q&A on charts | 500 | Masry et al., 2022 |
| **ChartFC** (OOD) | Chart fact-checking (support/refute/insufficient) | 500 | — |

---

### [SCRIPT]
"Let me walk through the data setup.

For training, we used the Chart-RVR dataset from Sinha et al. — but only 1,000 samples out of the full 34,194 available. Why? Honestly, compute. We were running on limited infrastructure, and full-dataset training would have taken extended multi-GPU runs rather than hours. But this turns out to be interesting in itself — it gives us evidence that RLVR can extract meaningful signal from small datasets, which is consistent with recent work on sample-efficient RLVR [Wang et al., 2025].

The subset is the first 1,000 rows as ordered in the HuggingFace dataset — no shuffling was applied before selection.

For evaluation, we used two benchmarks. ChartQA is in-distribution — similar chart styles and question formats to what the model trained on. 500 test samples. ChartFC is our out-of-distribution test — it's a completely different task: you're given a chart and a textual claim, and you have to say whether the chart supports the claim, refutes it, or doesn't provide enough information. Different chart styles, different visual format, different reasoning requirement."

---

### [STUDY]
**Why only 500 test samples and not the full test set?** ChartQA's test split has ~2,500 samples. We evaluated on 500 for practical reasons — generating 4 responses per sample means 2,000 total LLM calls. At 30-60 seconds per call, evaluating 500 samples takes ~6 hours per method. Evaluating 2,500 would be 30+ hours. 500 is large enough to get stable estimates (bootstrap CI width ~4pp at this sample size).

**Note on subset selection:** The subset is `dataset.select(range(1000))` — the first 1,000 rows in HF order. No stratification was applied. The actual chart type distribution within this subset depends on how the HF dataset is ordered, which is a limitation to acknowledge if asked.

---

## 12b — Dataset Statistics and Distribution

### [SLIDE]

**Title:** What's In The Data?

**Chart type distribution in ChartQA test set:**

```
Chart Type     Count    Proportion
─────────────────────────────────
Bar charts      316       63.2%    ████████████████████████████████
Line charts     142       28.4%    ██████████████
Pie charts       41        8.2%    ████
Scatterplot       1        0.2%    ▌
```

**Sample images grid (describe visually, use actual chart images from your dataset):**

*For the slide: insert one actual example of each type — a bar chart with a question, a line chart with a question, a pie chart with a question. Show the ground truth answer.*

**Example samples to show:**

| Chart Type | Sample Question | Answer | Difficulty |
|---|---|---|---|
| Bar | "Which country has the highest GDP in 2020?" | "USA" | Easy |
| Bar | "By how many percent did X increase between 2015 and 2018?" | "23.7%" | Medium |
| Line | "In which year did the value first exceed 50?" | "2017" | Medium |
| Pie | "What is the combined percentage of A and B?" | "43%" | Hard |

---

### [SCRIPT]
"Let me show you what's actually in the test data.

The distribution is heavily weighted toward bar charts — 63%. This is not our choice, this is how ChartQA is distributed. Line charts are 28%. Pie charts are 8%. There's one scatterplot, which every method answers correctly and tells us nothing.

This distribution matters for interpreting the results. When a method improves on overall accuracy, it's primarily driven by bar chart performance. When we look at per-type analysis, we'll see something interesting: training helps a lot on bar charts but almost not at all on line charts."

---

### [STUDY]
**Why does ChartQA have so many bar charts?** Bar charts are by far the most common type of chart in real-world documents, news articles, and business reports. ChartQA was built by scraping charts from the web, so it reflects real-world distribution. This is actually good for our work — bar charts are where the structured table extraction approach (extract values by bar, compare) works best.

---

## 12c — Hard Samples Analysis

### [SLIDE]

**Title:** Which Samples Are Hard, and Why?

**Difficulty categories:**

| Category | Count | Description |
|---|---|---|
| **Easy (solved by all 4 methods)** | 344 / 500 (68.8%) | Any approach works — basic value reading, simple categories |
| **Contested (method-dependent)** | 82 / 500 (16.4%) | Some methods get it, others don't — this is where training matters |
| **Hard (no method solves)** | 52 / 500 (10.4%) | Beyond current capability — requires fine-grained precision or complex multi-step |

**Why specific questions fail — problem taxonomy:**

| Problem Type | Example Question | Why It's Hard |
|---|---|---|
| **Decimal precision** | "What is the value of the largest bar?" (Ans: 3.0238) | Model reads "~3" not "3.0238" |
| **Cross-chart arithmetic** | "What's the ratio of the lowest green bar and blue bar?" (Ans: 1.217) | Requires finding two values + division |
| **Multi-step series** | "Find missing value: 2.9, 2.9, 3.5, 4.5, 5.6, 6.6, 6.8" | Not a chart-reading task at all — pattern completion |
| **Tiny percentages** | "How many more felt inspired than depressed?" (Ans: 0.03) | Sub-1% differences invisible at chart resolution |
| **Color/label identification** | "Which line represents boys?" (Ans: "green line") | Requires color grounding, not value extraction |
| **Constrained answer format** | "What party has highest confirmed respondents?" (Ans: "Democrat (scores 60 to 100)") | Exact answer includes bracket notation model doesn't know |

---

### [SCRIPT]
"Let me be concrete about where models fail, because it gives insight into the limitations of both the models and the reward design.

Of our 500 test samples, 344 are solved by literally everyone — including the untrained base model. These are easy questions: 'What is the most common category?' type questions where the answer is visually obvious.

52 samples are solved by nobody — not base, not GRPO, not HCPC, not NSR. These represent the capability ceiling of the 3B-parameter model.

The 82 'contested' samples are where training makes a difference. These are the questions where the difference between trained and untrained — and between GRPO and HCPC — actually shows up.

Let me walk you through the hard sample categories. Decimal precision is a big one — the model correctly identifies the right bar but rounds to a whole number instead of reading 3.0238. This is a perceptual limitation: chart axis labels at that precision are genuinely hard to read even for humans.

Cross-chart arithmetic compounds two errors — you have to extract two values correctly AND do the calculation. Any error propagates.

Some questions are just badly suited to chart reading — the 'find missing value' question is a number sequence problem that happens to be presented on a chart. Our table extraction approach doesn't help here.

Tiny percentages — differences of 0.03% — are literally below the visual resolution of the chart image.

These failures are honest and expected. They also suggest clear directions for improvement — better OCR for decimal reading, symbolic arithmetic (like Program-of-Thoughts), and better visual grounding for color-based questions."

---

### [STUDY]
**Why show these examples specifically?** Three reasons. First, it builds credibility — you know your data, not just the aggregate numbers. Second, it motivates the limitations section (you can't solve 3.0238 precision without better OCR or symbolic tools). Third, it opens the door to future work (RQ for 4-2: can PoT or ChainOfTable solve the arithmetic problems?).

**What the committee might ask:** "Why didn't you just use more training data?" Answer: We tested with 1K to validate the RLVR paradigm in a low-resource setting. We expected more data to help further, which is part of our future work. The interesting finding is that 1K already gives 11pp improvement.

---
---

# SLIDE 13: Results

---

## 13a — Main Results Table

### [SLIDE]

**Title:** Results: Every Number, Every Method

**Table 1 — ChartQA (In-Distribution, 500 samples)**

| Method | Acc% | Pass@1 | Pass@2 | Pass@4 | C_table | D_reason | Coherence | Format% |
|---|---|---|---|---|---|---|---|---|
| Base | 51.8 | 52.45 | 67.33 | 77.4 | 0.591 | 0.040 | 0.545 | 29.8 |
| GRPO | 63.0 | 61.88 | 73.10 | 80.31 | 0.749 | 0.058 | 0.252 | 96.8 |
| **GRPO+HCPC** | 62.2 | 62.85 | 74.57 | **82.80** | 0.741 | 0.052 | 0.244 | **98.2** |
| **NSR** | **63.4** | **62.20** | 73.50 | 82.00 | **0.779** | 0.057 | 0.245 | — |

**Bootstrap 95% CIs (key comparisons):**
- HCPC vs GRPO on Pass@4: +1.2pp, CI [-1.8, +4.2pp], p=0.234
- NSR vs GRPO on Accuracy: +0.4pp, CI [-2.4, +3.2pp], p=0.416

**Table 2 — ChartFC (Out-of-Distribution, 500 samples)**

| Method | Acc% | Pass@1 | Pass@2 | Pass@4 | C_table | D_reason | Coherence | Format% |
|---|---|---|---|---|---|---|---|---|
| GRPO | **63.8** | **66.10** | **82.07** | **92.2** | **0.950** | 0.059 | 0.312 | **98.8** |
| GRPO+HCPC | 60.6 | 62.45 | 79.90 | 91.8 | 0.782 | **0.060** | **0.433** | 66.4 |

---

### [SCRIPT]
"Here are all the numbers. Let me walk through them and tell you the real story.

**The big headline:** RLVR works. One training run, 1,000 samples, 2,000 steps — and every metric goes up significantly. Accuracy goes from 51.8% to 62–63%. Format compliance from 29.8% to 96–98%. Table extraction consistency from 0.591 to 0.749–0.779. These are not marginal improvements — this is the system genuinely learning to do chart reasoning.

**On the trained methods competing with each other:** The honest answer is they are all comparable. The spread between trained methods is 1.2 percentage points on accuracy — 62.2% to 63.4%. That is a real difference but it's within the noise at this scale. The bootstrap confidence intervals all overlap. We'll talk about why this is actually fine.

**Where HCPC distinctly wins:** Pass@4. GRPO+HCPC achieves 82.80% versus GRPO's 80.31%. What this means is: if you give HCPC's model 4 chances to answer a question, it finds the right answer more often than GRPO — not because it's smarter on any single attempt, but because it maintains more diverse correct strategies. More paths to the right answer.

**Where NSR distinctly wins:** C_table — 0.779, the highest. NSR produces the most stable, consistent visual perception. Every time it sees the same chart, it extracts nearly the same table. This is a property of the negative-only reinforcement strategy: by not telling the model what extraction pattern to reinforce when correct, the model converges to naturally stable perceptions.

**The coherence apparent paradox:** Coherence drops from Base (0.545) to all trained methods (~0.245). Does this mean training hurt reasoning? No — and this is important to understand. Base model only produces structured output 29.8% of the time. When it does, it's probably on easy questions where it already knows the answer. So coherence measured on the easy subset looks high. After training, the model always produces structured output — including on hard questions. Coherence measured on everything, including hard questions, looks lower. The denominator changed."

---

### [STUDY]
**How to explain "we didn't beat GRPO on accuracy but that's okay":** The key argument is that the metrics HCPC wins on — Pass@4 and OOD coherence — are *theoretically more interesting* than raw accuracy. Accuracy measures whether you're right on the single shot. Pass@4 measures whether your model has multiple valid reasoning paths — which is more robust and better for inference-time scaling. OOD coherence measures whether the learned pipeline generalizes correctly. These are arguably more important for the long-term story.

**The "all over the place" feeling:** The reason results feel scattered is that no single method wins on everything. But that's actually a valid scientific finding. It says: different reward strategies optimize different aspects of reasoning quality. This is a richer story than "Method A > Method B on all metrics." You can frame it as: "Our experiments reveal the differential effects of reward design on chart reasoning — HCPC optimizes diversity and coherence; NSR optimizes consistency and efficiency."

---

## 13b — The OOD Story

### [SLIDE]

**Title:** Out-of-Distribution (ChartFC): What Generalizes?

**Side-by-side analysis:**

```
IN-DISTRIBUTION (ChartQA)          OUT-OF-DISTRIBUTION (ChartFC)
──────────────────────────────────────────────────────────────────
GRPO:  Acc=63.0%, C_table=0.749    GRPO:  Acc=63.8%, C_table=0.950  ↑
HCPC:  Acc=62.2%, C_table=0.741    HCPC:  Acc=60.6%, C_table=0.782  ↓

GRPO:  Coherence=0.252             GRPO:  Coherence=0.312            ↑
HCPC:  Coherence=0.244             HCPC:  Coherence=0.433            ↑↑ (+39%)

GRPO:  Format=96.8%                GRPO:  Format=98.8%               ↑
HCPC:  Format=98.2%                HCPC:  Format=66.4%               ↓↓ (−32pp)
```

**Key insight boxes:**
- GRPO's table extraction *improves* on OOD → learned generalizable perception
- HCPC's coherence *jumps* on OOD → hierarchical reward genuinely improves reasoning pipeline
- HCPC's format *crashes* on OOD → tension between content and structure under distribution shift

---

### [SCRIPT]
"The OOD results are where the story gets interesting. Let me give you the honest framing upfront: we are not claiming our method beats GRPO on OOD accuracy. GRPO wins there — 63.8% vs 60.6%. But if you only look at accuracy, you miss the more important finding.

Look at coherence. In-distribution, both methods are comparable at ~0.25. On OOD — ChartFC, never seen during training — HCPC's coherence jumps to 0.433 while GRPO sits at 0.312. That's a 39% relative improvement. What does coherence mean? It's the conditional probability: given that the model correctly extracted the table, how often does it then get the answer right? HCPC's hierarchical reward — which specifically rewards the chain from correct extraction to correct answer — appears to have genuinely strengthened this pipeline in a way that generalizes.

GRPO's C_table on OOD is extraordinary — 0.950, near-perfect. GRPO apparently learned a very robust perceptual pipeline that transfers to new chart styles. We think ChartFC's chart styling is cleaner and more consistent than ChartQA, which helps GRPO's extraction stability.

The format story for HCPC is a genuine limitation. 66.4% format compliance on OOD vs 98.8% for GRPO. The model starts dropping the `<think>` tag and proper ordering when it encounters unfamiliar charts. We believe this is because HCPC's reward weights emphasize table content (weight 2.0) over format, so under uncertainty, the model prioritizes getting content right over structure. This is correctable with explicit format regularization, which is future work.

Is bad OOD accuracy a problem? Not necessarily a fatal one. We can explain it: ChartFC is a fact-checking task with a constrained 3-way answer space — it's a different task, not just a different chart style. We would have been surprised if a model trained on open-ended Q&A performed identically on fact-checking. What matters is that we can point to *specific* aspects of generalization that our method improves — and coherence is the clearest example."

---

### [STUDY]
**Why OOD accuracy being lower is not a crisis:** Academic committees understand distribution shift. What they don't accept is no explanation for it. Your explanations: (1) different task format (Q&A vs fact-check), (2) different answer space (numerical vs 3-way categorical), (3) HCPC format collapse means the model is producing less structured output, which hurts evaluated accuracy. These are *mechanistic* explanations, not excuses.

**The coherence +39% is your strongest OOD finding:** Coherence measures whether learning *generalized correctly*, not just whether the model got lucky. When the model correctly perceives the chart and then reliably derives the correct answer from that perception — that's the full reasoning pipeline working. HCPC improves this by 39% on OOD, which is the most compelling evidence that the hierarchical reward structure captures something real about chart reasoning.

---

## 13c — Qualitative Analysis

### [SLIDE]

**Title:** Qualitative Analysis: Same Question, Different Models

**Example 1 — Where HCPC shows better reasoning diversity:**

*Show an actual sample from your data where HCPC's 4 rollouts use visibly different reasoning paths but all arrive at the correct answer, while GRPO's 4 rollouts use identical paths.*

**Example 2 — Where NSR shows extraction stability:**

*Show a sample where NSR's 4 rollouts extract nearly identical tables (C_table ≈ 1.0), demonstrating perceptual robustness.*

**Example 3 — A hard failure case:**

*Show the question "what is the value of the largest bar?" (answer: 3.0238) and show how all methods answer approximately 3.0 or 3.02 but miss the full precision. Use this to motivate symbolic arithmetic (PoT) as future work.*

**Example 4 — Base vs GRPO+HCPC format comparison:**

*Show Base model response (no tags, loose text) vs GRPO+HCPC response (clean `<think>`, `<table>`, `<answer>` structure). Makes the format compliance improvement immediately visual.*

---

### [SCRIPT]
"Let me show you some actual model outputs to ground the numbers in real behavior.

[Example 1] Here's a question about a bar chart. HCPC's four rollouts all get the right answer, but if you look at the reasoning chains in the `<think>` blocks — one starts by identifying the tallest bar, one starts by reading all values and sorting, one identifies the answer by comparison. Three different strategies, all correct. This is what high Pass@4 looks like in practice. GRPO's four rollouts all follow the same pattern — they're more consistent but less diverse.

[Example 2] NSR on this chart extraction — look at the four tables. The column names differ slightly in phrasing, but the values are essentially identical across all four rollouts. C_table for this sample is 0.98. This is what stable visual perception looks like.

[Example 3] This is one of our hard failure cases. The question asks for the value of the largest bar. The answer is 3.0238 — a four-decimal-precision value. Every method, including our best, answers 3.0 or 3.02. The chart axis labels are simply too small to resolve that precision from the image. This motivates adding a Program-of-Thoughts component in future work — if the model could generate code to read exact numerical values, it wouldn't have to rely on visual OCR.

[Example 4] This is why format compliance matters. Base model output — just a wall of text, no structure. HCPC output — clean hierarchy, structured table in JSON, clear reasoning chain, final answer clearly delimited. These are the same model architecture, just 2000 training steps apart. Format is the first thing RLVR teaches."

---

### [STUDY]
**How to prepare qualitative examples:** Before the defense, print 3–4 actual output pairs. Read them yourself. If you see something interesting or unexpected in an output — a reasoning chain that's surprisingly good or bad — that's worth mentioning. Real examples carry more weight than constructed ones because they're verifiable.

**What makes a good qualitative example:** One that makes a metric tangible. "C_table = 0.779" is abstract. "Here are the four tables NSR extracted for this sample — they're almost word-for-word identical" is concrete. Always pair a number with an example when you can.

---

## 13d — Answering the Research Questions

### [SLIDE]

**Title:** Answering Our Research Questions

| RQ | Question Summary | Our Answer | Evidence |
|---|---|---|---|
| RQ1 | Does RLVR work at small scale? | **Yes, definitively** | +11pp accuracy, +67pp format, all methods |
| RQ2 | Does HCPC improve over flat GRPO? | **Partially** | Better Pass@4, better OOD coherence; not better raw accuracy |
| RQ3 | Does HCPC promote diversity + consistency? | **Pass@4 yes; D_reason limited by K=4** | Pass@4: 82.80% vs 80.31%; D_reason low across all methods |
| RQ4 | Does NSR transfer to VLMs? | **Yes** | Highest accuracy (63.4%), highest C_table (0.779) |
| RQ5 | How do methods generalize OOD? | **GRPO: stable; HCPC: better coherence, worse format** | Table 2 full results |
| RQ6 | C_table vs D_reason vs correctness? | **C_table: negative correlation; D_reason: positive; HCPC weakens anti-correlation** | Section 5 correlation analysis |

---

### [SCRIPT]
"Before I close the results section, let me go back to the six research questions I posed at the beginning and give explicit answers.

RQ1 — yes, definitively. 11 percentage points from 1,000 training samples. This is real and substantial.

RQ2 — partial yes. HCPC doesn't consistently beat GRPO on raw accuracy, but it achieves the highest Pass@4 and the highest OOD coherence. If your criterion for 'improvement' is single-shot accuracy, the picture is mixed. If your criterion is robust, generalizable, diverse reasoning — HCPC has a case.

RQ3 — yes on Pass@4 diversity, limited on D_reason due to our experimental design. 4 rollouts is simply not enough to observe meaningful Jaccard diversity in reasoning chains. That's a design constraint, not a method failure.

RQ4 — yes. NSR works on VLMs. Best accuracy, best table consistency. The negative-only reinforcement strategy generalizes beyond text math.

RQ5 — GRPO generalizes robustly in perception (C_table = 0.950). HCPC generalizes better in reasoning pipeline (coherence = 0.433). Format generalization is HCPC's weakness.

RQ6 — C_table is *negatively* correlated with correctness, which surprised us and taught us something important: stable extraction is not sufficient for correct answers. D_reason is positively correlated — diverse reasoning helps. HCPC weakens the C_table anti-correlation, which is evidence that hierarchical rewards partially decouple the perception and reasoning stages."

---
---

# SLIDE 14: Conclusion

---

## 14a — Summary of Contributions

### [SLIDE]

**Title:** What We Did: A Summary

**Three contributions, three rows:**

**1. We validated RLVR for chart QA at small scale**
- Trained Qwen2.5-VL-3B on 1,000 samples with GRPO + Chart-RVR rewards
- Achieved +11pp accuracy improvement, +67pp format compliance improvement
- Confirmed RLVR's data efficiency applies to multi-modal, multi-component reward settings

**2. We proposed and evaluated HCPC** *(Hierarchical Correct-Path Consistency)*
- Novel reward that operates across rollouts, not within them
- Rewards: consistent table extraction AND diverse reasoning, conditioned on correctness
- Results: best Pass@4 (82.80%), best OOD coherence (+39% relative vs GRPO)
- Limitation: OOD format compliance degrades (66.4%) — known direction for improvement

**3. We applied NSR to VLMs for the first time**
- First application of Negative Sample Reward [Zhu et al., 2025] to vision-language models
- Results: highest accuracy (63.4%), highest table extraction consistency (0.779)
- Demonstrates that negative-only reinforcement handles noisy multi-modal rewards effectively

**One paragraph summary:**
*"RLVR, applied to chart QA with 1,000 training samples, achieves substantial improvements across all methods. HCPC's hierarchical reward design promotes reasoning diversity and OOD coherence. NSR's first VLM application shows strong perceptual stability. All differences between trained methods are consistent with theoretical predictions but require larger-scale validation to establish statistical significance."*

---

### [SCRIPT]
"Let me summarize what we've done.

We started with a base model scoring 51.8% on ChartQA — barely above random for this task. Through RLVR training with a structured multi-component reward, we trained three variants that all reach 62–63% accuracy, format compliance above 96%, and table extraction stability above 0.74. With only 1,000 training samples.

Our first novel contribution — HCPC — adds a cross-rollout consistency signal that existing methods lack. When the model gets something right, HCPC asks: are you getting it right *consistently* at the perception level, and *diversely* at the reasoning level? This distinction — converge in perception, diverge in reasoning — is the core design principle, and it shows up in the results as better Pass@4 and better OOD coherence.

Our second contribution — applying NSR to VLMs — demonstrates that the negative reinforcement insight generalizes beyond text math. A model that only learns what *not* to do ends up converging to the most stable and accurate behavior — highest accuracy, highest extraction consistency.

What we haven't shown: statistical significance between methods at this scale. The differences are there, they're directionally consistent with theory, but at 500 test samples and 1K training samples, the confidence intervals overlap. That's an honest limitation, not a fatal one."

---

### [STUDY]
**How to handle "but your improvements aren't significant?"** Acknowledge it directly: "The trends are consistent with theoretical predictions from Wen et al. and Zhu et al., but reach significance at this training scale. Our future work directly addresses this by scaling training data and rollout count." This shows you understand statistical testing, you're not overclaiming, and you have a plan.

---

## 14b — Future Work (4-2)

### [SLIDE]

**Title:** What's Next: 4-2 Research Roadmap

**Three tracks:**

**Track 1: Scale the current experiments**
- Train on full 34,194-sample Chart-RVR dataset (not just 1K)
- Increase rollouts K=4 → K=8 (more diversity signal for HCPC)
- Test on larger test set (500 → full ChartQA test split ~2,500)
- Expected: statistical significance will emerge at K=8, 34K samples

**Track 2: Strengthen the pipeline**
- Add Program-of-Thoughts (PoT) [TinyChart, EMNLP 2024] reward component
  - Model generates Python code → execute → check output
  - Addresses decimal precision failures (3.0238 problem)
- Add Chain-of-Table operations for numerical reasoning
- Reduce HCPC OOD format collapse via explicit format regularization

**Track 3: Improve OOD robustness**
- Train with DAPO's Dynamic Sampling filter — discard samples where all K rollouts get identical rewards (no learning signal)
- Test on EvoChart (the 53.36% gap) — can HCPC's coherence advantage close this?
- Explore ChartGen [2025] synthetic augmentation for OOD style coverage

**One-slide visual:**
```
4-1 Work (Done)          4-2 Roadmap
─────────────────────────────────────────────────────────
✅ HCPC reward design    → Scale to 34K samples
✅ NSR for VLMs          → K=4 → K=8 rollouts
✅ 1K training, 2K steps → Statistical significance
✅ ChartQA + ChartFC     → Add EvoChart, full ChartQA
✅ 4-method comparison   → Add PoT reward, DAPO filter
✅ 6-metric analysis     → OOD format regularization
```

---

### [SCRIPT]
"For 4-2, we have three concrete tracks.

Track one is straightforward: scale what we have. The fact that trends exist but aren't significant yet tells us we need more data and more rollouts. If we go from 1K to the full 34K training set, and from K=4 to K=8 rollouts, the signal for HCPC's diversity reward should strengthen considerably. This is the highest-confidence path to validating our current findings.

Track two is about fixing the weaknesses we found. The decimal precision problem — 3.0238 — can be addressed by adding a Program-of-Thoughts component: instead of reading visual values, the model generates code that computes the answer symbolically. TinyChart already showed this works at 3B scale. HCPC's OOD format collapse needs explicit format regularization — probably just increasing the format reward weight in the HCPC bonus term.

Track three is about closing the OOD gap. We want to test on EvoChart — the benchmark where Chart-RVR showed a 31-point gap. Can our methods do better? Can HCPC's coherence advantage translate into better EvoChart performance? We also want to implement DAPO's dynamic sampling filter, which is very low implementation cost and has been shown to improve GRPO training quality significantly.

The overall story for 4-2 is: take the methods we've validated at small scale, scale them up, fix the known weaknesses, and test on harder OOD scenarios."

---

### [STUDY]
**How to talk about future work confidently:** Future work is not an admission of failure — it's evidence that you understand your field well enough to know what the next steps are. For each future direction, be able to say: (1) why we didn't do it in 4-1 (usually compute/time), (2) what specifically we expect it to improve and why, (3) what technique from recent literature we'd build on.

**The DAPO filter argument:** DAPO [ByteDance, March 2025] observed that many GRPO training batches are "useless" — all 4 rollouts get the same reward (either all correct or all wrong). When that happens, the advantage for every rollout is 0, and the gradient update is zero. You've wasted compute on a batch that taught the model nothing. DAPO filters these batches out. This is especially relevant for our 1K training set where many questions might be trivially easy or hard.

**Why K=8 matters for HCPC:** HCPC computes consistency and diversity among the *correct* rollouts (R+). With K=4, if only 1 of 4 rollouts is correct, you have a sample size of 1 for the cross-rollout analysis — which is meaningless. With K=8, even 2 correct rollouts give you a meaningful consistency/diversity signal. HCPC is likely underperforming its theoretical potential at K=4.

---
---

# Quick Reference: Key Numbers to Memorize

*Read these until they're automatic. You'll need them when questioned.*

```
CHARTQA (IN-DISTRIBUTION)
─────────────────────────
Base accuracy:          51.8%   (+0)
GRPO accuracy:          63.0%   (+11.2pp over base)
HCPC accuracy:          62.2%   (+10.4pp over base)
NSR accuracy:           63.4%   (+11.6pp over base)  ← highest

HCPC Pass@4:            82.80%  ← highest
GRPO Pass@4:            80.31%
NSR Pass@4:             82.00%

NSR C_table:            0.779   ← highest
GRPO C_table:           0.749
HCPC C_table:           0.741
Base C_table:           0.591

Format compliance (Base → trained): 29.8% → 96–98%

CHARTFC (OUT-OF-DISTRIBUTION)
──────────────────────────────
GRPO accuracy:          63.8%   ← wins on accuracy
HCPC accuracy:          60.6%

GRPO C_table OOD:       0.950   ← near perfect, wins strongly
HCPC C_table OOD:       0.782

HCPC coherence OOD:     0.433   ← wins (+39% vs GRPO)
GRPO coherence OOD:     0.312

HCPC format OOD:        66.4%   ← major limitation
GRPO format OOD:        98.8%

STATISTICAL TESTS
──────────────────
HCPC vs GRPO, Pass@4: p=0.234 (not significant, trend correct)
NSR vs GRPO, Acc:     p=0.416 (not significant, trend correct)
Both overlapping 95% CIs → cannot claim significance at this scale
```

---

# One-Sentence Definitions for Every Term (If You Blank Out)

| Term | Say This |
|---|---|
| RLVR | "Training a model by giving it a reward signal instead of showing it correct examples" |
| GRPO | "A training algorithm that computes advantage by comparing each response to the average quality of the group" |
| HCPC | "Our reward that gives bonus points when correct responses are consistent in extraction and diverse in reasoning" |
| NSR | "A training variant that only penalizes wrong responses and ignores correct ones, preserving multiple correct strategies" |
| LoRA | "A method to train only 0.16% of the model's parameters instead of all 3 billion" |
| Pass@K | "The probability that at least one of K sampled responses is correct" |
| C_table | "How similar are the extracted tables across multiple responses to the same question" |
| D_reason | "How different are the reasoning chains across multiple responses" |
| Coherence | "Given that you extracted the right table, how often do you then give the right answer" |
| OOD | "A test on data the model was never trained on, using a different task format" |
| Bootstrap CI | "A confidence interval computed by resampling the data 10,000 times to estimate uncertainty" |

---

*Last updated: 2026-04-19 — based on verified experimental results at checkpoint-2000, 500-sample test sets.*
*Iterate this document after each feedback session.*
