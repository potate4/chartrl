# Presentation Speech — HCPC-RLVR

---

Good morning everyone.

Today I want to talk about a problem that sits at the heart of visual chart understanding — and a training approach I've designed to address it.

---

When we ask a language model to reason about a chart, we typically evaluate it on whether it gives the right answer. But there's a deeper question worth asking: **how** did it get there?

Standard reinforcement learning methods like GRPO are very good at pushing a model toward correct answers. The problem is, they tend to do this by collapsing the model's reasoning into a single, repetitive strategy. Ask the same question four times with slight randomness, and you get four answers that look almost identical — same steps, same phrasing, same path. That's what I call **reasoning collapse**.

And it matters, because a model that has memorised one reasoning template isn't actually understanding the chart. It's pattern-matching. The moment you show it a chart style it hasn't seen before — a new layout, a different colour scheme, an unfamiliar axis format — that single template breaks down.

---

My thesis proposes a different training objective, called **HCPC — Hierarchical Correct-Path Consistency**.

The core idea is this. Chart reasoning has a natural hierarchy. At the top, you identify the chart type — is this a bar chart, a line chart, a pie chart? In the middle, you extract the data table — what are the actual values? At the bottom, you reason over those values to reach an answer.

My claim is that the top two levels should be **consistent** across rollouts. If you sample four responses to the same question, they should all agree on the chart type and produce similar data tables. But the reasoning in between — the step-by-step derivation — should be **diverse**. Multiple valid reasoning paths should be discovered and rewarded, not collapsed into one.

HCPC operationalises this as a group-level reward. Instead of rewarding each rollout in isolation, we look across all K rollouts together, filter down to the ones that got everything right, and ask: are the tables consistent? Are the reasoning traces diverse? The answers to those questions become a reward signal that shapes the entire group.

---

Now let me show you what the experiments found.

I trained two models on top of Qwen 2.5 VL 3 Billion using LoRA fine-tuning. The first is a GRPO baseline — it uses the standard reward stack from Chart-RVR, including a process reward that scores reasoning similarity to a reference trace. The second is GRPO-HCPC — identical setup, but the process reward is replaced by the group-level HCPC signal.

Both were evaluated on five hundred samples from the ChartQA test set, with four rollouts per question.

The first thing that jumps out is **accuracy**. The base model, with no training at all, gets 51.8%. GRPO pushes that to 63%. GRPO-HCPC lands at 62.2% — nearly identical to GRPO. Replacing the process reward with a fundamentally different group-level objective costs us less than one percentage point of accuracy.

But the more interesting story is in **pass at k**. Pass at k asks: if you sample k responses, what fraction of questions does the model get right at least once? At k equals four, GRPO scores 80.3%. GRPO-HCPC scores 82.8%. That gap is meaningful — it says the model's candidate set is more reliably correct.

And then there's a metric I computed directly from the per-sample files that doesn't appear in the standard summaries. I call it the **all-four-correct rate** — the fraction of questions where every single one of the four rollouts was correct. For the base model, that's 23.4%. For GRPO, 37.2%. For GRPO-HCPC — **44.4%**. Nearly half of all test questions are solved correctly on every rollout. That is, I think, the clearest number in this entire report. It says that HCPC doesn't just improve the best-case outcome — it improves the floor.

---

What about diversity? The D_reason metric measures pairwise dissimilarity across reasoning traces. Higher means more diverse. The base model scores 0.040. GRPO scores 0.052. GRPO-HCPC scores 0.058 — the highest of the three. The process reward in GRPO pulls all rollouts toward a single reference path. HCPC's diversity signal breaks that pull.

The absolute values are still modest — the maximum is around 0.07 when we look only at the correctly solved questions. There are two straightforward explanations for this. First, four rollouts is a small sample for measuring diversity — eight would give a more reliable estimate. Second, the diversity weight in the HCPC reward function is likely undertuned relative to the consistency weights. That's a direct knob I can turn in the next training run.

---

There are two things I want to be honest about.

First, GRPO still edges out GRPO-HCPC on first-rollout accuracy — 63% versus 62.2%. That gap is small, but it's real. If you only care about picking one answer and being right, the baseline is marginally better. HCPC's advantage is in reliability across multiple samples, not in single-shot precision.

Second, and more importantly — I don't yet have out-of-distribution results. The entire theoretical motivation for HCPC is that diversity prevents overfitting to a single reasoning template, which should improve generalisation to new chart styles. That argument needs EvoChart numbers to stand. Those experiments are the immediate next step, and they require no retraining — just running the existing checkpoints through the evaluation pipeline.

---

So where does this leave us?

The first tier of the thesis claim is supported. GRPO-HCPC produces rollout sets that are more reliable and more diverse than standard GRPO, with negligible accuracy cost and at base-model inference speed. The group-level HCPC objective is doing what it was designed to do.

The second tier — that this translates to better out-of-distribution generalisation — remains a hypothesis. One I expect the EvoChart results to confirm, but a hypothesis nonetheless.

Four of the six planned experiments are still to be run. NSR and W-REINFORCE, both with and without HCPC, will complete the ablation. NSR in particular is worth watching — by skipping gradient updates on correct samples entirely, it should push reasoning diversity even further than HCPC alone.

---

Thank you. I'm happy to take questions.

---

*Speech prepared: 2026-03-03*
*Based on: app/docs/cleaned.md*
