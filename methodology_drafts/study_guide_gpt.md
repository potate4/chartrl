# Study Guide: Understanding and Framing the Experimental Results

This document is a comprehensive study guide for interpreting the results in:

- `methodology_drafts/detailed_analysis.md`
- `methodology_drafts/experimental_results_draft.md`
- `app/scripts/analyze_results.py`
- `app/docs/CHARTQA_QWEN_COMPARISON_20260302.md`

Its purpose is not just to repeat the numbers, but to help you understand:

1. what each metric is actually measuring,
2. what each method appears to be doing behaviorally,
3. which conclusions are directly supported by the data,
4. which conclusions are plausible but should be framed cautiously,
5. how to write about the results in a thesis or paper without overclaiming.

This guide assumes the experiments are exactly the ones summarized in the analysis files:

- Base Qwen2.5-VL-3B-Instruct
- GRPO
- GRPO+HCPC
- NSR
- In-distribution evaluation on ChartQA
- Out-of-distribution evaluation on ChartFC for GRPO and HCPC
- 4 rollouts per sample
- 500 evaluation samples per benchmark

---

## 1. The Big Picture

At the highest level, your experiments support five main messages.

### 1.1 RLVR clearly helps this model family

All three trained methods beat the base model on the main in-distribution metrics. The jump from the base model to any RL-trained variant is much larger than the differences among the trained variants themselves. This means the strongest experimental conclusion is:

**RL with verifiable rewards is effective for structured chart reasoning, even with a small 3B VLM and limited training data.**

That is a strong, defensible claim because:

- Base Acc% is `51.8`, trained methods are `62.2-63.4`
- Base Pass@4 is `77.4`, trained methods are `80.3-82.8`
- Base format compliance is `29.8%`, trained methods are `96.8-98.2%`
- Base table consistency is `0.591`, trained methods are `0.741-0.779`

### 1.2 HCPC and NSR do not dominate in exactly the same way

The methods separate by *profile*, not by a single winner-takes-all result:

- **HCPC** looks best when you care about multi-rollout coverage and structured candidate-set quality.
- **NSR** looks best when you care about single-shot accuracy and stable table extraction.
- **GRPO** is the strongest simple baseline and remains competitive, especially on OOD ChartFC.

So the right framing is not "HCPC wins everything" or "NSR wins everything." The right framing is:

**Each method shifts the behavior of the rollout distribution differently.**

### 1.3 Pass@K matters a lot in this project

Your setup is not just a single-output classifier. It explicitly samples multiple rollouts. That means Pass@K is not a side metric. It is central.

This is especially important because HCPC is designed to shape behavior *across rollouts*, not just within a single response. So if HCPC helps Pass@4 more than raw Acc%, that is not a weakness of the method. It may be the most natural place where the method should show benefits.

### 1.4 Format learning is easy in-distribution, but not automatically robust OOD

One of the clearest results is that all trained methods almost solve the formatting problem on ChartQA. But HCPC drops sharply on ChartFC format compliance. This means:

- format compliance is easy to learn on the training distribution,
- format robustness across distribution shift is not guaranteed,
- HCPC may be trading off some structural rigidity for other aspects of behavior under shift.

That is one of the most important limitations in the whole study and should be discussed directly.

### 1.5 Most differences among trained methods are real trends, but not statistically decisive here

The paired bootstrap results do not support strong claims that HCPC or NSR is definitively better than GRPO at this experimental scale. The safe interpretation is:

- there are clear behavioral trends,
- the directions are meaningful,
- the magnitudes are still modest,
- the experiment is not large enough to establish definitive superiority between trained methods.

This is actually a good thesis result. It shows honesty and maturity in interpretation.

---

## 2. What Each Method Is Trying to Do

Before interpreting the numbers, you need a clean mental model of each method.

## 2.1 Base model

The base model is `Qwen2.5-VL-3B-Instruct` without RL training for this task.

What that means behaviorally:

- It already has some general chart understanding.
- It was not optimized for your required XML-like output structure.
- It was not trained to produce stable chart type, table extraction, and reasoning across repeated rollouts.
- It can still answer many questions correctly, but its outputs are noisy and inconsistent.

This is why the base model has:

- decent accuracy,
- poor format compliance,
- lower table consistency,
- a wider gap between Pass@1 and Pass@4.

## 2.2 GRPO

GRPO is the standard RLVR baseline in the Chart-RVR style setup.

Behaviorally, GRPO:

- rewards relatively better rollouts inside a group,
- penalizes relatively worse rollouts,
- pushes probability mass toward higher-reward trajectories,
- tends to improve correctness and formatting efficiently,
- may reduce diversity if the optimization concentrates too much on a narrow set of successful patterns.

What you should remember:

**GRPO is the baseline that best represents "standard RLVR for this task."**

If HCPC or NSR improve over GRPO, that means they improve over the task-appropriate baseline, not just over the raw base model.

## 2.3 HCPC

HCPC stands for Hierarchical Correct-Path Consistency.

Its logic is not just "be correct." It says:

- among correct rollouts,
- chart type should be consistent,
- table extraction should be consistent,
- reasoning paths should still preserve some diversity.

So HCPC is trying to encourage a very specific kind of rollout set:

- not random correctness,
- not one single collapsed template,
- but a structured family of correct responses that agree on perceptual/intermediate content while allowing some variation in reasoning.

This is why HCPC should be expected to show its best effects on:

- Pass@K,
- correct-rollout rate,
- robustness of the rollout set,
- coupling between extraction and reasoning.

It does **not** necessarily have to maximize a standalone diversity metric in isolation.

## 2.4 NSR

NSR stands for Negative Sample Reinforcement.

Its central idea is:

- do not strongly reinforce correct trajectories,
- mainly push down incorrect trajectories,
- let the model redistribute probability mass toward plausible alternatives already supported by its prior.

Behaviorally, that means NSR often:

- preserves more optionality than aggressive positive reinforcement,
- avoids collapsing too hard onto one favored path,
- may stabilize behavior when the base model already has useful knowledge,
- can help maintain broad solution coverage while still improving accuracy.

In your experiments, NSR also seems especially strong on table consistency. That suggests it may be allowing the model's perceptual extraction behavior to settle into a stable regime without overcommitting to any single exact successful path.

---

## 3. Metrics: What They Really Mean

This section is critical. A lot of result interpretation fails because metrics get described loosely instead of operationally.

## 3.1 Accuracy (Acc%)

From `analyze_results.py`, your Acc% is based on `relaxed_accuracy`, not `any(correct)`.

That distinction matters.

Acc% here means:

- a sample-level evaluation,
- using the evaluator's relaxed answer matching,
- not simply whether any rollout was marked correct by the stricter per-rollout signal.

So when you discuss Acc%, do **not** casually equate it with Pass@4.

Best way to frame it:

**Acc% is the benchmark-facing task accuracy under the evaluator's relaxed matching criterion.**

## 3.2 Pass@1

In your analysis, Pass@1 corresponds to the mean per-sample `correct_rate`, which under four rollouts is the average fraction of rollouts that are correct.

Since the unbiased Pass@K estimator is used, Pass@1 becomes equivalent to the average correctness probability of one sample drawn from the rollout distribution.

Interpretation:

- If Pass@1 is high, a random single rollout is often correct.
- This is the best proxy for single-shot usability.

## 3.3 Pass@4

Pass@4 measures the probability that at least one of four rollouts is correct.

Interpretation:

- If Pass@4 is high, the rollout set has good coverage.
- This metric benefits from either:
  - higher per-rollout correctness,
  - or multiple complementary correct paths,
  - or both.

Best framing:

**Pass@4 measures the practical value of sampling four candidates.**

That is why HCPC doing best on Pass@4 is meaningful.

## 3.4 The Pass@K estimator

Your script uses the unbiased estimator:

`Pass@K = 1 - C(n-c, k) / C(n, k)`

where:

- `n` is the number of rollouts,
- `c` is the number of correct rollouts.

The important bug fix in your analysis was:

- guaranteed success happens when `n - c < k`,
- not when `c >= k`.

This matters because the old logic would mis-handle edge cases and bias results.

How to mention this in writing:

**Pass@K values in this project were recomputed using the standard unbiased estimator introduced for code generation evaluation, with a corrected edge-case implementation in the analysis script.**

## 3.5 C_table

This is table extraction consistency across rollouts.

Interpretation:

- high `C_table` means the model is extracting a similar table repeatedly,
- low `C_table` means the model's perceptual/structured representation of the chart varies across rollouts.

What it does **not** mean automatically:

- it does not guarantee the table is correct,
- it does not guarantee the answer is correct,
- it does not guarantee good reasoning.

This is one of the most important conceptual cautions in your whole study.

## 3.6 D_reason

This is a reasoning diversity metric, computed from reasoning-step variation across rollouts.

Interpretation:

- higher `D_reason` means the model explores more varied reasoning traces,
- lower `D_reason` means rollouts use more similar reasoning patterns.

Important caution:

- it is a *proxy* for diversity,
- it is not the same thing as "better reasoning",
- low values do not necessarily mean the method failed,
- high values do not necessarily mean the method is more faithful or more correct.

That caution is especially important for HCPC, because HCPC is optimizing a joint objective, not maximizing diversity in isolation.

## 3.7 Coherence

Coherence is one of the trickiest metrics conceptually.

You define it as the fraction of rollouts where correct table extraction leads to a correct final answer.

Interpretation:

- if coherence is high, the model is using the extracted table effectively,
- if coherence is low, correct extraction often fails to translate into a correct final answer.

This is a reasoning-integrity metric, not a raw performance metric.

It is asking:

**Once the model has the right intermediate representation, does it reason correctly from it?**

## 3.8 Format compliance

This measures whether the model follows the required output schema.

Interpretation:

- high format compliance means the model reliably obeys the expected output structure,
- low format compliance means the pipeline is brittle, even if some answers are still correct.

This metric matters because:

- your evaluator depends on structured tags,
- downstream parsing depends on structured tags,
- interpretability claims depend on having extractable intermediate outputs.

## 3.9 Reasoning length

This is the average number of words in the `<think>` block.

Interpretation:

- longer is not automatically better,
- shorter is not automatically better,
- differences are useful only when linked to correctness, latency, or robustness.

Your results suggest incorrect answers are often slightly longer. That usually indicates:

- rambling,
- uncertainty,
- compensatory over-generation,
- or failure to terminate reasoning cleanly.

## 3.10 Overlap analysis

Overlap and Jaccard similarity tell you whether methods solve mostly the same samples.

Interpretation:

- high overlap means methods are largely working on the same solvable region,
- unique solves reveal complementary strengths,
- universally failed cases show the true frontier of difficulty.

This is important because it tells you whether a new method discovers a *different capability regime* or just slightly reweights the same one.

---

## 4. The Core In-Distribution Story on ChartQA

Here are the main ChartQA results again in compact form:

| Method | Acc% | Pass@1 | Pass@4 | C_table | D_reason | Coherence | Format% |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | 51.8 | 52.45 | 77.4 | 0.591 | 0.040 | 0.545 | 29.8 |
| GRPO | 63.0 | 61.88 | 80.31 | 0.749 | 0.058 | 0.252 | 96.8 |
| GRPO+HCPC | 62.2 | 62.85 | 82.80 | 0.741 | 0.052 | 0.244 | 98.2 |
| NSR | 63.4 | 62.20 | 82.00 | 0.779 | 0.056 | 0.245 | 97.0 |

## 4.1 What changed from Base to trained models

The jump from Base to any trained method is the cleanest result in the study.

The trained methods show:

- about `+10 to +12` points in Acc%
- about `+9 to +10` points in Pass@1
- about `+3 to +5` points in Pass@4
- massive format compliance improvement
- major table consistency improvement

What that means behaviorally:

- RL training substantially improved output regularity,
- the model became more reliable per rollout,
- the rollout set became more structured and machine-parseable,
- the model learned a more stable chart-reading pipeline.

This is the strongest section of the thesis evidence.

## 4.2 Why Pass@4 improves less than format compliance

You might wonder why format jumps by nearly `+67` points while Pass@4 rises by only `+3 to +5` points.

That is actually normal.

Reason:

- format compliance is relatively easy to reward and learn,
- answer correctness is harder because it requires perception, extraction, and reasoning,
- the base model already had non-trivial task ability,
- so format has much more headroom than task performance.

This supports a useful interpretation:

**Format rewards are easy gains; reasoning gains are real but harder.**

## 4.3 Why the base model has a huge Pass@1 to Pass@4 gap

Base:

- Pass@1 = `52.45`
- Pass@4 = `77.40`
- gap = `24.95` points

This means the base model often "knows" or can reach the correct answer somewhere in its rollout distribution, but does so inconsistently.

Behaviorally, that suggests:

- noisy latent competence,
- weak structural control,
- unstable reasoning paths,
- occasional correct answers emerging from stochastic sampling.

In plain terms:

**The base model is capable, but unreliable.**

## 4.4 Why the trained models narrow that gap

GRPO:

- gap = about `18.43`

HCPC:

- gap = about `19.95`

NSR:

- gap = about `19.80`

Compared to the base model, trained methods have:

- higher Pass@1,
- somewhat higher Pass@4,
- smaller Pass@1 to Pass@4 gap.

This usually means training is concentrating probability mass onto more reliable trajectories.

That is a standard and defensible interpretation:

**training improves the quality of the typical rollout, not just the best-of-four outcome.**

## 4.5 The subtle difference between HCPC and NSR

Among trained methods:

- HCPC has the best Pass@4 (`82.8`)
- NSR has the best Acc% (`63.4`)
- HCPC has the best Pass@1 (`62.85`) in the table from the analysis script
- NSR is essentially tied on single-shot behavior and stronger on `C_table`

This means the methods are separating by mechanism:

- HCPC appears to improve the *candidate set*,
- NSR appears to improve *stability of the internal representation* and competitive single-shot performance.

That is a better framing than trying to pick one scalar winner.

---

## 5. How to Interpret Each Method Specifically

## 5.1 Base: "latent capability without control"

The base model is not weak in an absolute sense. It is surprisingly capable.

Evidence:

- `51.8` Acc%
- `77.4` Pass@4

But it is uncontrolled:

- only `29.8%` fully compliant formatting,
- lower table consistency,
- broader rollout instability.

Best thesis framing:

**The base VLM already contains meaningful chart-reasoning ability, but lacks task-specific structural discipline and rollout stability.**

## 5.2 GRPO: "strong generic RLVR baseline"

GRPO improves nearly everything that matters over the base model:

- higher accuracy,
- much better format compliance,
- much better table consistency,
- good Pass@K gains.

Best framing:

**GRPO is a strong baseline because it already converts the base model's noisy competence into more reliable structured behavior.**

Do not undersell GRPO. Your novel method only looks credible if the baseline is strong.

## 5.3 HCPC: "best candidate-set quality and rollout robustness"

HCPC's strongest empirical case is:

- highest Pass@4,
- highest format compliance in-distribution,
- strong correct-rollout behavior,
- weaker negative correlation between `C_table` and correctness than other methods,
- much better OOD coherence than GRPO.

These together support the claim that HCPC is improving *how sets of rollouts behave*, not merely one sampled answer.

Best framing:

**HCPC improves structured multi-rollout robustness by encouraging correct rollout families that agree on intermediate chart representations while preserving some room for reasoning variation.**

This is much better than saying "HCPC improves diversity," which the numbers do not strongly prove on their own.

## 5.4 NSR: "best extraction stability and highly competitive accuracy"

NSR's strongest empirical case is:

- highest Acc%
- highest `C_table`
- high Pass@4
- strong positive correlation between `D_reason` and correctness
- competitive formatting and parsing

Best framing:

**NSR transfers well to chart reasoning and appears especially effective at stabilizing table extraction while remaining competitive on both single-shot and multi-rollout correctness.**

The "first application of NSR to VLMs" angle is strong if that is factually correct for your literature positioning.

---

## 6. The Most Important Non-Obvious Patterns

Some of the strongest insights in your results are not the obvious top-line numbers.

## 6.1 The negative correlation between C_table and correctness

ChartQA correlations:

- Base: `-0.658`
- GRPO: `-0.528`
- HCPC: `-0.353`
- NSR: `-0.423`

At first glance, this seems wrong. Why would more consistent table extraction correlate with lower correctness?

Possible interpretation:

- some samples are consistently extracted but still reasoned over incorrectly,
- some hard samples produce a mix of good and bad extractions, and occasional correct extractions can yield correct answers,
- consistency alone is not the same as correctness,
- extraction may be stable on the wrong representation.

The safe way to frame this is:

**Table consistency is a stability measure, not a correctness guarantee. A model can be consistently wrong, and some samples only become solvable because stochasticity occasionally produces a better extraction.**

This is a very valuable thesis insight.

## 6.2 HCPC weakens that negative correlation

HCPC's `r = -0.353` is the least negative among the trained methods.

That matters because it suggests HCPC may be partially improving the alignment between:

- stable extraction,
- and eventual answer correctness.

This is not proof, but it is a reasonable interpretation:

**HCPC seems to reduce the disconnect between stable intermediate structure and final answer success.**

That is a very good place to discuss the hierarchical nature of the reward.

## 6.3 D_reason correlates positively with correctness for all methods

ChartQA:

- Base: `0.546`
- GRPO: `0.695`
- HCPC: `0.661`
- NSR: `0.706`

This does **not** prove that diversity causes correctness.

But it does show:

- samples with more reasoning variety tend to be the ones with higher correctness rates,
- diversity is at least associated with better rollout coverage,
- this supports the idea that maintaining multiple reasoning paths can be useful.

Best framing:

**Reasoning diversity is positively associated with correct rollout coverage, although the causal direction cannot be established from these correlations alone.**

## 6.4 Coherence drops sharply after training

This is one of the most important things to explain carefully, because otherwise it looks like training made reasoning worse.

ChartQA coherence:

- Base: `0.545`
- GRPO: `0.252`
- HCPC: `0.244`
- NSR: `0.245`

At face value, that looks bad.

But the correct interpretation is more careful:

- the base model often fails formatting and structured extraction,
- trained models attempt structured extraction far more often,
- this expands the set of cases in which coherence is measured,
- the denominator becomes larger and includes harder examples,
- the metric becomes stricter and more meaningful after training.

So the right claim is not:

"Training reduced reasoning quality."

The right claim is:

**After RL training, coherence is measured over a much larger and more structurally explicit set of extraction attempts, making direct comparison with the base model misleading.**

This should be stated explicitly in the thesis.

---

## 7. Out-of-Distribution Story on ChartFC

ChartFC results:

| Method | Acc% | Pass@1 | Pass@4 | C_table | D_reason | Coherence | Format% |
|---|---:|---:|---:|---:|---:|---:|---:|
| GRPO-FC | 63.8 | 66.10 | 92.2 | 0.950 | 0.059 | 0.312 | 98.8 |
| HCPC-FC | 60.6 | 62.45 | 91.8 | 0.782 | 0.060 | 0.433 | 66.4 |

## 7.1 GRPO wins most top-line OOD metrics

GRPO is clearly stronger on:

- Acc%
- Pass@1
- Pass@4
- C_table
- format compliance

This means you cannot honestly claim HCPC is better overall on this OOD benchmark.

The data do not support that.

## 7.2 HCPC wins coherence decisively

HCPC's OOD coherence:

- `0.433` vs GRPO's `0.312`

This is a large relative difference.

Interpretation:

- when HCPC gets the table right on OOD examples,
- it is more likely than GRPO to convert that into a correct answer.

This is one of the strongest and most interesting HCPC findings in the whole project.

Best framing:

**HCPC does not outperform GRPO on raw OOD accuracy, but it substantially improves reasoning integrity conditional on correct extraction.**

That is a precise and non-overclaiming statement.

## 7.3 HCPC's OOD format collapse is a real limitation

HCPC format compliance on ChartFC drops to `66.4%`, driven largely by:

- `<think>` tag failures,
- ordering failures,
- weaker table-tag performance than GRPO.

This should not be hidden.

Best framing:

**HCPC's hierarchical reward improves conditional extraction-to-answer coherence under shift, but currently sacrifices structural robustness on OOD formatting.**

That tradeoff is actually interesting. It suggests the method may be helping deeper internal alignment while hurting surface-output discipline under distribution shift.

## 7.4 How to avoid overclaiming on OOD

Do not say:

- "HCPC generalizes better OOD."
- "HCPC is more robust than GRPO on unseen charts."

Instead say:

- "HCPC shows a selective OOD advantage in coherence, not in top-line accuracy."
- "GRPO remains stronger on overall OOD performance, while HCPC shows better extraction-to-answer coupling once extraction succeeds."

That distinction is crucial.

---

## 8. Sample Overlap: What It Says About Method Complementarity

ChartQA solved samples:

- Base: `387`
- GRPO: `408`
- HCPC: `414`
- NSR: `410`

Solved by all methods:

- `344`

Universally failed:

- `52`

Contested:

- `82`

## 8.1 Most methods solve mostly the same problems

The Jaccard overlaps are high. That means the methods are not opening radically different capability regions.

Interpretation:

- all trained methods improve on a largely shared solvable set,
- differences among them are concentrated on a relatively small contested subset,
- this helps explain why significance is weak despite visible metric differences.

## 8.2 Unique solves still matter

Each method uniquely solves `4-5` samples.

That is small, but meaningful.

It means:

- the methods are not identical,
- each method shifts some boundary cases,
- ensemble or reranking approaches could potentially benefit from method diversity.

Best framing:

**The methods are largely overlapping but not redundant; their differences appear on a narrow contested subset rather than across the entire test distribution.**

## 8.3 Universally failed samples reveal the real difficulty frontier

The failed examples include:

- fine-grained numeric reading,
- ratio questions,
- subtle comparisons,
- line-identification or legend-referencing cases,
- small decimal precision.

This tells you where the 3B model still struggles:

- exact numeric extraction,
- compositional arithmetic,
- visually ambiguous chart elements,
- language-to-chart alignment in harder questions.

This is useful thesis evidence because it shows the remaining error profile is not random.

---

## 9. Chart-Type Analysis: What It Means and What It Does Not Mean

ChartQA by type:

- bar charts dominate the test set (`n=316`)
- line charts are second (`n=142`)
- pie charts are small (`n=41`)
- scatterplot is effectively unusable for interpretation (`n=1`)

## 9.1 Bar charts are where training helps most clearly

HCPC gets the best bar-chart result:

- Base: `76.6%`
- GRPO: `82.9%`
- HCPC: `85.8%`
- NSR: `83.9%`

This is a strong but still narrow claim:

**HCPC appears especially effective on bar-chart reasoning, where the hierarchical decomposition of type -> values -> comparison is relatively clean.**

That is plausible and consistent with the reward design.

## 9.2 Line charts remain difficult

Line-chart accuracy stays roughly flat:

- Base: `76.8%`
- GRPO: `74.6%`
- HCPC: `76.1%`
- NSR: `75.4%`

Interpretation:

- RL training did not substantially improve line-chart reasoning,
- line charts likely require harder trend reasoning, interpolation, temporal comparisons, or more subtle perceptual grounding,
- current rewards may be better aligned with extraction-heavy tasks than with nuanced line reasoning.

This is one of the clearest places to discuss future work.

## 9.3 Pie-chart results should be treated cautiously

Pie-chart sample size is small (`n=41`).

That means:

- differences may be noisy,
- conclusions should be tentative,
- do not build a major thesis narrative around pie-chart rankings.

---

## 10. Reasoning Length and Inference Time

## 10.1 Incorrect responses are usually longer

ChartQA:

- Base correct vs incorrect: `137` vs `160`
- GRPO: `144` vs `159`
- HCPC: `134` vs `149`
- NSR: `138` vs `160`

This is a robust pattern.

Interpretation:

- incorrect answers often involve wandering or compensatory reasoning,
- correctness is associated with more direct and economical reasoning,
- longer chain-of-thought is not evidence of better reasoning.

This is worth stating explicitly because many readers incorrectly assume more tokens means more reasoning quality.

## 10.2 HCPC is relatively concise in-distribution

HCPC has the shortest mean reasoning length among trained ChartQA methods:

- HCPC mean = `140`
- GRPO mean = `150`
- NSR mean = `146`

That may help explain why HCPC is also faster than GRPO in-distribution.

Possible interpretation:

- HCPC encourages more efficient structured reasoning once the intermediate representation stabilizes,
- or at least avoids some of GRPO's verbosity.

This is a plausible interpretation, but keep it modest.

## 10.3 NSR is the fastest on ChartQA

Inference times:

- Base total = `317.9` min
- GRPO total = `485.8` min
- HCPC total = `316.1` min
- NSR total = `115.4` min

NSR being much faster is striking.

Possible explanations:

- shorter or more decisive generations,
- fewer meandering chains,
- different stopping behavior,
- more concentrated output format behavior.

You should present this as an empirical observation, not a guaranteed property of NSR in general.

Best framing:

**In this experimental setup, NSR produced the most inference-efficient responses, suggesting a more concise generation style.**

---

## 11. Statistical Significance: What You Can and Cannot Claim

From the paired bootstrap:

- HCPC vs GRPO Pass@4:
  - observed difference = `+1.20pp`
  - one-sided `p = 0.2334`
  - 95% CI = `[-1.80pp, 4.20pp]`

- NSR vs GRPO Acc:
  - observed difference = `+0.40pp`
  - one-sided `p = 0.4157`
  - 95% CI = `[-2.40pp, 3.20pp]`

## 11.1 What this means

It means:

- the direction of the differences is suggestive,
- the study does not have enough evidence to claim decisive superiority,
- effect sizes are modest relative to sample variability,
- your method-level story should be about *behavioral profiles* more than *statistically proven ranking*.

## 11.2 Safe language

Use phrases like:

- "shows a trend toward"
- "appears to improve"
- "is consistent with the hypothesis that"
- "suggests"
- "we observe"
- "at the present experimental scale, differences are not statistically significant"

Avoid phrases like:

- "demonstrates clear superiority"
- "significantly outperforms" unless the statistics truly support it
- "proves that"

This matters a lot for credibility.

---

## 12. How to Frame the Main Thesis Narrative

If you want the cleanest, most defensible paper story, it should look like this.

## 12.1 Primary claim

**RLVR is effective for small-scale chart reasoning and structured explanation generation.**

This is strongly supported.

## 12.2 HCPC claim

**HCPC improves the quality of multi-rollout candidate sets, especially on Pass@4 and conditional extraction-to-answer coherence, indicating benefits for structured rollout robustness rather than universally higher top-line accuracy.**

This is the best HCPC framing.

## 12.3 NSR claim

**NSR transfers effectively to the chart reasoning setting and yields the strongest extraction stability with highly competitive single-shot and multi-rollout performance.**

This is the best NSR framing.

## 12.4 Limitation claim

**HCPC's OOD format degradation shows that better hierarchical reasoning behavior does not automatically translate into robust structural output under distribution shift.**

This is an important and honest limitation.

## 12.5 Overall comparative claim

**The trained methods differ less in absolute accuracy than in how they distribute correctness, structure, and consistency across rollouts.**

This is a sophisticated framing and fits your data very well.

---

## 13. Ready-to-Use Interpretation Templates

Use these as sentence templates in the thesis.

## 13.1 For the base-to-trained jump

"All RL-trained variants substantially outperform the untrained base model, indicating that verifiable-reward optimization effectively converts latent chart reasoning ability into more reliable structured behavior."

## 13.2 For HCPC on ChartQA

"GRPO+HCPC achieves the strongest Pass@4 performance, suggesting that hierarchical correct-path consistency is most beneficial when evaluation rewards the quality of the overall rollout set rather than only a single sampled response."

## 13.3 For NSR

"NSR attains the highest in-distribution accuracy and table consistency, consistent with the hypothesis that suppressing incorrect trajectories can stabilize the model's internal chart representation without aggressively collapsing onto a single correct path."

## 13.4 For D_reason

"Although HCPC does not maximize the standalone reasoning-diversity metric, its gains on Pass@4 and OOD coherence suggest that useful rollout robustness is not captured by diversity alone."

## 13.5 For OOD ChartFC

"On ChartFC, GRPO remains stronger on top-line performance, whereas HCPC shows a selective advantage in coherence, indicating better conversion of correct intermediate extraction into correct final answers once extraction succeeds."

## 13.6 For statistical caution

"The relative ordering among trained methods should be interpreted cautiously, as paired bootstrap tests indicate that the observed differences remain within the range of sampling variability at the present experimental scale."

---

## 14. Common Misinterpretations to Avoid

## 14.1 "HCPC improves diversity because that was the design goal"

Not safe. Your measured `D_reason` does not show HCPC clearly leading.

Safer:

"HCPC improves multi-rollout robustness and may preserve useful diversity in combination with structural consistency."

## 14.2 "Higher C_table means better reasoning"

Wrong.

`C_table` is stability of extraction, not proof of correctness.

## 14.3 "Coherence dropped after training, so reasoning got worse"

Misleading.

The metric becomes stricter and applies over many more structured extraction attempts after training.

## 14.4 "HCPC generalizes better OOD"

Too broad.

HCPC only shows a selective OOD advantage in coherence, not in top-line accuracy or formatting.

## 14.5 "NSR is better because it has the best Acc%"

Too simplistic.

NSR has the best Acc%, but HCPC has the best Pass@4 and a different behavioral profile.

---

## 15. What the Results Suggest Mechanistically

These are not hard claims. These are the best *mechanistic interpretations* supported by the pattern of evidence.

## 15.1 GRPO

GRPO likely improves performance by:

- strongly reinforcing high-reward structural patterns,
- making outputs more parseable,
- improving the probability of task-appropriate behavior,
- but possibly narrowing the response distribution somewhat.

## 15.2 HCPC

HCPC likely improves performance by:

- rewarding agreement on intermediate perceptual structure among successful rollouts,
- favoring rollout sets that are structurally stable but not fully collapsed,
- helping correct extraction translate more reliably into correct reasoning under some conditions,
- especially noticeable in Pass@4 and OOD coherence.

## 15.3 NSR

NSR likely improves performance by:

- suppressing obviously bad trajectories,
- preserving more of the model's pre-existing plausible alternatives,
- stabilizing extraction without overcommitting to one rewarded path,
- which is consistent with strong `C_table` and competitive Pass@K.

These are hypotheses grounded in your empirical profile and in prior NSR literature, but they should still be presented as interpretations, not direct proofs.

---

## 16. Best Structure for a Thesis Discussion Section

If you want to explain these results clearly in a thesis chapter, this is a good order.

1. Start with the base-to-trained improvement.
2. Then compare the trained methods on ChartQA.
3. Then explain why Pass@K is central to the HCPC story.
4. Then discuss `C_table`, `D_reason`, and coherence as behavioral diagnostics.
5. Then discuss OOD ChartFC and be explicit about the HCPC tradeoff.
6. End with overlap analysis, limitations, and future work.

That order works because:

- it begins with the strongest result,
- then moves to subtler comparisons,
- then addresses potential contradictions,
- then ends with honest limitations.

---

## 17. Future Work That Naturally Follows from These Results

The results suggest several very natural next experiments.

## 17.1 Increase rollout count during training and evaluation

Since HCPC is a cross-rollout method, `K=4` may be too small to fully reveal its advantages.

## 17.2 Add explicit OOD format regularization

HCPC's OOD format drop suggests that format should be stabilized more explicitly under distribution shift.

## 17.3 Separate extraction diversity from reasoning diversity more carefully

Your current metrics show that stability and correctness are not aligned in a simple way. A more fine-grained decomposition would be valuable.

## 17.4 Evaluate larger models

A 7B or larger VLM could reveal whether these method differences widen or shrink with capacity.

## 17.5 Analyze contested samples qualitatively

The `82` contested examples are likely where the real behavioral differences live. A qualitative case study there would be highly valuable.

---

## 18. Final Takeaways

If you remember only a few things from this guide, remember these.

1. The clearest result is that RLVR works very well for this task.
2. HCPC's strongest case is not raw Acc%; it is candidate-set quality, Pass@4, and conditional coherence.
3. NSR's strongest case is stable extraction and highly competitive accuracy.
4. `C_table`, `D_reason`, and coherence are diagnostic metrics, not simple "higher is always better" scores.
5. HCPC's OOD format drop is a real limitation and should be discussed directly.
6. Differences among trained methods are meaningful trends, but not statistically decisive at this scale.
7. The most accurate overall framing is that the methods differ in how they shape the rollout distribution, not just in one final number.

---

## 19. Reference Papers and Sources

These are the main sources most relevant to the interpretation of your results.

### Core experimental context

- Sinha et al., *Chart-RVR: Reinforcement Learning with Verifiable Rewards for Explainable Chart Reasoning*.
  Link: `https://arxiv.org/abs/2510.10973`
  Relevance: provides the RLVR chart-reasoning framing, GRPO-based chart reward design, and the motivation for structured chart intermediate rewards.

- Local copy used in this repo:
  `app/docs/chart-rvr-paper.txt`

### NSR

- Zhu et al., *The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning*.
  Link: `https://arxiv.org/abs/2506.01347`
  Relevance: motivates NSR as a way to suppress incorrect trajectories while preserving alternative plausible paths.

- Local copy used in this repo:
  `app/docs/nsr-paper.txt`

### Pass@K

- Chen et al., *Evaluating Large Language Models Trained on Code*.
  Link: `https://arxiv.org/abs/2107.03374`
  Relevance: standard unbiased Pass@K estimator used in your analysis.

### LoRA

- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*.
  Link: `https://arxiv.org/abs/2106.09685`
  Relevance: parameter-efficient fine-tuning method used in your experimental setup.

### Chart benchmark background

- Masry et al., *ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning*.
  Link: `https://arxiv.org/abs/2203.10244`
  Relevance: in-distribution benchmark background.

- Methani et al., *PlotQA: Reasoning over Scientific Plots*.
  Link: `https://arxiv.org/abs/1909.00997`
  Relevance: chart reasoning benchmark background cited in the broader Chart-RVR setup.

### Project-local analysis sources

- `methodology_drafts/detailed_analysis.md`
- `methodology_drafts/experimental_results_draft.md`
- `app/scripts/analyze_results.py`
- `app/docs/CHARTQA_QWEN_COMPARISON_20260302.md`

These project-local files are the direct basis for all experiment-specific interpretations in this study guide.
