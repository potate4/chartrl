# ChartQA Evaluation Comparison: Base Qwen vs GRPO Qwen vs GRPO-HCPC Qwen

## Overview

This document compares three Qwen2.5-VL-3B-Instruct variants on the ChartQA test split:

1. Base Qwen
2. GRPO-trained Qwen
3. GRPO-HCPC-trained Qwen

The goal is to summarize the experimental setup, compare the observed results, and explain why the models behaved this way.

## Source Runs

- Base: `app/outputs/eval_results/main_base_chartqa_20260218_210622/summary.json`
- GRPO baseline: `app/outputs/eval_results/grpo_baseline_checkpoint-2000_chartqa_20260219_073124/summary.json`
- GRPO baseline verification: `app/outputs/eval_results/grpo_baseline_checkpoint-2000_chartqa_20260219_073124/per_sample.jsonl`
- GRPO-HCPC: `app/outputs/eval_results/grpo_hcpc_updated_checkpoint-2000_chartqa_20260301_170306/summary.json`

## Experimental Setup

### Base model

- Backbone: `Qwen/Qwen2.5-VL-3B-Instruct`
- Task: chart question answering with structured reasoning output
- Output format expected by the evaluator:
  - `<think>`
  - `<type>...</type>`
  - `<table>...</table>`
  - stepwise reasoning
  - `</think>`
  - `<answer>...</answer>`

### Evaluation setup

- Dataset: `chartqa`
- Split: `test`
- Evaluated subset size: `500`
- Decoding:
  - temperature = `0.8`
  - top-p = `0.95`
  - max new tokens = `768`
- Main reported answer metric:
  - `accuracy` is first-rollout relaxed accuracy
- Multi-rollout metrics:
  - `pass@1`, `pass@2`, `pass@4`
  - `c_table`: consistency of table extraction among correct rollouts
  - `d_reason`: diversity of reasoning among correct rollouts
  - `coherence`: whether reasoning references extracted table values
  - `correct_rate`: fraction of correct rollouts
  - `format_compliance_rate`: fraction of first responses that fully follow the required output format

### Training setup for the two RL models

From the saved training configs:

- Policy method: `GRPO`
- Epochs: `2`
- Batch size: `2`
- Gradient accumulation: `2`
- Learning rate: `1e-5`
- LoRA:
  - rank `8`
  - alpha `16`
  - dropout `0.05`
  - target modules: `q_proj`, `v_proj`
- Training subset size: `1000`
- Number of rollouts during training: `4`

### Reward difference between GRPO baseline and GRPO-HCPC

GRPO baseline uses the base Chart-RVR-style reward stack, including:

- format reward
- answer accuracy reward
- length reward
- token-count reward
- chart-type reward
- table reward
- process reward

GRPO-HCPC changes the objective:

- `process_reward` is disabled
- `HCPC` is enabled
- HCPC rewards:
  - high chart-type consistency among correct rollouts
  - high table consistency among correct rollouts
  - high reasoning diversity among correct rollouts

In short:

- GRPO baseline tries to match the expected reasoning path more directly
- GRPO-HCPC tries to encourage multiple valid correct reasoning paths while keeping chart type and table extraction stable

## Important Evaluation Note

The GRPO baseline `summary.json` contains a stale `pass@8` entry caused by a resume mismatch. Per your instruction, that value is ignored here.

For fairness, the GRPO baseline comparison below uses the verified 4-rollout data from `per_sample.jsonl`, which matches the evaluator log (`Rollouts: 4`).

## Quantitative Results

| Model | Accuracy | Pass@1 | Pass@2 | Pass@4 | C_table | D_reason | Coherence | Correct rate | Format rate | Avg time/sample |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base Qwen | 0.5180 | 0.5245 | 0.6733 | 0.7740 | 0.5914 | 0.0400 | 0.5452 | 0.5245 | 0.2980 | 38.14s |
| GRPO Qwen | 0.6300 | 0.6188 | 0.7310 | 0.8031 | 0.7493 | 0.0579 | 0.2516 | 0.6188 | 0.9680 | 58.30s |
| GRPO-HCPC Qwen | 0.6220 | 0.6285 | 0.7457 | 0.8280 | 0.7412 | 0.0521 | 0.2444 | 0.6540 | 0.9820 | 37.93s |

## Main Findings

### 1. Both RL-trained models clearly outperform the base model

Compared with base Qwen:

- GRPO improves accuracy from `0.518` to `0.630` (`+11.2` points)
- GRPO-HCPC improves accuracy from `0.518` to `0.622` (`+10.4` points)
- Both RL models dramatically improve formatting:
  - Base: `29.8%`
  - GRPO: `96.8%`
  - GRPO-HCPC: `98.2%`
- Both RL models also improve table stability:
  - Base `c_table = 0.5914`
  - GRPO `c_table = 0.7493`
  - GRPO-HCPC `c_table = 0.7412`

This indicates that RL training strongly helped the model internalize the required answer structure and the chart-to-table extraction routine.

### 2. GRPO baseline is best on first-answer accuracy, but only slightly

Among the two trained models:

- GRPO has the best first-rollout accuracy: `0.630`
- GRPO-HCPC is very close: `0.622`

The gap is small: only `0.8` percentage points.

That means HCPC does not hurt answer accuracy much on in-distribution ChartQA, even though it optimizes a different process objective.

### 3. GRPO-HCPC is best on multi-rollout success

GRPO-HCPC has the best `pass@1`, `pass@2`, and `pass@4`:

- `pass@1 = 0.6285`
- `pass@2 = 0.7457`
- `pass@4 = 0.8280`

This is especially important because HCPC is designed around cross-rollout behavior. The result suggests that HCPC improves the overall quality of the rollout set, even if the very first rollout is not always the single strongest one.

### 4. GRPO-HCPC has the highest correct-rollout rate

- Base: `0.5245`
- GRPO: `0.6188`
- GRPO-HCPC: `0.6540`

This is one of the strongest signals in favor of HCPC. It means that, across generated candidates, HCPC produces correct answers more often.

That aligns well with the intended HCPC design: reward groups of correct rollouts that are consistent on chart type and table extraction while allowing reasoning variation.

### 5. Coherence is much higher for the base model, but this should be interpreted carefully

- Base coherence: `0.5452`
- GRPO coherence: `0.2516`
- GRPO-HCPC coherence: `0.2444`

At first glance this looks bad for the trained models, but the metric definition matters. The evaluation computes coherence over all rollouts and checks whether reasoning text explicitly reuses extracted table values. A model can therefore be accurate and format-compliant while still scoring lower if:

- it uses shorter or more abstract reasoning
- it answers directly without repeatedly quoting numbers from the table
- it produces cleaner templates with less verbose numeric repetition

So this drop likely reflects a style shift in reasoning rather than a simple loss of capability.

## Analysis: Why the Results Look Like This

### Why both RL models beat base Qwen

The base model starts from a general-purpose instruction-following checkpoint. It can answer many questions correctly, but it was not specifically optimized for:

- strict XML-like structured output
- explicit chart type prediction
- table extraction in the required JSON schema
- stable chart reasoning under repeated stochastic sampling

After GRPO training, the model receives repeated reward pressure on exactly these behaviors. That explains the large gains in:

- answer accuracy
- format compliance
- table consistency
- multi-sample success rate

### Why GRPO baseline slightly wins on first-answer accuracy

The baseline includes `process_reward`, which favors similarity to the target reasoning path. That is a stronger direct bias toward the reference-style solution and can help the first sampled answer become more aligned with the evaluation target.

In contrast, HCPC does not directly reward matching one preferred reasoning trace. It rewards a set-level property over correct rollouts. Because of that, HCPC may trade a small amount of single-rollout sharpness for better rollout-set quality.

This matches the observed pattern:

- GRPO baseline has the best top-1 answer accuracy
- HCPC has better rollout-set behavior (`pass@k`, `correct_rate`, format rate)

### Why HCPC wins on pass@k and correct rate

HCPC explicitly rewards correct-rollout groups that are:

- consistent in chart type
- consistent in table extraction
- diverse in reasoning

This can improve robustness under sampling. Instead of collapsing to one narrow reasoning pattern, the model learns that several reasoning paths are acceptable if they preserve the important intermediate structure.

That makes it more likely that at least one or more rollouts in a sampled set are correct, which is exactly what `pass@k` and `correct_rate` capture.

### Why GRPO and HCPC have similar table consistency

Both trained models heavily outperform base Qwen on `c_table`, and the difference between them is very small:

- GRPO: `0.7493`
- HCPC: `0.7412`

This suggests that RL training itself already teaches stable table extraction, while HCPC’s extra gain is showing up more in rollout-level correctness than in pure table-consistency magnitude on this in-distribution benchmark.

### Why HCPC is faster than GRPO in this evaluation

The measured average times are:

- GRPO: `58.30s`
- HCPC: `37.93s`
- Base: `38.14s`

Since decoding settings were the same, the likely explanation is not algorithmic evaluation overhead but generation behavior:

- GRPO outputs may have been longer on average
- GRPO may have produced more verbose reasoning traces
- resumed or fragmented execution may also have affected timing in one run

So the timing difference should be treated as an empirical observation from these runs, not as proof that HCPC is intrinsically faster than GRPO.

## Overall Interpretation

If the goal is the single best first answer on ChartQA test, the current GRPO baseline is marginally best.

If the goal is stronger structured generation quality and more robust multi-rollout behavior, GRPO-HCPC is the better model in this comparison because it gives:

- near-best accuracy
- best `pass@k`
- best correct-rollout rate
- best format compliance
- nearly the same table consistency as GRPO
- much lower latency than the recorded GRPO baseline run

This is a favorable result for the HCPC thesis argument. HCPC does not need to beat the baseline on every scalar metric to be useful. Its value is that it shifts the model toward more reliable rollout sets and more controllable structured chart reasoning.

## Conclusion

The experiment shows three clear outcomes:

1. RL training is effective for chart reasoning with Qwen2.5-VL-3B-Instruct.
2. Standard GRPO gives the best first-answer accuracy on this ChartQA test subset.
3. GRPO-HCPC gives the best overall rollout robustness, with the strongest `pass@k`, `correct_rate`, and format compliance.

So the most defensible thesis claim is not that HCPC universally dominates plain GRPO on in-distribution top-1 accuracy. The stronger claim is that HCPC improves structured reasoning reliability and candidate-set quality while preserving nearly all of the answer accuracy gain obtained from GRPO training.
