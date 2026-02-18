# eval_run.py — Usage Guide

## Core Concept

**One run = one model + one dataset.** Each invocation is independent. If a run crashes, nothing else is affected. You build your comparison matrix by running the script multiple times with different args.

## All commands run from `app/`

```bash
cd c:\Users\sumai\DATA\RESEARCH\THESIS\chartrl\app
```

---

## 1. Evaluate Base Qwen (no checkpoint)

```bash
# ChartQA (in-distribution), 200 samples
python scripts/eval_run.py --dataset chartqa --subset 200

# EvoChart (out-of-distribution), 200 samples
python scripts/eval_run.py --dataset evochart --subset 200

# ChartQA Pro
python scripts/eval_run.py --dataset chartqapro --subset 200
```

Omitting `--checkpoint` loads the raw `Qwen/Qwen2.5-VL-3B-Instruct` model.

---

## 2. Evaluate a LoRA Checkpoint

```bash
# GRPO baseline checkpoint-70
python scripts/eval_run.py --dataset chartqa --subset 200 \
  --checkpoint outputs/grpo_baseline/run_20260217_133942/trl_output/checkpoint-70

# Same checkpoint on EvoChart (OOD)
python scripts/eval_run.py --dataset evochart --subset 200 \
  --checkpoint outputs/grpo_baseline/run_20260217_133942/trl_output/checkpoint-70
```

Any directory containing `adapter_config.json` + `adapter_model.safetensors` works.

---

## 3. All CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset`, `-d` | **required** | `chartqa`, `evochart`, `chartqapro`, or any HF dataset path |
| `--checkpoint`, `-c` | `None` | Path to LoRA checkpoint dir. Omit = base model |
| `--base-model` | `Qwen/Qwen2.5-VL-3B-Instruct` | Base model HF name |
| `--split` | `test` | Dataset split |
| `--subset` | `None` (all) | Evaluate first N samples only |
| `--num-samples`, `-n` | `8` | Rollouts per question (for Pass@k and diversity) |
| `--temperature` | `0.8` | Sampling temperature |
| `--top-p` | `0.95` | Top-p sampling |
| `--max-new-tokens` | `768` | Max generation length |
| `--output-dir` | `outputs/eval_results` | Root folder for results |
| `--run-name` | auto-generated | Custom label for this run |
| `--cache-dir` | `./cache` | HF cache dir |
| `--seed` | `42` | Random seed |

---

## 4. What You See During a Run

```
14:32:01 | ============================================================
14:32:01 | HCPC-RLVR Evaluation
14:32:01 | ============================================================
14:32:01 | Model:      base
14:32:01 | Dataset:    chartqa (split=test)
14:32:01 | Subset:     200
14:32:01 | Rollouts:   8
14:32:01 | ============================================================
14:32:01 | Loading model...
14:32:18 | Model loaded in 17.2s
14:32:20 | Dataset loaded: 200 samples
14:32:32 | [1/200] Q: "What is the total revenue?" | Label: 250 | Pred: 250 | OK | Acc: 1.000 (12.3s)
14:32:40 | [2/200] Q: "Which year had the highest..." | Label: 2019 | Pred: 2020 | WRONG | Acc: 0.500 (8.1s)
...
14:58:15 | [200/200] Q: "Is the trend increasing?" | Label: Yes | Pred: Yes | OK | Acc: 0.720 (7.5s)
14:58:15 |
14:58:15 | ========================================
14:58:15 |        FINAL RESULTS
14:58:15 | ========================================
14:58:15 | Accuracy:     72.00%
14:58:15 | Pass@1:       72.00%
14:58:15 | Pass@2:       78.00%
14:58:15 | Pass@4:       85.00%
14:58:15 | Pass@8:       91.00%
14:58:15 | C_table:      0.820
14:58:15 | D_reason:     0.340
14:58:15 | Coherence:    0.650
14:58:15 | Correct Rate: 0.450
14:58:15 | Format Rate:  88.00%
14:58:15 | Avg Time:     7.8s/sample
14:58:15 | Total Time:   26m 15s
14:58:15 | ========================================
14:58:15 | Results saved to: outputs/eval_results/base_chartqa_20260218_143201
```

---

## 5. Output Files

Each run creates a folder in `outputs/eval_results/`:

```
outputs/eval_results/base_chartqa_20260218_143201/
  |- config.json          # Exact CLI args used
  |- summary.json         # All aggregate metrics
  |- per_sample.jsonl     # Every sample's details
  |- eval.log             # Full console log (for later review)
```

### summary.json

What you use for your paper tables:

```json
{
  "model": "base",
  "accuracy": 0.72,
  "pass_at_k": {"1": 0.72, "2": 0.78, "4": 0.85, "8": 0.91},
  "c_table": 0.82,
  "d_reason": 0.34,
  "coherence": 0.65,
  "correct_rate": 0.45,
  "format_compliance_rate": 0.88,
  "avg_time_per_sample": 7.8,
  "total_time_human": "26m 15s"
}
```

### per_sample.jsonl

One JSON line per sample, for debugging or deep analysis. Each line contains:

- `question`, `label` — the input
- `predictions` — all 8 extracted answers
- `raw_outputs` — full model text for all 8 rollouts
- `correct` — which rollouts were right `[true, true, false, ...]`
- `reward_accuracy` — score from `accuracy_reward()` in base_rewards.py
- `relaxed_accuracy` — score from `relaxed_accuracy()` in metrics.py
- `running_accuracy` — cumulative accuracy up to this sample
- `format_compliance` — did it use `<think>`, `<answer>`, `<type>`, `<table>` tags
- `diversity` — `c_table`, `d_reason`, `coherence` for that sample
- `time_seconds` — inference time for that sample

---

## 6. Full Evaluation Matrix

Run these 4 commands to get your GRPO baseline comparison data:

```bash
# 1. Base model on ChartQA (ID)
python scripts/eval_run.py -d chartqa --subset 200 --run-name base_chartqa

# 2. Base model on EvoChart (OOD)
python scripts/eval_run.py -d evochart --subset 200 --run-name base_evochart

# 3. GRPO checkpoint on ChartQA (ID)
python scripts/eval_run.py -d chartqa --subset 200 --run-name grpo70_chartqa \
  -c outputs/grpo_baseline/run_20260217_133942/trl_output/checkpoint-70

# 4. GRPO checkpoint on EvoChart (OOD)
python scripts/eval_run.py -d evochart --subset 200 --run-name grpo70_evochart \
  -c outputs/grpo_baseline/run_20260217_133942/trl_output/checkpoint-70
```

Then compare the 4 `summary.json` files side by side. Later when you train `grpo_hcpc`, just add 2 more runs.

---

## 7. Speed Estimates (3090 24GB)

- ~3B model in bfloat16 uses ~6-7GB VRAM
- With 8 rollouts per sample: ~40-80s per sample
- 200 samples: ~2-4 hours per run
- Use `--num-samples 1` for a quick accuracy-only pass (~15-20 min for 200 samples, but no Pass@k or diversity)

---

## 8. Tips

- **Quick sanity check**: `--subset 5 --num-samples 2` to verify everything works before a full run
- **Custom run names**: use `--run-name` to keep results organized (`base_chartqa`, `grpo70_chartqa`, `hcpc70_chartqa`)
- **Fewer rollouts = faster**: `--num-samples 1` skips diversity/Pass@k but gives you accuracy in 1/8 the time
- **Resume after crash**: not built-in, but `per_sample.jsonl` is flushed after each sample so you can see exactly where it stopped

---

## 9. Metrics Glossary

| Metric | What it measures |
|--------|-----------------|
| **Accuracy** | Fraction of first-rollout answers that match the label (5% numeric tolerance) |
| **Pass@k** | Fraction of questions with at least 1 correct answer in k rollouts (unbiased estimator) |
| **C_table** | Table extraction consistency — avg pairwise similarity of extracted tables across correct rollouts |
| **D_reason** | Reasoning diversity — `1 - avg_pairwise_semantic_similarity` among correct rollouts |
| **Coherence** | Cross-level coherence — does the reasoning reference values from the extracted table |
| **Correct Rate** | Fraction of rollouts that are correct (across all rollouts, not just first) |
| **Format Rate** | Fraction of samples with fully compliant output format (`<think>`, `<type>`, `<table>`, `<answer>`) |
