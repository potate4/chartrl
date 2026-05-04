# Conversation History — HCPC-RLVR Thesis Project
*Last updated: 2026-04-21*

---

## Who You Are
- Thesis student working on chart QA with vision-language models
- Email: sumaiya@oleyn.ai
- Project path: `C:\Users\sumai\DATA\RESEARCH\THESIS\chartrl` (git branch: `feat/predef`)

---

## What the Thesis Is About

**Two novel contributions:**
1. **HCPC (Hierarchical Correct-Path Consistency)** — cross-rollout reward encouraging consistency in table extraction and diversity in reasoning across multiple rollouts
2. **NSR (Negative Sample Reward, Zhu et al. 2025)** applied to a VLM for the first time — originally text-LLM only

**Base framework:** Chart-RVR (Sinha et al., 2025) — 7-component reward for chart QA
**Training algorithm:** GRPO (Shao et al. 2024)
**Model:** Qwen2.5-VL-3B-Instruct + LoRA (r=8, α=16, q_proj/v_proj)

---

## Hardware
- **Local PC:** NVIDIA RTX 3090 24GB VRAM
- **Kaggle:** NVIDIA H100 40GB

---

## Training Configuration
- LR: 1e-5 | Rollouts K=4 | Temp: 1.0 (train), 0.8 (eval) | Top-p: 0.95
- Steps: 2000 | Precision: bf16 | Batch: 2 | Grad accum: 2

---

## HCPC Reward Formula

```
R⁺ = correct rollouts (answer matches ground truth)

Consistency →  C_table(R⁺)  = cosine_sim(tables across R⁺)
Diversity   →  D_reason(R⁺) = jaccard_dist(step sets across R⁺)

R_HCPC = 1.0 · type_consensus
        + 2.0 · C_table(R⁺)
        + 1.5 · D_reason(R⁺)
```

---

## Dataset: Chart-RVR GRPO Train (`sanchit97/chart-rvr-grpo-train`)

**Verified by reading HuggingFace Arrow blobs directly (4 blobs, ~1.78 GB total).**

### Full Dataset — 34,194 samples

| Chart Type | Count | Proportion | Numerical Labels | Avg Label Len | Avg Query Len | Avg Reasoning Len |
|---|---|---|---|---|---|---|
| Bar | 13,682 | 40.0% | 46.4% | 5.1 | 64 chars | 605 chars |
| Line | 3,682 | 10.8% | 67.9% | 5.4 | 64 chars | 547 chars |
| Stacked Bar | 2,740 | 8.0% | 71.9% | 6.1 | 69 chars | 554 chars |
| Pie | 2,136 | 6.2% | 61.4% | 5.2 | 58 chars | 476 chars |
| Scatterplot | 540 | 1.6% | 54.4% | 7.0 | 87 chars | 685 chars |
| Bubble | 8 | 0.0% | 75.0% | 8.8 | 140 chars | 660 chars |
| No annotation* | 11,398 | 33.3% | 54.5% | 5.3 | 65 chars | — |
| **Total** | **34,194** | **100%** | **54.5%** | 5.3 | 65 chars | 579 chars |

*11,398 raw rows have only image/query/label/prompt — no chart_type/reasoning/table fields.*

### Training Subset — first 1,000 samples
**Selection:** `dataset.select(range(1000))` — sequential, NO shuffle applied.

| Chart Type | Count | Proportion | Numerical Labels | Avg Query Len | Avg Reasoning Len |
|---|---|---|---|---|---|
| Bar | 485 | 48.5% | 69.5% | 61 chars | 488 chars |
| Line | 223 | 22.3% | 64.6% | 60 chars | 535 chars |
| Stacked Bar | 147 | 14.7% | 72.1% | 68 chars | 527 chars |
| Pie | 145 | 14.5% | 62.1% | 59 chars | 466 chars |
| **Total** | **1,000** | **100%** | **67.7%** | 62 chars | 501 chars |

No scatterplot or bubble in first 1K. Higher numerical rate (67.7% vs 54.5% full set).

### ChartQA Eval Set — 500 samples (in-distribution)
*Chart type from model's `<type>` tag in eval output. "Unknown" = parse failure.*

| Chart Type | Count | Proportion | Numerical Labels | Avg Label Len | Avg Query Len |
|---|---|---|---|---|---|
| Bar | 305 | 61.0% | 62.0% | 4.5 | 57 chars |
| Line | 124 | 24.8% | 54.8% | 5.4 | 64 chars |
| Pie | 41 | 8.2% | 48.8% | 4.6 | 52 chars |
| Stacked Bar | 10 | 2.0% | 90.0% | 2.9 | 63 chars |
| Scatterplot | 3 | 0.6% | 66.7% | 3.0 | 48 chars |
| Unknown/Parse | 17 | 3.4% | 82.4% | 3.9 | 58 chars |
| **Total** | **500** | **100%** | **60.4%** | 4.7 | 58 chars |

---

## Metric Definitions

| Metric | Definition |
|---|---|
| **Acc%** | `relaxed_accuracy` in per_sample.jsonl — fraction of samples with at least 1 correct rollout (relaxed string match) |
| **Correct Rate** | Total correct rollouts / total rollouts globally. Different from Pass@1. `summary.json` field: `correct_rate` |
| **Pass@1** | Per-sample `sum(correct)/len(correct)`, averaged across samples. `summary.json` field: `pass_at_k['1']` |
| **Pass@K** | Unbiased estimator `1 - C(n-c,k)/C(n,k)` per sample, averaged |
| **C_table** | Cosine similarity of extracted tables across correct rollouts — perception stability |
| **D_reason** | Jaccard distance of reasoning step sets across correct rollouts — strategy diversity |
| **Coherence** | P(correct answer | correct table) — reasoning pipeline integrity |
| **Format%** | Fraction fully compliant with `<think>...</think><answer>...</answer>` |

### Critical Pass@K Formula (bug was fixed)
```python
def pass_at_k(correct_matrix, k):
    from math import comb
    scores = []
    for correct_list in correct_matrix:
        n = len(correct_list)
        c = sum(correct_list)
        if n < k:
            scores.append(1.0 if c > 0 else 0.0)
        elif c == 0:
            scores.append(0.0)
        elif n - c < k:        # CORRECT — not "c >= k"
            scores.append(1.0)
        else:
            scores.append(1.0 - comb(n - c, k) / comb(n, k))
    return np.mean(scores) * 100
```

---

## All Experimental Results

### ChartQA — In-Distribution (500 samples, checkpoint-2000)

| Method | Acc% | Correct Rate | Pass@1 | Pass@2 | Pass@4 | C_table | D_reason | Coherence | Format% |
|---|---|---|---|---|---|---|---|---|---|
| Base | 51.8 | 52.45 | 52.45 | 67.33 | 77.4 | 0.591 | 0.040 | 0.545 | 29.8 |
| GRPO | 63.0 | 61.88 | 61.88 | 73.10 | 80.31 | 0.749 | 0.058 | 0.252 | 96.8 |
| GRPO+HCPC | 62.2 | **65.40** | 62.85 | 74.57 | **82.80** | 0.741 | 0.052 | 0.244 | **98.2** |
| NSR | **63.4** | 64.80 | 62.20 | 73.50 | 82.00 | **0.779** | 0.057 | 0.245 | — |

### ChartFC — Out-of-Distribution (500 samples, checkpoint-2000)

| Method | Acc% | Correct Rate | Pass@1 | Pass@2 | Pass@4 | C_table | D_reason | Coherence | Format% |
|---|---|---|---|---|---|---|---|---|---|
| GRPO | **63.8** | **66.10** | **66.10** | **82.07** | **92.2** | **0.950** | 0.059 | 0.312 | **98.8** |
| GRPO+HCPC | 60.6 | 62.40 | 62.45 | 79.90 | 91.8 | 0.782 | **0.060** | **0.433** | 66.4 |

---

## Key Narratives for the Defense

1. **HCPC generates more correct rollouts** — highest Correct Rate (65.4%) despite lower Acc% than GRPO. Generates more correct solutions per sample even if greedy pick is slightly worse.
2. **HCPC best Pass@4 on ChartQA (82.8%)** — at inference with multiple samples, HCPC wins. Better with self-consistency voting.
3. **NSR first VLM application** — matches GRPO accuracy, highest C_table (0.779) = most stable table extraction.
4. **HCPC boosts OOD coherence +39%** — 0.312 → 0.433 on ChartFC. More robust reasoning pipeline on unseen distributions.
5. **Format compliance trivially learned** — 29.8% → 96%+ after any RLVR training. Not a differentiator.
6. **HCPC OOD format drop (66.4%)** — honest limitation: HCPC may deprioritize format on unfamiliar data to focus on content.
7. **D_reason low (~0.05) everywhere** — K=4 rollouts insufficient for diversity signal. Future: K=8.
8. **Sample-efficient RLVR** — meaningful gains from only 1K/34K samples. Consistent with Wang et al. 2025.

---

## Per-Sample JSONL Paths (all verified, 500 lines each)

| Experiment | Path |
|---|---|
| Base ChartQA | `app/outputs/eval_results/main_base_chartqa_20260218_210622/per_sample.jsonl` |
| GRPO ChartQA | `app/outputs/eval_results/grpo_baseline_checkpoint-2000_chartqa_20260219_073124/per_sample.jsonl` |
| HCPC ChartQA | `app/outputs/eval_results/grpo_hcpc_updated_checkpoint-2000_chartqa_20260301_170306/per_sample.jsonl` |
| NSR ChartQA | `app/outputs/rakhi_outputs_full (1)/kaggle/working/chartrl/app/outputs/eval_results/checkpoint-2000_chartqa_20260301_084802/per_sample.jsonl` |
| GRPO ChartFC | `app/outputs/chartfc_grpo_outputs_full (1)/kaggle/working/chartrl/app/outputs/eval_results/checkpoint-2000_chartfc_20260310_114351/per_sample.jsonl` |
| HCPC ChartFC | `app/outputs/chartfc_grpo_hcpc_outputs_fullhcpc/kaggle/working/chartrl/app/outputs/eval_results/checkpoint-2000_chartfc_20260311_064248/per_sample.jsonl` |

### Per-Sample Schema
```
idx, question, label, chart_type, predictions[], raw_outputs[], correct[],
reward_accuracy, relaxed_accuracy, running_accuracy,
parsed_first{type, answer, parse_success, table_parse_success_strict},
format_compliance{has_think_tags, has_answer_tags, fully_compliant},
diversity{c_table, d_reason, coherence, correct_rate},
time_seconds
```

---

## Files Created/Modified

| File | Contents |
|---|---|
| `app/scripts/analyze_results.py` | Reads all 6 per_sample.jsonl, computes 12 analysis sections, verified Pass@K formula |
| `methodology_drafts/detailed_analysis.md` | Auto-generated analysis output (252 lines) |
| `methodology_drafts/experimental_results_draft.md` | Academic draft Sections 4 (Experimental Setup) + 5 (Results), ~290 lines |
| `methodology_drafts/study_guide_claude.md` | 600+ line guide explaining every metric and result with citations |
| `methodology_drafts/predefense.md` | Slides 8–14 with [SLIDE]/[SCRIPT]/[STUDY] sections, quick-reference table, glossary |
| `methodology_drafts/slide_content_senior_format.md` | Senior-format slide content slides 8–14, full dataset stats tables, results tables with Correct Rate column |
| `methodology_drafts/history.md` | This file |

---

## Corrections Made (Things That Were Previously Wrong)

| Wrong | Correct |
|---|---|
| Dataset ~30,000 samples | 34,194 (verified from HuggingFace Arrow blobs) |
| "Stratified sampling" for 1K subset | Sequential `dataset.select(range(1000))`, no shuffle |
| Hardware = T4 GPU × 2 (Kaggle) | H100 40GB (Kaggle) + RTX 3090 24GB (local PC) |
| Pass@K condition `elif c >= k` | `elif n - c < k` — correct unbiased estimator |
| Acc% computed as `any(correct)` | Acc% = `relaxed_accuracy` field from per_sample.jsonl |
| Bar=316, Line=142, Pie=41 claimed as "training data" | Those numbers are from the ChartQA eval set (500 samples), not training |
| Dataset has `chart_type` column in raw Arrow files | Raw blobs only have image/query/label/prompt; chart_type is in annotated/cache blobs |

---

## References to Cite

| Citation | Why |
|---|---|
| Chart-RVR (Sinha et al., 2025) | Base 7-component reward framework you extend |
| GRPO (Shao et al., 2024) | Training algorithm |
| DeepSeek-R1 (Guo et al., 2025) | RLVR motivation |
| NSR (Zhu et al., 2025) | Negative Sample Reward — you apply to VLM first |
| Wang et al., 2025 | 1-shot RLVR — justifies 1K sample efficiency |
| Koksal & Alatan, 2025 | Satellite RLVR — few-shot VLM RLVR precedent |
| Wen et al., 2025 (Incentivises) | CoT Pass@K theory |
| Masry et al., 2022 (ChartQA) | Eval benchmark |

---

## Outstanding / Future Work
- Scale to full 34K training set
- Increase K=4 → K=8 rollouts (D_reason signal too noisy at K=4)
- Explain/address HCPC OOD format drop (66.4%)
- No statistical significance tests in paper yet (CIs exist in detailed_analysis.md)
- Only 3B model tested — unclear if patterns hold at 7B/72B
