# HCPC-RLVR Project Memory

## Project Overview
- Thesis: RLVR for chart QA with two contributions: (1) HCPC reward, (2) GRPO vs NSR comparison on VLMs
- Base model: Qwen2.5-VL-3B-Instruct with LoRA (r=8, alpha=16, q_proj/v_proj)
- Training: 1K samples from Chart-RVR CoT, 2000 steps, GRPO with 4 rollouts

## Key Data Locations
- Per-sample eval files (all 500 lines, same schema): See `memory/data_paths.md`
- Summary JSONs are in the same directories as per_sample.jsonl
- Analysis script: `app/scripts/analyze_results.py`
- Paper draft: `methodology_drafts/experimental_results_draft.md`
- Detailed analysis: `methodology_drafts/detailed_analysis.md`

## Per-Sample Schema
Fields: idx, question, label, chart_type, predictions[], raw_outputs[], correct[] (per-rollout relaxed match bools), reward_accuracy, relaxed_accuracy (sample-level), running_accuracy, parsed_first{type, answer, parse_success, table_parse_success_strict}, format_compliance{has_think_tags, has_answer_tags, has_type_tags, has_table_tags, proper_order, single_tags, fully_compliant}, diversity{c_table, d_reason, coherence, correct_rate}, time_seconds

## Critical Metric Definitions
- **Acc%** = `relaxed_accuracy` rate (NOT `any(correct)`, NOT `reward_accuracy`)
- **Pass@1** = mean(correct_rate) = mean(sum(correct)/len(correct) per sample)
- **Pass@4** = any(correct) rate (at least 1 of 4 rollouts correct)
- **correct[]** field = per-rollout relaxed match (True/False)
- Pass@K unbiased estimator: `1 - C(n-c,k)/C(n,k)` where n=total rollouts, c=num correct
- BUG FIX: Don't use `c >= k` as shortcut for score=1.0; correct condition is `n-c < k`

## Key Results (Verified)
- Base: Acc=51.8%, Pass@1=52.45%, Pass@4=77.4%
- GRPO: Acc=63.0%, HCPC: 62.2%, NSR: 63.4%
- HCPC best Pass@4: 82.80% (vs GRPO 80.31%)
- NSR highest C_table: 0.779
- HCPC OOD coherence: 0.433 vs GRPO 0.312 (+39%)
- HCPC OOD format drop: 66.4% vs GRPO 98.8%
- All differences not statistically significant (bootstrap p>0.05)

## Lessons Learned
- Pass@K estimator: the condition for guaranteed pass is `n-c < k` (not enough failures), NOT `c >= k`
- `relaxed_accuracy` and `correct[]` use relaxed matching; `reward_accuracy` uses a different scale
- Windows paths with spaces work in bash if properly quoted
- Bootstrap CIs with 10K resamples take ~30s for 6 methods x 500 samples
