# Data Paths Reference

## Per-Sample JSONL Files (all 500 lines, identical schema)

### In-Distribution (ChartQA)
- **Base**: `app/outputs/eval_results/main_base_chartqa_20260218_210622/per_sample.jsonl`
- **GRPO**: `app/outputs/eval_results/grpo_baseline_checkpoint-2000_chartqa_20260219_073124/per_sample.jsonl`
- **GRPO+HCPC**: `app/outputs/eval_results/grpo_hcpc_updated_checkpoint-2000_chartqa_20260301_170306/per_sample.jsonl`
- **NSR**: `app/outputs/rakhi_outputs_full (1)/kaggle/working/chartrl/app/outputs/eval_results/checkpoint-2000_chartqa_20260301_084802/per_sample.jsonl`

### Out-of-Distribution (ChartFC)
- **GRPO**: `app/outputs/chartfc_grpo_outputs_full (1)/kaggle/working/chartrl/app/outputs/eval_results/checkpoint-2000_chartfc_20260310_114351/per_sample.jsonl`
- **GRPO+HCPC**: `app/outputs/chartfc_grpo_hcpc_outputs_fullhcpc/kaggle/working/chartrl/app/outputs/eval_results/checkpoint-2000_chartfc_20260311_064248/per_sample.jsonl`

## Summary JSON Files
Same directories as above, file named `summary.json`.

## Scripts
- `app/scripts/analyze_results.py` — comprehensive per-sample analysis (12 sections)
- `app/scripts/evaluate.py` — evaluation runner
- `app/scripts/train.py` — training runner

## Outputs
- `methodology_drafts/detailed_analysis.md` — auto-generated analysis report
- `methodology_drafts/experimental_results_draft.md` — paper sections 4 & 5 draft
