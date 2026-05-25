# HCPC-RLVR

Implementation of Hierarchical Correct-Path Consistency (HCPC), a
cross-rollout reward bonus for chart-reasoning RLVR, together with a
GRPO trainer and an NSR-masked trainer variant.

## Directory layout

```
code/
  configs/         # TrainingConfig, RewardConfig, and experiment recipes
  rewards/         # Base rewards, HCPC bonus, reward aggregator
  trainers/        # GRPO trainer and NSR-masked trainer
  utils/           # Output parsing, MiniLM similarity, logging
  data/            # Dataset loading, image preprocessing, prompt template
  evaluation/      # Accuracy + diversity metrics + evaluator
  scripts/         # train.py, eval_run.py
  analysis/        # Statistics and plotting scripts
  requirements.txt
```

## Configurations

`configs/experiment.py` defines four training recipes that share every
hyperparameter and differ only by advantage estimator and whether HCPC
is added:

- `grpo_baseline`     : GRPO + base rewards
- `grpo_hcpc`         : GRPO + base rewards + HCPC
- `nsr_baseline`      : NSR  + base rewards
- `nsr_hcpc`          : NSR  + base rewards + HCPC

The untuned baseline corresponds to evaluation of
`Qwen/Qwen2.5-VL-3B-Instruct` without any training.

## Training

```bash
python scripts/train.py --experiment grpo_baseline
python scripts/train.py --experiment grpo_hcpc
python scripts/train.py --experiment nsr_baseline
python scripts/train.py --experiment nsr_hcpc
```

Shared hyperparameters (1K-sample subset, K=4 rollouts, learning rate
1e-6, no KL penalty, LoRA r=8 alpha=16 on Q and V, 2 epochs,
training temperature 0.8, bf16) are in `_SHARED` in
`configs/experiment.py`.

## Evaluation

```bash
python scripts/eval_run.py \
    --checkpoint outputs/<run>/trl_output/checkpoint-2000 \
    --dataset chartqa \
    --subset 500 \
    --num_generations 4
```

Writes `per_sample.jsonl` (per-sample raw outputs, correctness,
diversity metrics, format compliance) and `summary.json`.

## Analysis

```bash
python analysis/compute_stats.py       # Pass@K, CIs, McNemar, Delta_tbl
python analysis/intersection_lift.py   # Intersection-set Delta_tbl
python analysis/hcpc_firing.py         # HCPC firing-rate distribution
python analysis/verify_dissociation.py # Format/accuracy decoupling
python analysis/plot_passk.py          # Pass@K curves
```
