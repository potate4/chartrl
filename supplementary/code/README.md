# Code Release

Code release accompanying the paper. This directory contains everything
needed to reproduce the main experimental results and the analyses
reported in the paper.

## Directory layout

```
code/
  configs/         # Training configs (TrainingConfig, RewardConfig) and
                   # the 4 experiment recipes used in Table 1
  rewards/         # Chart-RVR base rewards, HCPC cross-rollout bonus,
                   # and the reward aggregator
  trainers/        # GRPO trainer wrapper + NSR-masked trainer
  utils/           # Output parsing, MiniLM similarity, logging
  data/            # Dataset loading, image preprocessing, prompt template
  evaluation/      # Accuracy + diversity metrics + evaluator
  scripts/         # train.py, eval_run.py
  analysis/        # Scripts that reproduce every number in the paper
  requirements.txt
```

## Paper-to-code map

| Paper claim | Code |
|---|---|
| HCPC reward $B_{\text{HCPC}}$ (§3) | `rewards/hcpc_reward.py` |
| Filter $G^+$: $\hat y_i = y^* \wedge \mathrm{sim}(\hat T_i, T^*) \ge \tau$ | `HCPCComputer._filter_correct` |
| $\tau = 0.8$, $w_{\text{table}}=2.0$, $w_{\text{reason}}=1.5$ | `HCPCComputer.__init__` defaults |
| NSR advantage rule: zero advantage when $R_i \ge 0.5 R_{\max}$ | `trainers/nsr_masked_trainer.py` |
| Reward backbone: format / type / table / process / accuracy / length / token-count | `rewards/base_rewards.py` |
| Chart-RVR-style reward aggregation | `rewards/reward_aggregator.py` |
| Pass@K (unbiased estimator) | `analysis/paper_stats.py: pass_at_k_unbiased` |
| $\Delta_{\text{tbl}}$ definition | `analysis/paper_stats.py` and `analysis/intersection_lift.py` |
| Per-tag emission rate | `analysis/verify_dissociation.py` |
| HCPC firing-rate analysis (§3, Appendix) | `analysis/hcpc_firing.py` |
| Pass@K curves (Figures 1 and 2) | `analysis/plot_passk.py` |

## Configurations (matching Table 1)

`configs/experiment.py` defines four training recipes that share every
hyperparameter listed in the Setup section and differ only by advantage
estimator and whether HCPC is added:

- `grpo_baseline`     : GRPO + Chart-RVR rewards
- `grpo_hcpc`         : GRPO + Chart-RVR + HCPC
- `nsr_baseline`      : NSR  + Chart-RVR rewards
- `nsr_hcpc`          : NSR  + Chart-RVR + HCPC

The Base row in the table corresponds to evaluation of
`Qwen/Qwen2.5-VL-3B-Instruct` without any training.

## Training

```bash
# Run any of the four experiments
python scripts/train.py --experiment grpo_baseline
python scripts/train.py --experiment grpo_hcpc
python scripts/train.py --experiment nsr_baseline
python scripts/train.py --experiment nsr_hcpc
```

All shared hyperparameters (1K-sample subset, K=4 rollouts, learning rate
1e-5, no KL penalty, LoRA r=8 alpha=16 on Q and V, 2 epochs ~ 2000 steps,
training temperature 1.0, bf16) are in `_SHARED` in
`configs/experiment.py`.

## Evaluation

```bash
# Evaluate a checkpoint on ChartQA (in-distribution) or ChartFC (OOD)
python scripts/eval_run.py \
    --checkpoint outputs/<run>/trl_output/checkpoint-2000 \
    --dataset chartqa \
    --subset 500 \
    --num_generations 4
```

This writes `per_sample.jsonl` (per-sample raw outputs, correctness,
diversity metrics, format compliance) and `summary.json`. The paper's
Table 1 is built from `summary.json` plus the analysis scripts below.

## Reproducing every number in the paper

All scripts read from `final_outputs/<run>/per_sample.jsonl` and emit the
exact numbers reported.

```bash
# Pass@1, Pass@4, Wilson CIs, paired-bootstrap CIs, McNemar tests,
# per-answer-type breakdown, per-tag emission rates, and
# P(correct | table parsed) vs P(correct | no table) (Delta_tbl)
python analysis/paper_stats.py

# Intersection-set Delta_tbl analysis (Appendix C)
python analysis/intersection_lift.py

# HCPC firing-rate distribution
python analysis/hcpc_firing.py

# Per-rollout format/accuracy decoupling (the central claim of Section 4.4)
python analysis/verify_dissociation.py

# Pass@K curves (Figures 1 and 2)
python analysis/plot_passk.py
```

## Notes on alignment with the paper

This release was reconciled against the paper's main text. A few points
worth flagging:

- The HCPC formulation here uses two terms (`C_table`, `D_reason`). An
  earlier prototype also included a chart-type consistency term
  `C_type`; this was removed because the chart-type-correctness filter
  in $G^+$ already collapses the term to a near-constant value. The
  chart-type *surrogate-task* reward from Chart-RVR is preserved in
  `rewards/base_rewards.py:chart_type_reward` and contributes to
  $R_{\text{base}}$.
- NSR's correctness threshold uses `reward_threshold * max_reward` with
  `reward_threshold = 0.5`, matching the value reported in the paper.
- No KL penalty is applied during training (`kl_coef=None`,
  `beta=0.0`), matching the paper.
- HCPC's table-similarity threshold $\tau$ defaults to $0.8$.

## Citation

(Anonymized for review.)
