# Supplementary Materials

Supplementary materials for the paper. Everything needed to reproduce the
main results and to inspect the per-sample model outputs is included.

## Contents

```
supplementary/
├── README.md                 # This file
├── code/                     # Full code release (training + eval + analysis)
├── eval_outputs/             # Per-sample model outputs for every Table 1 row
│   ├── base_chartqa/
│   ├── grpo_chartqa/
│   ├── grpo_chartfc/
│   ├── hcpc_chartqa/         # GRPO+HCPC
│   ├── hcpc_chartfc/
│   ├── nsr_baseline_chartqa/
│   ├── nsr_baseline_chartfc/
│   ├── nsr_hcpc_chartqa/
│   └── nsr_hcpc_chartfc/
├── training_logs/            # Training-time metrics + full TRL trainer state
│   ├── nsr_baseline/
│   └── nsr_hcpc/
├── analysis_outputs/         # Captured outputs from the analysis scripts
│   ├── paper_stats_output.txt
│   ├── intersection_lift_output.txt
│   └── hcpc_firing_output.txt
└── figures/                  # PDF figures used in the paper
    ├── fig_passk.pdf
    └── fig_passk_chartfc.pdf
```

## What's in each subfolder

### `code/`

The full source code release. See `code/README.md` for the paper-to-code
map. Highlights:

- `code/rewards/hcpc_reward.py` — the HCPC bonus exactly as reported in
  the paper (filter, table-consistency term, reasoning-diversity term,
  per-rollout application).
- `code/trainers/nsr_masked_trainer.py` — NSR's advantage-masking rule
  (zero advantage when raw reward $\ge$ `reward_threshold * R_max`,
  with `reward_threshold = 0.5`).
- `code/rewards/base_rewards.py` — the Chart-RVR backbone rewards
  (format, length, accuracy, chart-type, table, process-conformity).
- `code/configs/experiment.py` — the four experiment recipes
  (`grpo_baseline`, `grpo_hcpc`, `nsr_baseline`, `nsr_hcpc`).
- `code/analysis/` — every script that produces a number or figure in
  the paper.

### `eval_outputs/`

For each of the configurations evaluated in Table 1, three files:

- `per_sample.jsonl` — one JSON record per test sample with:
  - `idx`, `question`, `label`, `chart_type`
  - `predictions` — the 4 parsed answers (one per rollout)
  - `raw_outputs` — the 4 raw model outputs
  - `correct` — per-rollout binary correctness
  - `parsed_first` — structured parse of rollout 0 (type, answer,
    parse_success, table_parse_success_strict)
  - `format_compliance` — per-tag emission flags
  - `diversity` — c_table, d_reason, correct_rate
  - `time_seconds`
- `summary.json` — aggregate metrics (accuracy, Pass@K, c_table,
  d_reason, format_compliance_rate, timestamp)
- `config.json` — evaluation configuration (subset_size,
  num_generations, temperature, top_p, seed)

500 test samples per benchmark, 4 generations per sample (8 for one
GRPO ChartQA file; see `summary.json`). Every number in Table 1 can be
recomputed from these files using `code/analysis/paper_stats.py`.

### `training_logs/`

For both NSR baseline and NSR+HCPC training runs:

- `metrics.jsonl` — per-step training metrics (one line every 10
  optimization steps, ~200 lines per run): `avg_total`,
  `avg_base_total`, `avg_base_accuracy`, `avg_base_format`,
  `avg_base_table`, `avg_base_type`, `avg_hcpc`.
- `trainer_state.json` — TRL's full per-step log_history (200+
  entries), with `reward`, `reward_std`, `entropy`, `grad_norm`,
  `loss`, completions length statistics, clipping ratios, and the
  exact per-step group mean used to compute GRPO advantages.
- `config.json` — the full training config used for this run
  (hyperparameters exactly as in the paper).

### `analysis_outputs/`

Captured stdout from the analysis scripts so reviewers can read the
results without re-running.

- `paper_stats_output.txt` — Wilson 95% CIs for Pass@1 / Pass@4, full
  Pass@K curves, paired-bootstrap and McNemar tests for every method
  pair, per-answer-type breakdown, per-tag emission rates,
  conditional answer-correctness given table parsing.
- `intersection_lift_output.txt` — $\Delta_{\text{tbl}}$ computed on
  the common-table subset (Appendix C of the paper).
- `hcpc_firing_output.txt` — distribution of $|G^+|$ (HCPC firing
  rate) per method.

### `figures/`

PDFs of the Pass@K curves shown in the paper (Figure 1 = ChartQA,
Figure 2 = ChartFC).

## Reproducing every number in the paper

After unzipping, from inside `supplementary/`:

```bash
# Pass@1 / Pass@4 with Wilson CIs + paired bootstrap + McNemar tests
python code/analysis/paper_stats.py

# Intersection-set Delta_tbl analysis (Appendix)
python code/analysis/intersection_lift.py

# HCPC firing-rate distribution
python code/analysis/hcpc_firing.py

# Per-rollout format/accuracy decoupling check
python code/analysis/verify_dissociation.py

# Pass@K figures
python code/analysis/plot_passk.py
```

All scripts read from `eval_outputs/` and require only `numpy` and
`matplotlib` (figures only). Each produces output matching what is
already captured in `analysis_outputs/`.

## Training reproducibility

Training was done with the configs in
`code/configs/experiment.py`. Each row of Table 1 corresponds to one
of the four named experiments:

| Table 1 row | Experiment name |
|---|---|
| GRPO | `grpo_baseline` |
| GRPO+HCPC | `grpo_hcpc` |
| NSR | `nsr_baseline` |
| NSR+HCPC | `nsr_hcpc` |

From inside `code/`:

```bash
python scripts/train.py --experiment grpo_baseline
python scripts/train.py --experiment grpo_hcpc
python scripts/train.py --experiment nsr_baseline
python scripts/train.py --experiment nsr_hcpc
```

Hyperparameters in `code/configs/experiment.py:_SHARED` match the
Setup section of the paper: Qwen2.5-VL-3B + LoRA r=8 alpha=16 on
$W_q, W_v$; 1K-sample subset of the Chart-RVR CoT mixture; K=4
rollouts; learning rate 1e-5; no KL penalty; bf16; ~2000 steps.

Evaluation:

```bash
python scripts/eval_run.py \
    --checkpoint <path-to-checkpoint> \
    --dataset chartqa --subset 500 --num_generations 4
```

## Notes on the code release

- The HCPC formulation in `code/rewards/hcpc_reward.py` uses two terms
  ($C_{\text{table}}$ and $D_{\text{reason}}$). An earlier prototype
  also included a chart-type consistency term; this was removed
  because it sits at ceiling for almost every group where HCPC fires
  (the correct-path filter already requires answer agreement on
  charts with relatively easy chart-type identification). The
  chart-type surrogate-task reward from Chart-RVR is preserved in
  `code/rewards/base_rewards.py:chart_type_reward` and contributes to
  $R_{\text{base}}$.
- NSR is implemented as advantage masking on the TRL trainer
  (`code/trainers/nsr_masked_trainer.py`): TRL computes the GRPO
  advantage first, then the mask zeros the advantage of rollouts whose
  raw reward $\ge 0.5\,R_{\max}$.
- No KL penalty is applied (`kl_coef=None`, `beta=0.0` in the run
  configs).
- HCPC's table-similarity threshold $\tau = 0.8$.

## Anonymization

This package has been anonymized for review: author-identifying paths,
usernames, and host names have been redacted. Training checkpoints are
omitted from the supplementary (release post-acceptance) but every
per-sample evaluation output, training log, and analysis script needed
to verify the paper's claims is included.
