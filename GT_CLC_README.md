# GT-CLC: Ground-Truth Anchored Cross-Level Coherence Reward

## What it is

GT-CLC is a reward signal that checks whether the model's reasoning chain explicitly
cites the key numerical values from the ground-truth answer. It was introduced to
replace `process_reward` (which measured similarity to GT reasoning steps — unavailable
in the sanchit97/chart-rvr-grpo-train dataset) without conflicting with HCPC's
diversity objective.

The core idea: a model that arrives at the correct answer through fabricated
intermediate values should not be rewarded the same as one that reads the correct
values from the extracted table and derives the answer from them.

---

## How it works

### Target construction

For each training sample, a set of weighted targets is built from the ground truth:

```
gt_table_values  = numeric values extracted from the GT table
gt_key_values    = values in GT reasoning that also appear in GT table
                   (empty in sanchit97 — no GT reasoning field exists)
targets          = {gt_key_value: value_weight, ..., gt_answer_num: answer_weight}
```

In practice (sanchit97 dataset) there is no GT reasoning, so targets always reduces to:

```
targets = {gt_answer_num: 2.0}   # answer_only mode
```

For categorical/text answers (e.g. "blue", "Yes"), `gt_answer_num` is None and the
function returns a neutral reward of 0.5 (contributes zero GRPO advantage).

### Recall computation

The model's structured reasoning steps are extracted and numbers are parsed from them.
Reward = `w_gt_clc * (earned_weight / total_weight)`.

In answer_only mode this is binary: 1.0 if the GT answer value appears in the model's
reasoning, 0.0 if it does not.

### Reward range

```
gt_clc = 0.0    model reasoning has no numeric values, or GT answer not cited
gt_clc = 0.5    no numeric targets (text/categorical answer)  [GRPO-neutral]
gt_clc = 1.0    GT answer value found in model reasoning
```

---

## GRPO integration

GT-CLC is per-rollout, not per-group. Each of the K=4 rollouts gets its own
`gt_clc` value. GRPO advantage = `(r_i - mean(rewards)) / std(rewards)`, so
GT-CLC contributes to gradient only when there is variance across the group
(i.e. some rollouts cite the answer and some do not). Once the model reliably
cites the answer for a question type, gt_clc variance collapses and the signal
contribution drops to zero — which is expected behaviour.

---

## Experiment: grpo_gt_clc

Active reward components:

| Component       | Max   | Notes                                              |
|-----------------|-------|----------------------------------------------------|
| base_accuracy   | 1.0   | Soft numeric match via exp(-10*rel_error)          |
| base_format     | 1.4   | Full 2.0 requires perfect one-shot regex           |
| base_length     | ~2.5  | Step count (0.25/step, cap 1.5) + char length      |
| base_table      | 1.75  | JSON parse + column/row match                      |
| gt_clc          | 1.0   | w_gt_clc=1.0, answer_weight=2.0                    |
| base_chart_type | 0     | Disabled — GT field empty in sanchit97             |
| base_token_count| 0     | Disabled — Qwen produces two think blocks          |
| base_process    | 0     | Disabled — replaced by GT-CLC                     |
| hcpc            | 0     | Disabled for this experiment (use_hcpc=False)      |

Config name: `grpo_gt_clc` in `app/configs/experiment.py`.

---

## Files changed

### `app/rewards/clc_reward.py`
- Added `import re`
- `GTCLCComputer.compute()`:
  - Fix A: extracts the LAST `<think>` block instead of using `parse_response`
    (which returned the first block — Qwen's native chain-of-thought — rather
    than the structured `<step-N>` block)
  - Fix B: for percentage GT labels (e.g. "14.2%"), also matches the
    non-divided form (14.2) in model reasoning to avoid false negatives when the
    model omits the % sign in intermediate steps
  - Step-tag contamination fix: strips `<step-N>:` tags before extracting
    numbers so tag indices (1, 2, 3...) don't trivially match small answers
  - Percentage alignment fix: `try_parse_numeric("14.2%")` returns 14.2 but
    `extract_numbers("14.2%")` returns {0.142}; divides `gt_answer_num` by 100
    when label ends with `%` to align both representations
  - Added `details["mode"]` field: "full" when GT reasoning+table values are
    available, "answer_only" when only GT label acts as target

### `app/configs/experiment.py`
- `_make_reward_config`: added `use_gt_clc: bool = True` parameter; disabled
  `use_chart_type_reward` and `use_token_count_reward` (both always fire at 0)
- Added `grpo_true_baseline` experiment (process_reward=True, no GT-CLC) for
  reproducing published Chart-RVR numbers
- Added `grpo_gt_clc` experiment (same config as grpo_baseline, separate output dir)
- Added `max_completion_length=1280` to `_SHARED` (up from base default of 768)
  to prevent truncation on complex charts with large JSON tables

### `app/utils/parsing.py`
- Added `_WORD_TO_NUM` lookup table and word-number check at the top of
  `try_parse_numeric` so "Three" → 3.0, "Four" → 4.0, etc.
  Fixes evaluation metric bug where correct word-form answers were marked wrong

### `app/data/prompts.py`
- Added `### Rules` block before `### Output format` in `SYSTEM_PROMPT`:
  - Always write numbers as digits
  - For yes/no questions, answer with exactly "Yes" or "No"
  - For single-point questions, give one value not a range

### `app/trainers/base_trainer.py`
- Added `gt_clc` to `avg_total` sum in metrics summary (was missing, causing
  understated total reward in logs)
- Added `avg_gt_clc` and `avg_gt_clc_recall` keys to summary dict

### `app/rewards/hcpc_reward.py`
- Removed unused `compute_similarity` import

### `app/scripts/train_grpo_gt_clc.py` (new file)
- Dedicated training script for the grpo_gt_clc experiment
- CLI flags: `--resume`, `--resume-from`, `--from-scratch`, `--subset-size`,
  `--num-epochs`, `--batch-size`, `--learning-rate`, `--num-generations`,
  `--no-wandb`, `--output-dir`, `--cache-dir`
- Logs each active reward component at startup

---

## How to run

All commands assume your working directory is the project root (`chart/`).

---

### 1. Sanity check — verify everything starts without errors (~5 min)

```bash
python app/scripts/train_grpo_gt_clc.py --subset-size 50 --no-wandb
```

What to verify in the output before committing to a full run:
- No import errors on startup
- Reward summary block prints `gt_clc  enabled`
- First reward log shows `gt_clc` values between 0.0 and 1.0 (not stuck at 0.5)
- `base_accuracy` has variance across the 4 rollouts (not all 0.0 or all identical)

---

### 2. Full training run

```bash
python app/scripts/train_grpo_gt_clc.py
```

Checkpoints are saved every 10 steps to:
```
./outputs/grpo_gt_clc/run_<YYYYMMDD_HHMMSS>/checkpoints/step_<N>/
```
The 3 most recent checkpoints are kept (`keep_last_n=3`). Metrics are written
to `./outputs/grpo_gt_clc/run_<timestamp>/metrics.jsonl`.

---

### 3. Full training run with wandb disabled

```bash
python app/scripts/train_grpo_gt_clc.py --no-wandb
```

---

### 4. Resume from the latest checkpoint of the latest run

```bash
python app/scripts/train_grpo_gt_clc.py --resume
```

The script finds the most recent run directory under `./outputs/grpo_gt_clc/`
and resumes from its latest checkpoint automatically.

---

### 5. Resume from a specific checkpoint path

```bash
python app/scripts/train_grpo_gt_clc.py --resume-from outputs/grpo_gt_clc/run_20250522_103045/checkpoints/step_100
```

Replace the path with the actual checkpoint directory you want to continue from.

---

### 6. Start a completely fresh run (ignores existing checkpoints)

```bash
python app/scripts/train_grpo_gt_clc.py --from-scratch
```

Old checkpoints are not deleted — a new timestamped run directory is created.

---

### 7. Override training length

```bash
# Run for 1 epoch instead of 2
python app/scripts/train_grpo_gt_clc.py --num-epochs 1

# Use 8 rollouts per sample instead of 4
python app/scripts/train_grpo_gt_clc.py --num-generations 8
```

---

### 8. From a Kaggle / Colab notebook

```python
# Full run
!python app/scripts/train_grpo_gt_clc.py --no-wandb

# Sanity check
!python app/scripts/train_grpo_gt_clc.py --subset-size 100 --no-wandb

# Resume latest
!python app/scripts/train_grpo_gt_clc.py --resume --no-wandb

# Resume specific checkpoint
!python app/scripts/train_grpo_gt_clc.py \
    --resume-from outputs/grpo_gt_clc/run_xxx/checkpoints/step_100 \
    --no-wandb
```

---

### All CLI flags

| Flag | Default | Description |
|---|---|---|
| `--subset-size N` | full dataset | Limit training set to N samples |
| `--no-wandb` | wandb enabled | Disable WandB logging |
| `--resume` | off | Resume from latest checkpoint of latest run |
| `--resume-from PATH` | none | Resume from a specific checkpoint path |
| `--from-scratch` | off | Force a new run (old checkpoints kept) |
| `--num-epochs N` | 2 | Number of training epochs |
| `--num-generations N` | 4 | Rollouts per sample |
| `--batch-size N` | 1 | Per-device batch size |
| `--learning-rate F` | 1e-6 | Learning rate |
| `--output-dir PATH` | `./outputs` | Root directory for run outputs |
| `--cache-dir PATH` | `./cache` | HuggingFace model/dataset cache |

---

## Known limitations

**GT-CLC saturates as training progresses.**
Once the model consistently cites the answer value in its steps (the desired
behaviour), all 4 rollouts get gt_clc=1.0 for that question type and the signal
contributes zero GRPO advantage. This is expected — `base_accuracy` and
`base_table` carry the gradient after saturation.

**Binary signal for numeric answers.**
In answer_only mode (which is always active for sanchit97), gt_clc is 0 or 1.0.
Maximum training signal occurs when the model cites the answer ~50% of the time.

**Text/categorical answers are neutral.**
"Yes", "No", color names, category labels — these get gt_clc=0.5 regardless of
reasoning quality. GT-CLC provides no gradient for ~30% of ChartQA questions.

**5% tolerance can produce false positives on dense charts.**
If a chart has bars at 14.1 and 14.2 (GT answer), citing 14.1 passes the
tolerance check. Unavoidable without a stricter threshold that would also
penalise legitimate rounding differences.

**Off-by-one counting and visual color misreads are not fixed by this reward.**
"How many bars?" answered as 14 when correct is 13, or wrong legend color
identification, are visual perception errors that require better training data or
a larger model — not addressable via reward shaping.
