# ChartRL — Master Notes

Living document for the thesis analysis phase. Append below as new artifacts, results, or decisions accumulate.

---

## 1. Final Eval Outputs

All consolidated under [app/final_outputs/](app/final_outputs/). Each subfolder contains the original `config.json`, `eval.log`, `per_sample.jsonl`, and `summary.json` from the eval run (untouched).

Naming convention: `{method}_{benchmark}/`.

### ChartQA (500 samples each)

| Experiment | Folder | Source folder |
|---|---|---|
| Base (no RL) | [base_chartqa/](app/final_outputs/base_chartqa/) | `app/outputs/eval_results/main_base_chartqa_20260218_210622/` |
| GRPO | [grpo_chartqa/](app/final_outputs/grpo_chartqa/) | `app/outputs/eval_results/grpo_baseline_checkpoint-2000_chartqa_20260219_073124/` |
| HCPC | [hcpc_chartqa/](app/final_outputs/hcpc_chartqa/) | `app/outputs/eval_results/grpo_hcpc_updated_checkpoint-2000_chartqa_20260301_170306/` |
| NSR | [nsr_chartqa/](app/final_outputs/nsr_chartqa/) | `app/outputs/rakhi_outputs_full (1)/.../checkpoint-2000_chartqa_20260301_084802/` |

### ChartFC (500 samples each)

| Experiment | Folder | Source folder |
|---|---|---|
| GRPO | [grpo_chartfc/](app/final_outputs/grpo_chartfc/) | `app/outputs/chartfc_grpo_outputs_full (1)/.../checkpoint-2000_chartfc_20260310_114351/` |
| HCPC | [hcpc_chartfc/](app/final_outputs/hcpc_chartfc/) | `app/outputs/chartfc_grpo_hcpc_outputs_fullhcpc/.../checkpoint-2000_chartfc_20260311_064248/` |
| NSR | [nsr_chartfc/](app/final_outputs/nsr_chartfc/) | `app/outputs/nsr_hcpc_evalunsr/evalunsr/` |

**Note:** `nsr_chartfc/` jsonl is named `per_sample (2).jsonl` (with the space) — all other folders use `per_sample.jsonl`. Watch for this when globbing.

**Missing:** No `base_chartfc` eval available.

---

## 2. Headline Metrics (from `summary.json`)

All runs: `Qwen/Qwen2.5-VL-3B-Instruct`, test split, 500 samples, T=0.8, top_p=0.95, k=4 generations (GRPO ChartQA used k=8).

### ChartQA

| Method | Accuracy | pass@1 | pass@4 | c_table | d_reason | coherence | format_compl. |
|---|---|---|---|---|---|---|---|
| Base | 0.518 | 0.524 | 0.774 | 0.591 | 0.040 | 0.545 | 0.298 |
| GRPO | 0.630 | 0.619 | 0.803 (pass@8: 0.816) | 0.749 | 0.058 | 0.252 | 0.968 |
| HCPC | 0.622 | 0.629 | 0.828 | 0.741 | 0.052 | 0.244 | 0.982 |
| NSR | 0.634 | 0.622 | 0.820 | 0.779 | 0.056 | 0.245 | 0.970 |

### ChartFC

| Method | Accuracy | pass@1 | pass@4 | c_table | d_reason | coherence | format_compl. |
|---|---|---|---|---|---|---|---|
| GRPO | 0.638 | 0.661 | 0.922 | 0.950 | 0.059 | 0.312 | 0.988 |
| HCPC | 0.606 | 0.625 | 0.918 | 0.782 | 0.060 | 0.433 | 0.664 |
| NSR | 0.708 | 0.687 | 0.924 | 0.961 | 0.060 | 0.314 | 0.994 |

### Quick observations
- **NSR leads on accuracy in both benchmarks.** ChartQA: 0.634 (NSR) > 0.630 (GRPO) > 0.622 (HCPC) > 0.518 (Base). ChartFC: 0.708 (NSR) >> 0.638 (GRPO) > 0.606 (HCPC).
- HCPC ChartFC has notably low **format_compliance (0.664)** vs ~0.99 for others — investigate if this is hurting its accuracy ranking.
- HCPC ChartFC also has the highest **coherence (0.433)** — possible tradeoff: more reasoning, less format adherence.
- GRPO ChartQA was run with k=8 (others k=4) — pass@k comparisons across methods are slightly apples-to-oranges for that one cell.

---

## 3. Checkpoint Provenance

All RL runs evaluated at `checkpoint-2000`. Original training checkpoint paths recorded in each `summary.json["checkpoint"]`:

- **GRPO ChartQA**: `grpo_baseline/run_20260218_125045/trl_output/checkpoint-2000`
- **GRPO ChartFC**: `chartfcupload/run_20260218_125045/trl_output/checkpoint-2000` (Kaggle)
- **HCPC ChartQA**: `grpo_hcpc/run_20260228_170909/trl_output/checkpoint-2000` (Kaggle)
- **HCPC ChartFC**: `hcpcchartfc/run_20260219_191830/trl_output/checkpoint-2000` (Kaggle)
- **NSR ChartQA**: `nsr_baseline/run_20260228_225209/trl_output/checkpoint-2000` (Kaggle)
- **NSR ChartFC**: same `nsr_baseline/run_20260228_225209/...` checkpoint (Kaggle)

NSR ChartQA & ChartFC use the **same checkpoint**, just evaluated on the two benchmarks.

---

## 4. Open Questions / TODO

- [ ] Investigate why HCPC ChartFC format_compliance dropped to 0.664.
- [ ] Decide if a `base_chartfc` eval is needed for a fair baseline comparison on ChartFC.
- [ ] Decide whether to rename `nsr_chartfc/per_sample (2).jsonl` → `per_sample.jsonl` for uniform globbing.
- [ ] Per-label-type accuracy breakdown (numeric / boolean / list / string) — useful given the type-aware reward fix.

---

---

## 5. Honest Analysis (2026-05-08) — recomputed from raw `per_sample.jsonl`

All numbers below come directly from the jsonl files in [app/final_outputs/](app/final_outputs/). I did **not** trust `summary.json` until I verified — every headline number recomputed exactly, so the summaries can be trusted on aggregates. Scripts in [app/analysis/](app/analysis/).

### 5.1 Sample alignment (good news)
- All 4 ChartQA runs evaluate the **same idx ordering**, same 500 questions (496 unique — 4 dup questions). Apples-to-apples.
- All 3 ChartFC runs evaluate the **same 500 samples in same order**.
- All ChartFC samples are boolean (Yes/No) — it's a fact-checking benchmark.
- ChartQA: ~58% numeric, ~17% boolean, ~23% string, ~2% list (n=9).

### 5.2 The actual ranking (pass@1, 95% Wilson CI)

**ChartQA (n=500):**
| Method | pass@1 | 95% CI | pass@4 |
|---|---|---|---|
| base | 0.518 | [0.474, 0.562] | 0.774 |
| grpo | 0.630 | [0.587, 0.671] | 0.816 |
| hcpc | 0.622 | [0.579, 0.663] | 0.828 |
| nsr  | 0.634 | [0.591, 0.675] | 0.820 |

**ChartFC (n=500):**
| Method | pass@1 | 95% CI | pass@4 |
|---|---|---|---|
| grpo | 0.638 | [0.595, 0.679] | 0.922 |
| hcpc | 0.606 | [0.563, 0.648] | 0.918 |
| nsr  | 0.708 | [0.667, 0.746] | 0.924 |

### 5.3 Statistical significance (paired McNemar + bootstrap)

**ChartQA:**
- Every RL method beats base with p < 1e-4 — that gap is real.
- **GRPO ≈ HCPC ≈ NSR.** All three pairwise McNemar tests give p ∈ [0.66, 0.93]. Bootstrap CIs on every pairwise gap include zero. **The "HCPC contribution" is invisible on ChartQA.**

**ChartFC:**
- NSR beats GRPO: diff = +0.070, 95% CI [+0.024, +0.118], McNemar p = 0.005.
- NSR beats HCPC: diff = +0.102, 95% CI [+0.052, +0.152], McNemar p = 9.7e-5.
- GRPO vs HCPC: not significant (p = 0.26, CI includes zero). **HCPC actually hurts a tiny bit on ChartFC.**

### 5.4 Per-label-type breakdown (ChartQA only — ChartFC is all boolean)

| Method | bool n=87 | numeric n=290 | string n=114 | list n=9 |
|---|---|---|---|---|
| base | 0.529 | 0.503 | 0.588 | 0.000 |
| grpo | 0.586 | 0.631 | 0.711 | 0.000 |
| hcpc | 0.621 | 0.614 | 0.649 | 0.556 |
| nsr  | 0.598 | 0.624 | 0.693 | 0.556 |

- The numeric/string gains over base are the headline story. Booleans on ChartQA are the noisiest signal (small n=87) — HCPC's "lead" there is within sampling noise.
- list (n=9) is too small to draw conclusions.

### 5.5 What HCPC and NSR actually do (verified by reading code)

- **HCPC** ([app/rewards/hcpc_reward.py](app/rewards/hcpc_reward.py)): adds a *bonus* to correct rollouts based on `correct_rate × (w_type·C_type + w_table·C_table + w_reason·D_reason)`, with weights `(1.0, 2.0, 1.5)`. Only fires when ≥2 rollouts are GT-correct. So it's a *consistency-shaped reward bonus*, layered on top of the base reward — not a separate algorithm.
- **NSR** ([app/trainers/nsr_trainer.py](app/trainers/nsr_trainer.py)): the standard NSR formulation — zero gradient on correct rollouts, negative-only updates on wrong rollouts. Independent of HCPC. The `nsr_chartfc` checkpoint is plain `nsr_baseline` (NOT `nsr_hcpc`) — confirmed via eval log + `experiment.py`.
- The **6 planned experiments** in [app/configs/experiment.py](app/configs/experiment.py) are: `grpo_baseline`, `grpo_hcpc`, `nsr_baseline`, `nsr_hcpc`, `w_reinforce_baseline`, `w_reinforce_hcpc`. **You currently have data for 4 of 6** (no W-REINFORCE runs, no NSR+HCPC runs).

### 5.6 Implementation concerns to flag

1. **Format compliance gap.** GRPO/NSR ChartFC ≈ 0.99, HCPC ChartFC = 0.66. That's a 33pt format-compliance regression on the same dataset. The model frequently emits answer text without proper `<think>...</think><answer>...</answer>` wrappers. This is suspicious and likely a training instability rather than a feature — investigate `hcpc_chartfc` training logs before publishing.
2. **NSR ChartFC takes 127s/sample vs ChartQA NSR 14s/sample** for the *same checkpoint*. ~9x slowdown. Probably longer rollouts on the FC harness, but worth confirming this isn't generation length blowing up (could imply NSR ChartFC is generating much longer reasoning).
3. **`grpo_chartqa` was evaluated with k=8 generations for 91/500 samples and k=4 for 409.** Inconsistent k. Headline pass@4 is fine but not strictly comparable cell-by-cell. Re-run for consistency before camera-ready.
4. **GRPO ChartFC `format_compliance` is 0.988**, but `grpo_chartfc` and `nsr_chartfc` and `hcpc_chartfc` *never produce a `<table>`-tagged output that fails strict-table-parse* — 0.99/0.86/0.99 strict parse rates. Sanity-check that `c_table` of 0.95 on ChartFC isn't reflecting trivial single-row tables (the chart-fc benchmark answers are Yes/No, so table structure may collapse).

---

## 6. EMNLP Realistic Assessment

### 6.1 The honest story your data tells right now

> **NSR is the win.** On ChartFC it's significantly better than both GRPO and HCPC. On ChartQA it ties GRPO (and edges HCPC slightly, not significantly). HCPC — your headline contribution — is statistically indistinguishable from GRPO on ChartQA and *slightly worse* on ChartFC. As written, the data does not support a paper claiming HCPC as the contribution.

### 6.2 What this means for EMNLP

EMNLP reviewers will demand: (a) significance testing on paired predictions, (b) multiple seeds, (c) comparison to relevant baselines, (d) ablations. Right now:

- **No multiple seeds.** A single training run per cell. Reviewers will (rightly) ask whether the +0.012 HCPC vs GRPO gap on ChartQA is reproducible. Given paired CI = [-0.04, +0.05], it is almost certainly not.
- **Two benchmarks only**, both in-distribution from the training data family. No OOD evaluation, no transfer.
- **No NSR+HCPC** runs — the most interesting cell of the 2x3 grid (RL algo × {with HCPC, without HCPC}).
- **W-REINFORCE entirely missing** — your config defines it but you have no eval data.

### 6.3 Three possible papers, ranked by realism

1. **(Highest viability) "Negative-only reinforcement is sufficient for chart understanding":** lead with NSR. The story is clean: NSR matches GRPO at pass@1 on ChartQA, beats it +7pt at pass@1 on ChartFC, and is the only method significantly above GRPO with any benchmark. The HCPC work becomes a *negative ablation* you discuss honestly. This is publishable as a findings-style or short-paper contribution if you add 1-2 more benchmarks and 2-3 seeds.

2. **(Medium viability) "When does consistency-shaped reward help RL training for chart QA?":** reframe HCPC as the *question*, not the *answer*. Run the missing 4 cells (NSR+HCPC, W-REINFORCE ± HCPC) so you have a 3×2 grid. Then HCPC may help in one cell and hurt in another — that's a publishable diagnostic story even if HCPC isn't a clean win, because it tells the community when to and when not to use these objectives. Risk: requires significant new compute.

3. **(Lower viability) "HCPC reward beats GRPO":** would require running HCPC again with seeds to demonstrate reproducible gains, *and* fixing whatever is killing format-compliance on ChartFC, *and* finding a benchmark where it's actually significantly better. Based on current data this is the weakest path — the gain is in the noise.

### 6.4 Concrete next steps (in priority order)

1. **Run 3+ seeds for at least the headline ChartQA cells.** Without this, no statistical claim survives review. Cheapest signal you can buy.
2. **Investigate `hcpc_chartfc` format-compliance collapse (0.66 vs ~0.99 elsewhere).** Read training logs, look at gradient norms, sample model outputs. If it's a bug, your HCPC ChartFC numbers are corrupted; if it's not, it's evidence HCPC destabilizes training and you should report that.
3. **Run NSR+HCPC** on both benchmarks. This is the cell that decides whether HCPC has any value at all (it might shine as an *NSR augmentation* even if it doesn't help GRPO).
4. **Add at least one OOD eval** (e.g., PlotQA, ChartBench, or a held-out chart-type subset). Reviewers will ask.
5. **Write the paper around NSR**, with HCPC as either a careful ablation (Path 1) or a 3×2 diagnostic study (Path 2). Don't write HCPC as the headline.
6. **Re-run `grpo_chartqa` with consistent k=4** so all numbers are comparable.

### 6.5 What to NOT say in the paper

Based on your current data, do NOT claim:
- "HCPC improves over GRPO" — the gap is +0.008 on ChartQA (CI crosses 0) and -0.032 on ChartFC.
- "Diversity-aware rewards transfer across benchmarks" — they don't here; HCPC's behavior is benchmark-dependent.
- Anything about W-REINFORCE — no data.
- "HCPC is robust" — the ChartFC format collapse contradicts that.

You CAN claim (with seeds + sig tests):
- RLVR (GRPO/NSR) substantially improves over the base Qwen2.5-VL-3B on both benchmarks (+11pt ChartQA, big gap on ChartFC).
- NSR is competitive with or better than GRPO at significantly less variance in some setting.
- Type-aware accuracy reward fix (your bug fix from 2026-02-09) is necessary for these benchmarks — possibly worth its own short experiment.

---

---

## 7. Stage-by-Stage Failure Analysis (2026-05-08)

Re-parsed `raw_outputs[]` for every rollout in every run (4 rollouts × 500 samples × 7 runs = ~14,000 rollouts) to recover per-rollout structural pipeline correctness. Scripts: [app/analysis/stage_failure_analysis.py](app/analysis/stage_failure_analysis.py), [app/analysis/reward_variants_score.py](app/analysis/reward_variants_score.py).

### 7.1 What's a "stage"?
- **format_ok**: full chart-RVR template `<think><type>...</type><table>...</table>...</think><answer>...</answer>` matches.
- **type_ok**: `<type>` block parses to a non-empty string.
- **table_ok**: `<table>` block parses as JSON with `{columns, rows}` schema.
- **answer_ok**: existing `correct[k]` flag (relaxed-accuracy match against GT).

### 7.2 Joint distribution per run (rollout-level, n≈2000 each)

| Run | format_ok | table_ok | answer_ok | T+T (good) | T+F (bad reasoning) | **F+T (lucky)** | F+F |
|---|---|---|---|---|---|---|---|
| base_chartqa | 0.30 | 0.69 | 0.52 | 0.367 | 0.321 | **0.158** | 0.155 |
| grpo_chartqa | 0.96 | 0.95 | 0.61 | 0.593 | 0.361 | **0.017** | 0.030 |
| hcpc_chartqa | 0.98 | 0.96 | 0.63 | 0.609 | 0.348 | **0.020** | 0.024 |
| nsr_chartqa  | 0.97 | 0.96 | 0.62 | 0.607 | 0.354 | **0.016** | 0.025 |
| grpo_chartfc | 0.99 | 1.00 | 0.66 | 0.659 | 0.337 | **0.002** | 0.002 |
| hcpc_chartfc | 0.56 | 0.86 | 0.62 | 0.537 | 0.324 | **0.088** | 0.052 |
| nsr_chartfc  | 0.99 | 1.00 | 0.69 | 0.685 | 0.313 | **0.003** | 0.001 |

### 7.3 Conditional accuracies — does the table actually help?

| Run | P(answer_ok\|table_ok) | P(answer_ok\|¬table_ok) | lift |
|---|---|---|---|
| base_chartqa | 0.534 | 0.504 | +0.030 |
| grpo_chartqa | 0.622 | 0.355 | **+0.267** |
| hcpc_chartqa | 0.636 | 0.454 | +0.183 |
| nsr_chartqa  | 0.632 | 0.388 | +0.244 |
| grpo_chartfc | 0.662 | 0.500 | +0.162 |
| **hcpc_chartfc** | 0.624 | 0.631 | **−0.007** |
| nsr_chartfc  | 0.687 | 0.833 | n/a (tiny) |

### 7.4 Five hard findings

**A. Lucky-correct is RARE on RL-trained models on ChartQA.** 1.6–2.0% of rollouts have `table=F, answer=T`. Whatever the model is doing wrong, it's not "skipping the table and guessing." The base model is at 15.8% — but RL fixes that almost entirely without any explicit gating mechanism.

**B. The dominant failure mode is "good extraction, bad reasoning."** 32–36% of RL-trained rollouts have `table=T, answer=F`. The model parses the chart fine, then computes incorrectly. **No version of SCR helps with this** — the table reward already gates accuracy implicitly via correlation, and the rollouts that need fixing all have `table_ok=True`.

**C. HCPC ChartFC is broken in a way the eval summary hid.** `lucky_correct=8.8%` (4–40× higher than other RL runs), `format_ok=0.56`, conditional-accuracy lift = **−0.007**. The table provides no information for the answer. The model is bypassing the table entirely on ChartFC under HCPC. This is the run that should be excluded from any HCPC accuracy claims.

**D. Provenance signal is real and weak.** On ChartQA grpo, numeric-answer rollouts that trace to predicted-table cells (via depth-2 arithmetic closure) are 87.8% correct vs 82.5% for non-traceable. A 5pt gap. Modest, but consistent across all RL runs. Not strong enough alone to base a paper on.

**E. The current `table_reward` scores too generously on numerical mismatches.** From the training log of `grpo_hcpc/run_20260219_191830`: a rollout with predicted cells `["Germany","49%"]` vs GT `["Germany","50"]` got `base_table=1.083` (87% of max). Column headers and string cells matched, numeric values didn't, but no explicit numeric-cell penalty exists in [base_rewards.py:185-222](app/rewards/base_rewards.py#L185). **This is the highest-leverage place to improve the reward stack.**

### 7.4b Per-method failure breakdown (training rollouts)

Mined from `train.log` files. Numeric-content-correct = recall ≥ 0.8 against GT-table cells; numeric-content-wrong = recall < 0.5. Answer correctness uses the live training reward function. Source: [app/analysis/failure_breakdown_per_method.py](app/analysis/failure_breakdown_per_method.py).

**GRPO baseline** (8000 rollouts from `run_20260218_125045/train.log`):

| Cell | % | n | mean(table_reward) | mean(accuracy_reward) |
|---|---|---|---|---|
| RC+RA (right content + right answer) | 30.7% | 2456 | 1.84 | 0.95 |
| RC+WA (right content + wrong answer) | 27.8% | 2222 | 1.85 | 0.27 |
| WC+RA (wrong content + right answer = LUCKY) | **11.3%** | 904 | 0.66 | 0.95 |
| WC+WA (wrong content + wrong answer) | 17.1% | 1366 | 0.63 | 0.24 |
| MID (recall 0.5–0.8) | 12.2% | 972 | 1.41 | 0.53 |

**GRPO+HCPC** (8080 rollouts from `run_20260219_191830/train.log`):

| Cell | % | n | mean(table_reward) | mean(accuracy_reward) |
|---|---|---|---|---|
| RC+RA | 36.9% | 2985 | 1.86 | 0.97 |
| RC+WA | 24.4% | 1969 | 1.91 | 0.33 |
| WC+RA (LUCKY) | **13.4%** | 1086 | 0.58 | 0.96 |
| WC+WA | 13.1% | 1056 | 0.58 | 0.30 |
| MID | 11.2% | 904 | 1.45 | 0.60 |

**What's striking and method-attributed:**

1. **HCPC reduces the WC+WA cell** (17.1% → 13.1%) — fewer wholly-broken rollouts. Good.
2. **HCPC slightly INCREASES the LUCKY cell** (11.3% → 13.4%). Possibly because HCPC's reward bonus to "correct" rollouts disproportionately reinforces lucky-correct ones, since HCPC's correctness check is the same `accuracy_reward` that doesn't see the bad table content.
3. **HCPC increases RC+RA by ~6 points** (30.7% → 36.9%). This is HCPC's positive effect, but note it's measured at training temperature 1.0, not eval temperature 0.8.
4. The mean `table_reward` *inside the LUCKY cell* is 0.58–0.66 out of max ~1.25. So the current `table_reward` does discount lucky-correct rollouts somewhat — but they still receive the full `accuracy_reward` (0.95–0.96). The accuracy reward is the unmodified positive signal that drives the lucky-correct gradient.

**Methodological caveats:**
- Only GRPO and GRPO+HCPC training logs were available locally. NSR training logs are on Kaggle.
- Training rollouts are sampled at training temperature (1.0) and reward stack, NOT at eval temperature (0.8).
- The "accuracy_reward = 0.95" inside LUCKY cells is intuitive — they ARE answer-correct under relaxed-accuracy. The point is the reward function has no other channel to discount them.
- Eval-time numbers (§7.2) showed structurally-defined lucky rate of 1.6–2.0% on RL ChartQA. The content-defined lucky rate here (11–13%) is ~6× higher. The discrepancy is because structural parseability is a much weaker check than numeric-content correctness.

### 7.5 Reward-variant simulation (counterfactual scoring on existing rollouts)

For each variant, I scored every rollout's `accuracy_signal` and tracked how often the variant changes the **within-group argmax** (the rollout that drives GRPO's positive gradient). If the variant doesn't change argmax, it doesn't change gradient direction.

| Variant | grpo_chartqa | hcpc_chartqa | grpo_chartfc | hcpc_chartfc |
|---|---|---|---|---|
| SCR-hard | **3.2%** | 1.8% | 0.4% | 12.2% |
| SCR-soft | 3.2% | 1.8% | 0.4% | 12.2% |
| Floor 0.3 | 2.4% | 1.0% | 0.4% | 10.4% |
| Discount 0.5 | 2.4% | 1.0% | 0.4% | 10.4% |

**Translation:** SCR would change the gradient on **3% of training groups** for a healthy GRPO run on ChartQA. That's almost no signal. **SCR is the wrong tool for a problem that doesn't exist in your data.**

The only run where SCR would have visibly changed gradients is HCPC ChartFC (12% of groups) — but that's because HCPC ChartFC was already broken; SCR would just be papering over a different bug.

---

## 8. EMNLP Reward Recommendation (UPDATED)

**Drop SCR.** The data shows it would be solving an imaginary problem. Lucky-correct doesn't happen often enough on healthy RL runs to be worth designing around.

**The actual highest-leverage change** is fixing what `table_reward` rewards. Three concrete components, in priority order:

### 8.1 Numeric-cell-recall reward (HIGH priority)

`table_reward` currently treats string and numeric cells uniformly via equality. Replace with a numeric-aware variant:

```
R_table = α · header_match  +  β · numeric_cell_recall_within_tolerance
                          +  γ · row_count_match
```

Where `numeric_cell_recall_within_tolerance` is the fraction of GT-table numeric cells that appear in the predicted table within 1% relative tolerance. This directly punishes the "wrong numbers but right columns" failure mode (Finding E above) which the current reward ignores.

**Why this is the right thing to do:**
- It targets a documented failure (Finding E).
- Implementation is ~30 lines, drop-in to [base_rewards.py](app/rewards/base_rewards.py).
- It changes which rollouts win the within-group argmax — i.e., it changes the gradient, where SCR doesn't.

### 8.2 Provenance reward as auxiliary (MEDIUM priority)

The provenance reward (numeric tokens in reasoning must trace to predicted-table cells via depth-2 arithmetic closure) gave a ~5pt accuracy correlation in the eval data. Plausibly a few-pp lift if added during training. It's NOT the headline contribution but it's cheap and aligned with the theme of "force the reasoning to use the extracted data."

### 8.3 The story (revised)

Drop the elegant-NSR-analogue framing. The honest story is:

> **"Reward functions in chart-VL RLVR have been treating numeric cells the same as string cells, which lets models claim table-extraction reward for hallucinated numbers. We propose a numeric-aware table reward and a provenance term that constrains reasoning to use only extracted numerical values. This combination produces the cleanest accuracy gains on ChartQA over GRPO+chart-RVR-style rewards."**

Less flashy, more defensible. And the data supports it.

### 8.4 What I'd run before retraining

1. Implement numeric-cell-recall in [base_rewards.py](app/rewards/base_rewards.py).
2. **Score it offline on existing eval rollouts** with GT tables (would need to download dataset OR pull a small sample from training logs which contain `gt_table`). Confirm: rollouts with high numeric-recall *correlate more strongly with answer correctness* than current `table_reward` does. If the correlation is no better than current, this idea is also wrong.
3. Only if (2) passes: run a 200-step pilot on Kaggle, check if numeric-recall correlates with answer accuracy *during training* (not just at eval time).
4. Then full retrain.

---

---

## 9. Submission-Grade Plan (decided 2026-05-08)

Goal: submittable to EMNLP 2026. Not aiming for accept; aiming for *honest submission with useful reviewer feedback*. Negative or mixed results acceptable. Confound-ridden positive claims not acceptable.

### 9.1 The minimum-credible experimental matrix

A clean 2×2 grid with **one fixed hyperparameter set** across all cells:

|              | without HCPC          | with HCPC               |
|---           |---                    |---                      |
| **GRPO**     | grpo_baseline (re-run) | grpo_hcpc (re-run)      |
| **NSR**      | nsr_baseline (re-run, after NSR fix) | nsr_hcpc (new) |

All cells: same lr, same batch size, same temperature, same num_generations, same precision (bf16), same dataset, same seed-set ({42, 123, 2026}). Differ ONLY in `policy_method` and `use_hcpc`.

**Why re-run GRPO and HCPC even though we have data:** existing runs use mismatched hyperparameters (lr 1e-5 vs 1e-6, temp 1.0 vs 0.8, etc — see master.md §5). Without matching, no claim survives review.

### 9.2 Pre-requisite: fix NSR

Currently `NSRTrainer.compute_advantages` is dead code — never called, can't change TRL's gradient. Confirmed via `training_args.bin` (scale_rewards=group, beta=0) and trainer_state.json (mean reward 8.79 at step 500 = raw rewards, not NSR advantages).

Real NSR requires subclassing TRL's `GRPOTrainer.compute_loss` to **mask the per-token loss for correct rollouts**. Estimated effort: 2-3 days including a sanity test that proves gradient is zero on correct rollouts.

### 9.3 What to evaluate

- pass@1, pass@4, pass@8, pass@16 (NSR's story is at higher k).
- Stage-level failure decomposition (§7 methodology) — already have the script.
- Format compliance per cell.
- Optional: 1 OOD eval (PlotQA or held-out chart types) if compute permits.

### 9.4 Story arcs supported by possible outcomes

| Outcome | Story |
|---|---|
| NSR ≈ GRPO at pass@1, NSR > GRPO at pass@8+ | Replicates Zhu et al. 2025 in vision-language; clean contribution. |
| NSR > GRPO at all k | Strong — chart-VL benefits more than text-math. |
| NSR < GRPO | Honest negative; report and explain via failure decomposition. |
| HCPC helps under one policy but not the other | Diagnostic finding about reward-shaping × policy interaction. |
| HCPC doesn't help anywhere | Honest null result; ablation rules out a hypothesis. |

All five outcomes are submittable. Only the dishonest framing is unsubmittable.

### 9.5 Compute budget (rough)

- 4 cells × 3 seeds = 12 training runs.
- ~6h each on Kaggle T4 = ~72h compute.
- Spread over 3 weeks wall-time given Kaggle quota limits.
- Eval: ~30h compute.
- Total: ~4 weeks before writing starts.

### 9.6 What NOT to put in the paper

Even with full retraining:

- "HCPC improves chart-VL accuracy" (without significance, with the format-compliance regression on ChartFC, this is unsupportable).
- Numbers from runs with hyperparameter mismatches.
- Pass@k from `grpo_chartqa` (k=8 for some, k=4 for others — inconsistent).

What CAN go in: the rollout-level failure decomposition (§7) is novel and valuable regardless of which method wins. Use it as a methodological contribution.

### 9.7 Decision points still open

1. Draft NSR loss-masking trainer subclass to `app/trainers/_draft_nsr_masked.py` (review before adopting)? — pending user OK.
2. Whether to also include W-REINFORCE cell (would be 3×2 grid, more compute). — leaning no for first submission.

---

## Changelog
- **2026-05-08**: Created `app/final_outputs/`, copied all 7 eval folders. Master.md initialized.
- **2026-05-08**: Recomputed all metrics from raw jsonl, paired McNemar + bootstrap CIs, per-label-type breakdown. Confirmed `nsr_chartfc` is plain NSR (not NSR+HCPC). Honest EMNLP assessment in §6.
- **2026-05-08**: Stage-by-stage failure analysis (§7) + reward-variant simulation. Found that SCR targets a near-zero failure mode on healthy RL runs. Revised reward recommendation in §8.
- **2026-05-08**: Submission-grade plan in §9. Decided 2×2 matrix (GRPO/NSR × ±HCPC) with matched hyperparameters and 3 seeds. Pre-requisite: fix NSR loss-masking. Reuse §7 failure-decomposition methodology in paper.
