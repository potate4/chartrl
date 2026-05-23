# EMNLP Short Paper Draft

`main.tex` is the draft. To compile, drop in the ACL style files
from <https://github.com/acl-org/acl-style-files> (you need `acl.sty`,
`acl_natbib.bst`) into this directory along with an `anthology.bib` /
`custom.bib`, then:

```
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

## Current title
**Reward-Channel Separability in Multimodal Chart RLVR: A Cross-Rollout Bonus and a Negative-Sample Caveat**

## Source of numbers in Table 1
All numbers come from `app/final_outputs/<run>/summary.json`:

| Method | Source folder |
|---|---|
| Base | `base_chartqa` |
| GRPO | `grpo_chartqa`, `grpo_chartfc` |
| GRPO+HCPC | `hcpc_chartqa`, `hcpc_chartfc` |
| NSR | `nsr_baseline_chartqa`, `nsr_baseline_chartfc` |
| NSR+HCPC | `nsr_hcpc_chartqa` |

The "prev" NSR runs (`nsr_chartqa_prev`, `nsr_chartfc_prev`) were
deliberately excluded — they were trained before the NSR advantage bug
was fixed and reported NSR-like accuracy only because the advantage
estimator was effectively still GRPO-like.

## Stats backing Table 1 and Table 2
All numbers are reproducible by running:

```
cd app && python analysis/paper_stats.py
```

The captured output is in `app/analysis/paper_stats_output.txt`.
Key numbers used in the paper:

- ChartQA Pass@1: base 0.518, GRPO 0.630, HCPC 0.622, NSR 0.592, NSR+HCPC 0.574
- ChartQA Pass@4: base 0.774, GRPO 0.816, HCPC 0.828, NSR 0.808, NSR+HCPC 0.800
- ChartQA Δ_tbl (conditional lift P(ans|table) − P(ans|~table)):
  base +0.021, GRPO +0.132, HCPC +0.216, NSR +0.090, NSR+HCPC +0.118
- ChartFC Pass@1: GRPO 0.638, HCPC 0.606, NSR 0.590, NSR+HCPC 0.632
- ChartFC Pass@4: GRPO 0.922, HCPC 0.918, NSR 0.884, NSR+HCPC 0.894
  (GRPO vs NSR sig: +0.038 p=0.028; GRPO vs NSR+HCPC ns, +0.028 p=0.060)
- ChartFC Δ_tbl (per-sample rollout-0): GRPO +0.139, HCPC −0.059,
  NSR −0.059, NSR+HCPC +0.002
- ChartFC Fmt%: GRPO 98.8, HCPC 66.4, NSR 16.2, NSR+HCPC 17.8
  (key: NSR+HCPC matches GRPO Pass@1 at near-zero format compliance —
  format/accuracy dissociation discussed in §5.2)
- Per-tag (think/answer/type/table/order) format rates: see Table 2 in paper

## Bib entries still to fill in
The `.tex` references the following keys; you'll need entries for each
in `custom.bib`:

chartrvr2025, chartr12025, bigcharts2025, distill-vcr-2025, deepseekmath,
zhu2025nsr, wang2022sc, chartqa, chartfc, passatk, minilm, chartx2024,
chartgemma, chartinstruct, chartpoint2025, chartqax2026, vprochart2025,
chartcitor2025, chartagent2025, su2025crossing, wen2025rlvr, rlvmr2025,
koksal2025fewshot, ettrl2025, evochart, chartqapro.

CLC references have been removed.

The matching arXiv/ACL anthology URLs are already in
`final report/paper_list.txt`.
