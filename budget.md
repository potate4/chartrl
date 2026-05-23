# Compute Budget — HCPC-RLVR / NSR Chart VL 

**Rate**: L4 GPU @ $0.40/hr (avg Runpod or Modal)
**Real benchmarks**: Training ~15hr on L4, ~9hr on H100 | Eval ~6hr on L4

---

## A. Already Spent (No Additional Cost)

| Resource | What ran | Est. hours | Cost |
|---|---|---|---|
| Lab PC (RTX 3090) | GRPO baseline (2000 steps) | ~20hr | $0 (lab) |
| Lab PC (RTX 3090) | GRPO+HCPC (2000 steps) | ~25hr | $0 (lab) |
| Lab PC (RTX 3090) | GRPO+HCPC partial (400 steps) | ~5hr | $0 (lab) |
| Kaggle H100 (free) | NSR baseline (2000 steps, dead algo) | ~9hr | $0 (free tier) |
| Kaggle H100 (free) | Various eval runs (base, grpo, hcpc, nsr) | ~30hr | $0 (free tier) |
| Google Colab Pro | 1 training run + eval + miscellaneous | — | $9.99 + VAT ≈ **$12** |
| **Total already spent** | | | **~$12** |

---

## B. Current Runs (EMNLP ARR May 25)

These are the 4 runs needed for the clean 2×2 grid + evals.

| # | Experiment | Train hrs | Eval hrs (ChartQA + ChartFC) | L4 cost |
|---|---|---|---|---|
| 1 | NSR baseline (running) | 15hr | 12hr | $10.80 |
| 2 | NSR+HCPC (running) | 15hr | 12hr | $10.80 |
| 3 | GRPO baseline rerun (matched HPs) | 15hr | 12hr | $10.80 |
| 4 | GRPO+HCPC rerun (matched HPs) | 15hr | 12hr | $10.80 |
| | **Subtotal** | 60hr | 48hr | **$43.20** |

**Immediate total**: ~$43

---

## C. Extension — Few-Shot NSR Transfer (Future Runs)

Scaling curve: GRPO and NSR at 16 / 64 / 256 shot + 2 more full-data seeds each.

| # | Experiment | Count | Train hrs each | Total train hrs | Total eval hrs | L4 cost |
|---|---|---|---|---|---|---|
| Few-shot GRPO | 16 / 64 / 256 shot | 3 | ~4hr avg | 12hr | 18hr | $12.00 |
| Few-shot NSR | 16 / 64 / 256 shot | 3 | ~4hr avg | 12hr | 18hr | $12.00 |
| Extra seeds GRPO (full) | 2 more seeds | 2 | 15hr | 30hr | 12hr | $16.80 |
| Extra seeds NSR (full) | 2 more seeds | 2 | 15hr | 30hr | 12hr | $16.80 |
| | **Subtotal** | 10 runs | | 84hr train + 60hr eval | | **$57.60** |

> Few-shot runs are short (~4hr each on L4) — cheapest part of the extension.

---

## D. Total Budget Summary

| Phase | Cost |
|---|---|
| A. Already spent | $12 |
| B. Current (4 runs + evals) | $43 |
| C. Extension (few-shot + seeds, 1k data) | $58 |
| **Grand total (1k scale)** | **~$113** |
| E. If scaled to full dataset (34k) | $200–$440 |
| **Grand total (full scale)** | **~$213–$453** |

---

## E. Full-Scale Training (34k train / 34k eval)

Current experiments use 1000/500 samples. If scaled to the full dataset (34k training, ~34k eval), costs increase substantially. Assumes batch size scaled up to keep wall time reasonable (~4× longer training, ~8× longer eval vs current).

| Experiment | Train hrs (L4) | Eval hrs (L4) | L4 cost |
|---|---|---|---|
| GRPO baseline (full data) | ~60hr | ~50hr | $44.00 |
| GRPO+HCPC (full data) | ~60hr | ~50hr | $44.00 |
| NSR baseline (full data) | ~60hr | ~50hr | $44.00 |
| NSR+HCPC (full data) | ~60hr | ~50hr | $44.00 |
| Few-shot curve (6 runs, full eval) | ~120hr | ~200hr | $128.00 |
| Extra seeds ×2 per main method | ~240hr | ~100hr | $136.00 |
| **Subtotal** | | | **~$440** |


---

## Notes

- Kaggle H100 free competition tier no longer available. All future compute is paid L4 @ $0.40/hr on Runpod or Modal with some additional charges for storage.
- Colab Pro subscription ($9.99+VAT) already paid — not counted in future costs.
