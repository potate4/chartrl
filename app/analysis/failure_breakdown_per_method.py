"""Per-method 4-way failure mode breakdown from training logs.

Splits each rollout into one of four cells:
  WC+WA  Wrong content + wrong answer  (numeric_recall < 0.5, answer wrong)
  WC+RA  Wrong content + right answer  (numeric_recall < 0.5, answer correct)  -- "lucky"
  RC+WA  Right content + wrong answer  (numeric_recall >= 0.8, answer wrong)
  RC+RA  Right content + right answer  (numeric_recall >= 0.8, answer correct)
  MID    Recall in [0.5, 0.8)          -- ambiguous

Reports row %, plus how the existing table_reward and accuracy_reward
behave inside each cell.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from table_content_analysis import parse_log, parse_table_from_completion, parse_answer_from_completion, numeric_recall, answers_match

LOGS = {
    "grpo_baseline":  Path(r"app/outputs/grpo_baseline/run_20260218_125045/train.log"),
    "grpo_hcpc":      Path(r"app/outputs/grpo_hcpc/run_20260219_191830/train.log"),
}


def categorize(recall, ans_ok):
    if recall is None: return "no-num"
    if recall >= 0.8:
        return "RC+RA" if ans_ok else "RC+WA"
    if recall < 0.5:
        return "WC+RA" if ans_ok else "WC+WA"
    return "MID"


def analyze(name, log_path):
    print(f"\n========== {name} ==========")
    print(f"  source: {log_path}")
    rows = []
    for rec in parse_log(log_path):
        if rec["gt_table"] is None or not rec["completion"]:
            continue
        pred_tab = parse_table_from_completion(rec["completion"])
        pred_ans = parse_answer_from_completion(rec["completion"])
        nrec = numeric_recall(pred_tab, rec["gt_table"])
        ans_ok = answers_match(pred_ans, rec["gt_label"])
        rows.append({
            "cat": categorize(nrec, ans_ok),
            "recall": nrec,
            "ans_ok": ans_ok,
            "table_reward": rec["rewards"].get("base_table"),
            "accuracy_reward": rec["rewards"].get("base_accuracy"),
        })

    n = len(rows)
    print(f"  n_rollouts (with GT numeric table): {n}")

    cats = ["RC+RA", "RC+WA", "WC+RA", "WC+WA", "MID", "no-num"]
    print(f"\n  {'cell':<8s} {'%':>7s} {'n':>6s} {'mean(table_reward)':>20s} {'mean(accuracy_reward)':>22s}")
    for c in cats:
        sub = [r for r in rows if r["cat"] == c]
        if not sub: continue
        pct = 100 * len(sub) / n
        # avg current rewards in this cell
        trs = [r["table_reward"] for r in sub if r["table_reward"] is not None]
        ars = [r["accuracy_reward"] for r in sub if r["accuracy_reward"] is not None]
        mtr = sum(trs)/len(trs) if trs else float("nan")
        mar = sum(ars)/len(ars) if ars else float("nan")
        print(f"  {c:<8s} {pct:>6.2f}% {len(sub):>6d} {mtr:>20.4f} {mar:>22.4f}")

    # Headline summary line
    rcra = sum(1 for r in rows if r["cat"] == "RC+RA")
    rcwa = sum(1 for r in rows if r["cat"] == "RC+WA")
    wcra = sum(1 for r in rows if r["cat"] == "WC+RA")
    wcwa = sum(1 for r in rows if r["cat"] == "WC+WA")
    mid  = sum(1 for r in rows if r["cat"] == "MID")
    print(f"\n  Headline (% of all rollouts):")
    print(f"    Right content + Right answer : {100*rcra/n:5.1f}%")
    print(f"    Right content + Wrong answer : {100*rcwa/n:5.1f}%   <- 'good extraction, bad reasoning'")
    print(f"    Wrong content + Right answer : {100*wcra/n:5.1f}%   <- LUCKY-CORRECT")
    print(f"    Wrong content + Wrong answer : {100*wcwa/n:5.1f}%")
    print(f"    Mid-recall (0.5-0.8)         : {100*mid/n:5.1f}%")


for name, log in LOGS.items():
    analyze(name, log)
