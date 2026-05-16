"""Score reward variants on existing eval rollouts.

Variants compared:
1. ADDITIVE  (current): base_total = format + acc + table + ...
2. SCR-HARD: zero accuracy reward when table is structurally broken
3. SCR-SOFT: multiply accuracy reward by table_quality in [0,1]
4. FLOOR(0.3): final_acc = max(0.3, table_q) * acc, never zero out
5. DISCOUNT: final_acc = acc - 0.5 * (1 - table_q)

We compute:
- The signal differential each variant would give to the model.
- How often each variant changes the relative ranking of rollouts WITHIN a group of 4
  (this is what drives GRPO's gradient).
- Pearson-like correlation: does the variant correlate better with eventual answer
  correctness than ADDITIVE does?

Key practical question: does the variant put MORE positive signal on
(table_ok=T, answer=T) and LESS on (table_ok=F, answer=T) compared to ADDITIVE?
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[0] / "final_outputs"

RUNS = {
    "grpo_chartqa":  ROOT / "grpo_chartqa"  / "per_sample.jsonl",
    "hcpc_chartqa":  ROOT / "hcpc_chartqa"  / "per_sample.jsonl",
    "grpo_chartfc":  ROOT / "grpo_chartfc"  / "per_sample.jsonl",
    "hcpc_chartfc":  ROOT / "hcpc_chartfc"  / "per_sample.jsonl",
}


def extract_block(text: str, o: str, c: str) -> str:
    if o not in text or c not in text:
        return ""
    return text.split(o, 1)[1].split(c, 1)[0]


def parse_table_strict(raw: str) -> Optional[dict]:
    blk = extract_block(raw, "<table>", "</table>").strip()
    if not blk:
        return None
    if "```json" in blk:
        blk = blk.split("```json", 1)[1].split("```", 1)[0].strip()
    try:
        obj = json.loads(blk)
    except Exception:
        return None
    if isinstance(obj, dict) and "columns" in obj and "rows" in obj:
        return obj
    return None


def table_quality(raw: str) -> float:
    """Fraction in [0,1]: 0 = no table block, 0.5 = parseable JSON,
    0.75 = right schema, 1.0 = also has rows."""
    blk = extract_block(raw, "<table>", "</table>").strip()
    if not blk:
        return 0.0
    score = 0.0
    if "```json" in blk:
        blk = blk.split("```json", 1)[1].split("```", 1)[0].strip()
    try:
        obj = json.loads(blk)
        score = 0.5
    except Exception:
        return 0.0
    if isinstance(obj, dict) and "columns" in obj and "rows" in obj:
        score = 0.75
        rows = obj.get("rows", [])
        if isinstance(rows, list) and len(rows) > 0:
            score = 1.0
    return score


def variants_for_rollout(raw: str, answer_correct: bool) -> Dict[str, float]:
    """Return reward signal under each variant. Use a stylized 'accuracy'
    component that's 1.0 if answer_correct else 0.0. We're comparing variants
    of the GATING, not measuring the absolute reward magnitudes."""
    acc = 1.0 if answer_correct else 0.0
    tq = table_quality(raw)

    return {
        "additive":     acc,                    # current behavior
        "scr_hard":     acc if tq >= 0.75 else 0.0,
        "scr_soft":     acc * tq,
        "floor_0.3":    acc * max(0.3, tq),
        "discount_0.5": max(0.0, acc - 0.5 * (1 - tq)),
    }


def analyze_run(name: str, path: Path) -> None:
    print(f"\n========== {name} ==========")
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            raws = r.get("raw_outputs", []) or []
            corrects = r.get("correct", []) or []
            row = []
            for k in range(min(len(raws), len(corrects))):
                v = variants_for_rollout(raws[k], bool(corrects[k]))
                v["correct"] = bool(corrects[k])
                v["table_q"] = table_quality(raws[k])
                row.append(v)
            samples.append(row)

    # Aggregate signal per variant
    n_total = sum(len(s) for s in samples)
    print(f"  rollouts: {n_total}")

    variant_names = ["additive", "scr_hard", "scr_soft", "floor_0.3", "discount_0.5"]

    print(f"\n  Mean reward signal (over all rollouts):")
    for vn in variant_names:
        vals = [r[vn] for s in samples for r in s]
        print(f"    {vn:<14s} : {statistics.mean(vals):.4f}")

    # Signal lost / gained on the lucky-correct rollouts
    print(f"\n  On 'lucky-correct' rollouts (correct=T, table_q<0.75):")
    lucky = [r for s in samples for r in s if r["correct"] and r["table_q"] < 0.75]
    if lucky:
        for vn in variant_names:
            mean_v = statistics.mean(r[vn] for r in lucky)
            print(f"    {vn:<14s} : {mean_v:.4f}  (n={len(lucky)})")
    else:
        print(f"    (none)")

    print(f"\n  On 'good-pipeline-correct' rollouts (correct=T, table_q=1.0):")
    good = [r for s in samples for r in s if r["correct"] and r["table_q"] >= 0.99]
    if good:
        for vn in variant_names:
            mean_v = statistics.mean(r[vn] for r in good)
            print(f"    {vn:<14s} : {mean_v:.4f}  (n={len(good)})")

    print(f"\n  On 'wrong despite good pipeline' rollouts (correct=F, table_q=1.0):")
    wgp = [r for s in samples for r in s if not r["correct"] and r["table_q"] >= 0.99]
    if wgp:
        for vn in variant_names:
            mean_v = statistics.mean(r[vn] for r in wgp)
            print(f"    {vn:<14s} : {mean_v:.4f}  (n={len(wgp)})")

    # Within-group ranking change: GRPO uses (r - mean(r)) / std(r) in groups of 4.
    # If the variant doesn't change the within-group order, it doesn't change the
    # gradient direction (only magnitude). Count how often the variant changes the
    # argmax within a group.
    print(f"\n  Within-group argmax changes vs ADDITIVE:")
    for vn in variant_names:
        if vn == "additive":
            continue
        n_changed = 0
        n_groups = 0
        for s in samples:
            if len(s) < 2:
                continue
            n_groups += 1
            add_max = max(range(len(s)), key=lambda i: s[i]["additive"])
            v_max   = max(range(len(s)), key=lambda i: s[i][vn])
            if add_max != v_max:
                n_changed += 1
        print(f"    {vn:<14s} : {n_changed}/{n_groups} ({100*n_changed/max(n_groups,1):.1f}%)")


for name, path in RUNS.items():
    analyze_run(name, path)
