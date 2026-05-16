"""Paired comparisons across runs (same idx -> same sample).

We compute:
- per-(idx) first_correct vector for each run
- McNemar contingency for each pair (where it makes sense)
- agreement / disagreement counts
- "wins" matrix
"""
import json
import math
from pathlib import Path
from itertools import combinations

ROOT = Path(__file__).resolve().parents[0] / "final_outputs"

GROUPS = {
    "chartqa": {
        "base":  ROOT / "base_chartqa"  / "per_sample.jsonl",
        "grpo":  ROOT / "grpo_chartqa"  / "per_sample.jsonl",
        "hcpc":  ROOT / "hcpc_chartqa"  / "per_sample.jsonl",
        "nsr":   ROOT / "nsr_chartqa"   / "per_sample.jsonl",
    },
    "chartfc": {
        "grpo":  ROOT / "grpo_chartfc"  / "per_sample.jsonl",
        "hcpc":  ROOT / "hcpc_chartfc"  / "per_sample.jsonl",
        "nsr":   ROOT / "nsr_chartfc"   / "per_sample (2).jsonl",
    },
}


def load_idx_to_first(path):
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            cl = r.get("correct", []) or []
            out[r["idx"]] = bool(cl[0]) if cl else False
    return out


def load_idx_to_any(path):
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            cl = r.get("correct", []) or []
            out[r["idx"]] = any(cl)
    return out


def mcnemar_exact(b, c):
    """Two-sided exact binomial p for McNemar with discordant counts b,c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # P(X <= k) under Binomial(n, 0.5), times 2
    from math import comb
    p = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * p)


for group, runs in GROUPS.items():
    print(f"\n========== {group} ==========")
    first = {name: load_idx_to_first(p) for name, p in runs.items()}
    anyc  = {name: load_idx_to_any(p)   for name, p in runs.items()}

    # accuracy at first
    print("\n  pass@1 (first prediction correct):")
    for name, m in first.items():
        acc = sum(m.values()) / len(m)
        print(f"    {name:6s}  {acc:.4f}  ({sum(m.values())}/{len(m)})")

    print("\n  pass@k (any of k correct):")
    for name, m in anyc.items():
        acc = sum(m.values()) / len(m)
        print(f"    {name:6s}  {acc:.4f}  ({sum(m.values())}/{len(m)})")

    print("\n  Pairwise McNemar on first-pred (a vs b: a-only-correct, b-only-correct, p):")
    names = list(first)
    for a, b in combinations(names, 2):
        ma, mb = first[a], first[b]
        only_a = sum(1 for i in ma if ma[i] and not mb[i])
        only_b = sum(1 for i in ma if mb[i] and not ma[i])
        both   = sum(1 for i in ma if ma[i] and mb[i])
        neither = sum(1 for i in ma if not ma[i] and not mb[i])
        p = mcnemar_exact(only_a, only_b)
        print(f"    {a:>6s} vs {b:<6s}  both={both}  neither={neither}  only-{a}={only_a}  only-{b}={only_b}   p={p:.4g}")
