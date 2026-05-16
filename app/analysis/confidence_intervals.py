"""Confidence intervals.

- Wilson 95% CI for each method's pass@1 / pass@k.
- Paired bootstrap 95% CI on accuracy gaps between methods.
"""
import json
import math
import random
from pathlib import Path
from itertools import combinations

ROOT = Path(__file__).resolve().parents[0] / "final_outputs"
random.seed(2026)

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

def wilson_ci(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    half = (z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / denom
    return (centre - half, centre + half)


def load_first(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            cl = r.get("correct", []) or []
            out[r["idx"]] = bool(cl[0]) if cl else False
    return out


def load_any(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            cl = r.get("correct", []) or []
            out[r["idx"]] = any(cl)
    return out


def paired_bootstrap_ci(arr_a, arr_b, n_boot=10000):
    """arr_a, arr_b are bool lists, paired by index. Return 95% CI on mean(a) - mean(b)."""
    n = len(arr_a)
    diffs = []
    a = [int(x) for x in arr_a]
    b = [int(x) for x in arr_b]
    for _ in range(n_boot):
        idxs = [random.randrange(n) for _ in range(n)]
        ma = sum(a[i] for i in idxs) / n
        mb = sum(b[i] for i in idxs) / n
        diffs.append(ma - mb)
    diffs.sort()
    lo = diffs[int(0.025 * n_boot)]
    hi = diffs[int(0.975 * n_boot)]
    return lo, hi


for group, runs in GROUPS.items():
    print(f"\n========== {group} ==========")
    first = {n: load_first(p) for n, p in runs.items()}
    anyc  = {n: load_any(p)   for n, p in runs.items()}
    names = list(first)
    n_total = len(first[names[0]])

    print(f"  Wilson 95% CI for pass@1 (n={n_total}):")
    for n in names:
        k = sum(first[n].values())
        lo, hi = wilson_ci(k, n_total)
        print(f"    {n:6s}  acc={k/n_total:.4f}  CI=[{lo:.4f}, {hi:.4f}]")

    print(f"\n  Wilson 95% CI for pass@k (n={n_total}):")
    for n in names:
        k = sum(anyc[n].values())
        lo, hi = wilson_ci(k, n_total)
        print(f"    {n:6s}  acc={k/n_total:.4f}  CI=[{lo:.4f}, {hi:.4f}]")

    print(f"\n  Paired bootstrap 95% CI on pass@1 gap (a - b):")
    indices = list(first[names[0]])
    for a, b in combinations(names, 2):
        arr_a = [first[a][i] for i in indices]
        arr_b = [first[b][i] for i in indices]
        diff = sum(int(x) for x in arr_a)/len(arr_a) - sum(int(x) for x in arr_b)/len(arr_b)
        lo, hi = paired_bootstrap_ci(arr_a, arr_b, n_boot=5000)
        sig = "***" if (lo > 0 or hi < 0) else "ns "
        print(f"    {sig}  {a:>6s} - {b:<6s}  diff={diff:+.4f}  CI=[{lo:+.4f}, {hi:+.4f}]")
