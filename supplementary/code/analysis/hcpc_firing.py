"""HCPC firing rate analysis.

We can't access training-time rollouts; the closest proxy is eval-time
correctness rate. For each method we compute empirically:
- the distribution of |G+| (number of fully-correct rollouts per group)
  using the eval-time 'correct' list as a per-rollout binary proxy for the
  HCPC fully-correct filter.
- the fraction of groups where HCPC would fire (|G+| >= 2).

We also report the theoretical Bernoulli bound from per-rollout correctness:
  P(|G+| < 2 | K=4, p) = (1-p)^4 + 4 p (1-p)^3
"""
import json
from collections import Counter
from pathlib import Path

_SUPPL = Path(__file__).resolve().parents[2]
ROOT = _SUPPL / "eval_outputs"

RUNS = {
    "chartqa": [
        ("base", "base_chartqa"),
        ("grpo", "grpo_chartqa"),
        ("hcpc", "hcpc_chartqa"),
        ("nsr",  "nsr_baseline_chartqa"),
        ("nsr_hcpc", "nsr_hcpc_chartqa"),
    ],
    "chartfc": [
        ("grpo", "grpo_chartfc"),
        ("hcpc", "hcpc_chartfc"),
        ("nsr",  "nsr_baseline_chartfc"),
        ("nsr_hcpc", "nsr_hcpc_chartfc"),
    ],
}


def load(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out.append(r)
    return out


def hcpc_silent(p, K=4):
    # P(|G+| < 2 | K=4, p)
    return (1-p)**4 + 4*p*(1-p)**3


for bench, runs in RUNS.items():
    print(f"\n========== {bench} ==========")
    print(f"  {'method':<10s}  pass1   silent%   |G+|=0   |G+|=1   |G+|=2   |G+|=3   |G+|=4   theory_silent")
    for name, run_dir in runs:
        data = load(ROOT / run_dir / "per_sample.jsonl")
        gp_counts = Counter()
        n_correct = 0
        n_total = 0
        for r in data:
            cl = r.get("correct", []) or []
            cl = [bool(x) for x in cl][:4]  # trim to 4
            if not cl:
                continue
            n_correct += sum(cl)
            n_total += len(cl)
            g_plus = sum(cl)
            gp_counts[g_plus] += 1
        p1 = n_correct / max(1, n_total)
        n_groups = sum(gp_counts.values())
        silent = (gp_counts.get(0, 0) + gp_counts.get(1, 0)) / max(1, n_groups)
        theory = hcpc_silent(p1)
        dist = "  ".join(f"{gp_counts.get(k,0)/max(1,n_groups):.3f}" for k in range(5))
        print(f"  {name:<10s}  {p1:.3f}  {silent*100:5.1f}%   {dist}   {theory*100:5.1f}%")
