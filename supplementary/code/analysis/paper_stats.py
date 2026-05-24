"""Statistics for the EMNLP paper.

Reads per_sample.jsonl from final_outputs/ and prints:
  - Wilson 95% CIs for Pass@1 and Pass@4 per method
  - Paired-bootstrap 95% CIs on accuracy gaps + bootstrap two-sided p-values
  - McNemar exact tests on Pass@1
  - Pass@K curves (K=1..4)
  - Pass@K for "all-K correct" (robustness)
  - Per-answer-type breakdown on ChartQA (HCPC vs GRPO)
  - Format-tag breakdown (where does NSR collapse?)
  - Correlation between table-correctness and answer-correctness within rollouts (a real,
    apples-to-apples coherence number that doesn't suffer the trivial-1.0 bug)
"""
import json
import math
import random
import re
from itertools import combinations
from collections import defaultdict
from math import comb
from pathlib import Path

random.seed(2026)
_SUPPL = Path(__file__).resolve().parents[2]
ROOT = _SUPPL / "eval_outputs"

GROUPS = {
    "chartqa": {
        "base": ROOT / "base_chartqa"         / "per_sample.jsonl",
        "grpo": ROOT / "grpo_chartqa"         / "per_sample.jsonl",
        "hcpc": ROOT / "hcpc_chartqa"         / "per_sample.jsonl",
        "nsr":  ROOT / "nsr_baseline_chartqa" / "per_sample.jsonl",
        "nsr_hcpc": ROOT / "nsr_hcpc_chartqa" / "per_sample.jsonl",
    },
    "chartfc": {
        "grpo": ROOT / "grpo_chartfc"         / "per_sample.jsonl",
        "hcpc": ROOT / "hcpc_chartfc"         / "per_sample.jsonl",
        "nsr":  ROOT / "nsr_baseline_chartfc" / "per_sample.jsonl",
        "nsr_hcpc": ROOT / "nsr_hcpc_chartfc" / "per_sample.jsonl",
    },
}


def load(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["idx"]] = r
    return out


def first_correct(r):
    cl = r.get("correct", []) or []
    return bool(cl[0]) if cl else False


def any_correct(r):
    cl = r.get("correct", []) or []
    return any(cl)


def all_correct(r):
    cl = r.get("correct", []) or []
    return all(cl) if cl else False


def pass_at_k_unbiased(num_correct, k, n_total=4):
    """Standard unbiased Pass@k estimator (k <= n_total)."""
    if num_correct >= n_total - k + 1:
        return 1.0
    if num_correct == 0:
        return 0.0
    return 1.0 - comb(n_total - num_correct, k) / comb(n_total, k)


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (centre - half, centre + half)


def paired_bootstrap(a, b, n_boot=10000):
    """Return (lo, hi, p_two_sided) for mean(a) - mean(b)."""
    n = len(a)
    a = [float(x) for x in a]
    b = [float(x) for x in b]
    diffs = []
    for _ in range(n_boot):
        idxs = [random.randrange(n) for _ in range(n)]
        diffs.append(sum(a[i] - b[i] for i in idxs) / n)
    diffs.sort()
    lo = diffs[int(0.025 * n_boot)]
    hi = diffs[int(0.975 * n_boot)]
    point = sum(a[i] - b[i] for i in range(n)) / n
    # two-sided bootstrap p: 2 * min(P(d<=0), P(d>=0)) under bootstrap distribution
    n_le = sum(1 for d in diffs if d <= 0)
    n_ge = sum(1 for d in diffs if d >= 0)
    p = 2 * min(n_le, n_ge) / n_boot
    return point, lo, hi, p


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * p)


BOOL_WORDS = {"yes", "no", "true", "false", "y", "n"}


def label_type(label):
    s = str(label).strip()
    if s.lower() in BOOL_WORDS:
        return "boolean"
    if (s.startswith("[") and s.endswith("]")) or " and " in s.lower():
        inner = s.strip("[]")
        parts = re.split(r"[,]| and ", inner)
        if len([p for p in parts if p.strip()]) >= 2:
            return "list"
    cleaned = re.sub(r"[\$,%]", "", s.lower())
    cleaned = re.sub(r"\s*(million|billion|thousand|k|m|b)\s*$", "", cleaned).strip()
    try:
        float(cleaned)
        return "numeric"
    except ValueError:
        return "string"


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ============================================================================
# Per-method headline numbers with CIs
# ============================================================================
for bench, runs in GROUPS.items():
    section(f"{bench}: Wilson 95% CIs (Pass@1, Pass@4, all-4)")
    data = {name: load(p) for name, p in runs.items()}
    idxs = sorted(next(iter(data.values())).keys())
    n = len(idxs)
    print(f"  n = {n}")
    print(f"  {'method':<10s} {'Pass@1':>8s}  {'95% CI':<18s} {'Pass@4':>8s}  {'95% CI':<18s} {'all-4':>8s}")
    for name, d in data.items():
        p1 = sum(first_correct(d[i]) for i in idxs)
        p4 = sum(any_correct(d[i]) for i in idxs)
        a4 = sum(all_correct(d[i]) for i in idxs)
        lo1, hi1 = wilson_ci(p1, n)
        lo4, hi4 = wilson_ci(p4, n)
        print(f"  {name:<10s} {p1/n:>8.4f}  [{lo1:.3f},{hi1:.3f}]  {p4/n:>8.4f}  [{lo4:.3f},{hi4:.3f}]  {a4/n:>8.4f}")


# ============================================================================
# Pass@K curves
# ============================================================================
section("Pass@K curves (unbiased estimator, k=1..4)")
for bench, runs in GROUPS.items():
    print(f"\n  --- {bench} ---")
    data = {name: load(p) for name, p in runs.items()}
    idxs = sorted(next(iter(data.values())).keys())
    print(f"  {'method':<10s}  K=1     K=2     K=3     K=4")
    for name, d in data.items():
        avg = [0.0] * 4
        for i in idxs:
            cl = d[i].get("correct", []) or []
            c = sum(1 for x in cl if x)
            for k in range(1, 5):
                avg[k - 1] += pass_at_k_unbiased(c, k, 4)
        n = len(idxs)
        print(f"  {name:<10s}  " + "  ".join(f"{x/n:.4f}" for x in avg))


# ============================================================================
# Paired-bootstrap CIs + McNemar p-values (the table the paper needs)
# ============================================================================
for bench, runs in GROUPS.items():
    section(f"{bench}: Paired bootstrap CIs + McNemar p (10K resamples)")
    data = {name: load(p) for name, p in runs.items()}
    idxs = sorted(next(iter(data.values())).keys())
    names = list(data)
    print(f"  {'pair (a vs b)':<22s} {'metric':<8s} {'mean(a)':>8s} {'mean(b)':>8s} "
          f"{'a-b':>9s}  {'95% CI':<22s} {'boot-p':>8s} {'McN-p':>8s}")
    for a, b in combinations(names, 2):
        for metric_name, fn in [("Pass@1", first_correct), ("Pass@4", any_correct)]:
            va = [fn(data[a][i]) for i in idxs]
            vb = [fn(data[b][i]) for i in idxs]
            ma = sum(va) / len(va)
            mb = sum(vb) / len(vb)
            diff, lo, hi, p = paired_bootstrap([int(x) for x in va], [int(x) for x in vb], n_boot=10000)
            only_a = sum(1 for j in range(len(idxs)) if va[j] and not vb[j])
            only_b = sum(1 for j in range(len(idxs)) if vb[j] and not va[j])
            mcp = mcnemar_exact(only_a, only_b)
            sig = "*" if (lo > 0 or hi < 0) else " "
            print(f"  {sig} {a:>8s} vs {b:<8s} {metric_name:<8s} {ma:>8.4f} {mb:>8.4f} "
                  f"{diff:+9.4f}  [{lo:+.4f},{hi:+.4f}] {p:>8.4f} {mcp:>8.4f}")


# ============================================================================
# Per-answer-type breakdown (ChartQA only — ChartFC is all boolean)
# ============================================================================
section("ChartQA: per-answer-type Pass@1 (HCPC vs GRPO)")
g = load(GROUPS["chartqa"]["grpo"])
h = load(GROUPS["chartqa"]["hcpc"])
idxs = sorted(g.keys())
buckets = defaultdict(list)
for i in idxs:
    t = label_type(g[i].get("label", ""))
    buckets[t].append((first_correct(g[i]), first_correct(h[i])))
print(f"  {'type':<10s} {'n':>5s} {'GRPO':>8s} {'HCPC':>8s} {'diff':>8s} {'p':>8s}")
for t in ["numeric", "boolean", "string", "list"]:
    items = buckets.get(t, [])
    if not items:
        continue
    mg = sum(int(x[0]) for x in items) / len(items)
    mh = sum(int(x[1]) for x in items) / len(items)
    only_g = sum(1 for x in items if x[0] and not x[1])
    only_h = sum(1 for x in items if x[1] and not x[0])
    p = mcnemar_exact(only_g, only_h)
    print(f"  {t:<10s} {len(items):>5d} {mg:>8.4f} {mh:>8.4f} {mh-mg:+8.4f} {p:>8.4f}")


# ============================================================================
# Format-tag breakdown — where exactly does NSR fail?
# ============================================================================
section("Per-tag format compliance (% of rollouts with each tag, ChartQA)")
data = {name: load(p) for name, p in GROUPS["chartqa"].items()}
idxs = sorted(next(iter(data.values())).keys())
tag_keys = ["has_think_tags", "has_answer_tags", "has_type_tags",
            "has_table_tags", "proper_order", "single_tags", "fully_compliant"]
print(f"  {'method':<10s} " + "  ".join(f"{k.replace('has_','').replace('_tags',''):>10s}" for k in tag_keys))
for name, d in data.items():
    pct = []
    for k in tag_keys:
        n = 0
        s = 0
        for i in idxs:
            v = d[i].get("format_compliance", {}).get(k)
            if v is not None:
                n += 1
                s += int(bool(v))
        pct.append(s / n if n else 0.0)
    print(f"  {name:<10s} " + "  ".join(f"{x:>10.3f}" for x in pct))


# ============================================================================
# Honest coherence: P(answer correct | per-rollout)
# Build a real conditional from per_sample.jsonl by computing
# P(answer_i correct | rollout_i has well-formed table) per method.
# This is the metric the paper draft prose actually describes.
# ============================================================================
section("Conditional accuracy: P(answer correct | table tag present and parsed)")
for bench, runs in GROUPS.items():
    print(f"\n  --- {bench} ---")
    data = {name: load(p) for name, p in runs.items()}
    idxs = sorted(next(iter(data.values())).keys())
    print(f"  {'method':<10s}  n_rollouts  P(table)  P(ans|table)  P(ans|~table)  lift")
    for name, d in data.items():
        n_rollouts = 0
        n_table = 0
        ans_given_table = 0
        ans_given_no_table = 0
        n_no_table = 0
        for i in idxs:
            cl = d[i].get("correct", []) or []
            parsed_first = d[i].get("parsed_first", {}) or {}
            # We only have parsed_first reliably; assume the per-rollout extras
            # follow the rollout-0 quality. As a proxy, use sample-level table parse.
            has_table = bool(parsed_first.get("table_parse_success_strict"))
            for c in cl:
                n_rollouts += 1
                if has_table:
                    n_table += 1
                    ans_given_table += int(bool(c))
                else:
                    n_no_table += 1
                    ans_given_no_table += int(bool(c))
        p_tbl = n_table / max(1, n_rollouts)
        p_at = ans_given_table / max(1, n_table)
        p_ant = ans_given_no_table / max(1, n_no_table)
        print(f"  {name:<10s}  {n_rollouts:>10d}   {p_tbl:>6.3f}    {p_at:>9.3f}     {p_ant:>9.3f}    {p_at-p_ant:+.3f}")
