"""HCPC vs GRPO head-to-head, every metric we can pull from per_sample.jsonl.

We want to be honest about where HCPC ACTUALLY differs from GRPO.
"""
import json
import math
import random
from pathlib import Path

random.seed(2026)
ROOT = Path(__file__).resolve().parents[1] / "final_outputs"

PAIRS = [
    ("chartqa", ROOT / "grpo_chartqa" / "per_sample.jsonl",
                ROOT / "hcpc_chartqa" / "per_sample.jsonl"),
    ("chartfc", ROOT / "grpo_chartfc" / "per_sample.jsonl",
                ROOT / "hcpc_chartfc" / "per_sample.jsonl"),
]


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


def correct_rate(r):
    cl = r.get("correct", []) or []
    return sum(cl) / len(cl) if cl else 0.0


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    from math import comb
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * p)


def boot_ci_paired(a, b, n_boot=10000):
    n = len(a)
    diffs = []
    a = [float(x) for x in a]
    b = [float(x) for x in b]
    for _ in range(n_boot):
        idxs = [random.randrange(n) for _ in range(n)]
        diffs.append(sum(a[i] for i in idxs) / n - sum(b[i] for i in idxs) / n)
    diffs.sort()
    return diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)]


for bench, grpo_path, hcpc_path in PAIRS:
    print(f"\n========== {bench}: GRPO vs HCPC ==========")
    g = load(grpo_path)
    h = load(hcpc_path)
    idxs = sorted(g)

    metrics = {
        "pass@1 (first correct)":  (first_correct, "binary"),
        "pass@k (any correct)":    (any_correct,   "binary"),
        "all-k correct":           (all_correct,   "binary"),
        "correct_rate (mean over k)":(correct_rate, "real"),
    }

    print(f"  n={len(idxs)}\n")
    print(f"  {'metric':<30s} {'GRPO':>8s} {'HCPC':>8s} {'diff(H-G)':>10s} {'95% CI':>20s} {'p (McNemar)':>14s}")
    print("  " + "-" * 92)
    for name, (fn, kind) in metrics.items():
        a_g = [fn(g[i]) for i in idxs]
        a_h = [fn(h[i]) for i in idxs]
        mg = sum(float(x) for x in a_g) / len(a_g)
        mh = sum(float(x) for x in a_h) / len(a_h)
        lo, hi = boot_ci_paired(a_h, a_g, n_boot=5000)
        if kind == "binary":
            only_g = sum(1 for i in range(len(idxs)) if a_g[i] and not a_h[i])
            only_h = sum(1 for i in range(len(idxs)) if a_h[i] and not a_g[i])
            p = mcnemar_exact(only_g, only_h)
            psig = f"{p:.3g}"
        else:
            psig = "    n/a"
        print(f"  {name:<30s} {mg:>8.4f} {mh:>8.4f} {(mh-mg):+10.4f} [{lo:+.4f},{hi:+.4f}]  {psig:>14s}")

    # diversity / format metrics
    extras = ["format_compliance.fully_compliant",
              "parsed_first.parse_success",
              "parsed_first.table_parse_success_strict",
              "diversity.c_table",
              "diversity.d_reason",
              "diversity.coherence",
              "diversity.correct_rate"]
    print()
    print(f"  {'aux metric':<40s} {'GRPO':>8s} {'HCPC':>8s} {'diff':>8s}")
    print("  " + "-" * 70)
    for path in extras:
        keys = path.split(".")
        def get(rec):
            v = rec
            for k in keys:
                v = (v or {}).get(k) if isinstance(v, dict) else None
            if isinstance(v, bool):
                return 1.0 if v else 0.0
            return float(v) if v is not None else None
        gv = [get(g[i]) for i in idxs]
        hv = [get(h[i]) for i in idxs]
        gv = [x for x in gv if x is not None]
        hv = [x for x in hv if x is not None]
        if not gv or not hv:
            continue
        mg = sum(gv) / len(gv)
        mh = sum(hv) / len(hv)
        print(f"  {path:<40s} {mg:>8.4f} {mh:>8.4f} {(mh-mg):+8.4f}")

    # by-label-type for ChartQA
    if bench == "chartqa":
        import re
        BOOL = {"yes","no","true","false","correct","wrong","incorrect","right","yeah","yep","nope","y","n","1","0"}
        def ltype(label):
            s = str(label).strip()
            if s.lower() in BOOL: return "boolean"
            if (s.startswith("[") and s.endswith("]")) or " and " in s.lower():
                inner = s.strip("[]")
                parts = re.split(r"[,]| and ", inner)
                if len([p for p in parts if p.strip()]) >= 2: return "list"
            cleaned = re.sub(r"[\$,%]", "", s.lower())
            cleaned = re.sub(r"\s*(million|billion|thousand|k|m|b)\s*$", "", cleaned).strip()
            try:
                float(cleaned); return "numeric"
            except ValueError:
                return "string"

        print(f"\n  Per-label-type pass@1 (GRPO -> HCPC):")
        from collections import defaultdict
        buckets = defaultdict(list)
        for i in idxs:
            t = ltype(g[i].get("label",""))
            buckets[t].append((first_correct(g[i]), first_correct(h[i])))
        for t in ["numeric","boolean","string","list"]:
            items = buckets.get(t, [])
            if not items: continue
            mg = sum(int(x[0]) for x in items)/len(items)
            mh = sum(int(x[1]) for x in items)/len(items)
            n = len(items)
            only_g = sum(1 for x in items if x[0] and not x[1])
            only_h = sum(1 for x in items if x[1] and not x[0])
            p = mcnemar_exact(only_g, only_h)
            print(f"    {t:8s}  n={n:4d}  GRPO={mg:.4f}  HCPC={mh:.4f}  diff={mh-mg:+.4f}  p={p:.3g}")
