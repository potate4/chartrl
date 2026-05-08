"""Sanity-check the structure and recompute headline numbers across all 7 runs.

We trust ONLY per_sample.jsonl. summary.json is treated as suspect.
"""
import json
import os
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "final_outputs"

RUNS = {
    "base_chartqa":  ROOT / "base_chartqa"  / "per_sample.jsonl",
    "grpo_chartqa":  ROOT / "grpo_chartqa"  / "per_sample.jsonl",
    "hcpc_chartqa":  ROOT / "hcpc_chartqa"  / "per_sample.jsonl",
    "nsr_chartqa":   ROOT / "nsr_chartqa"   / "per_sample.jsonl",
    "grpo_chartfc":  ROOT / "grpo_chartfc"  / "per_sample.jsonl",
    "hcpc_chartfc":  ROOT / "hcpc_chartfc"  / "per_sample.jsonl",
    "nsr_chartfc":   ROOT / "nsr_chartfc"   / "per_sample (2).jsonl",
}


# --- label classification (mirror app/rewards/base_rewards.py logic) -------

BOOL_TRUE  = {"yes", "true", "correct", "right", "yeah", "yep", "y", "1"}
BOOL_FALSE = {"no", "false", "wrong", "incorrect", "nope", "n", "0"}

def classify_label(label: str) -> str:
    if label is None:
        return "missing"
    s = str(label).strip()
    if not s:
        return "missing"
    low = s.lower()
    if low in BOOL_TRUE or low in BOOL_FALSE:
        return "boolean"
    # list-ish
    if (s.startswith("[") and s.endswith("]")) or (" and " in low) or ("," in s and any(c.isdigit() for c in s)):
        # be conservative - check it has multi-element feel
        inner = s.strip("[]")
        parts = re.split(r"[,]| and ", inner)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) >= 2:
            return "list"
    # numeric (allow $, %, comma, million/billion)
    cleaned = re.sub(r"[\$,%]", "", low)
    cleaned = re.sub(r"\s*(million|billion|thousand|k|m|b)\s*$", "", cleaned).strip()
    try:
        float(cleaned)
        return "numeric"
    except ValueError:
        pass
    return "string"


def best_of_k(correct_list):
    """pass@k where k = len(correct_list). True if any prediction was correct."""
    return any(correct_list) if correct_list else False


def first_correct(correct_list):
    return bool(correct_list[0]) if correct_list else False


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


# --- per-run scan --------------------------------------------------------

def load(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def analyze(name, rows):
    n = len(rows)
    by_type = defaultdict(list)         # label_type -> list of dicts {first_correct, any_correct, all_correct}
    fmt_compl = []
    table_parse_strict = []
    parsed_first = []
    times = []
    n_preds_per = Counter()
    diversity_c_table = []
    diversity_d_reason = []
    diversity_coherence = []

    # exact-match (relaxed_accuracy) at first-pred and any-pred level
    relaxed_first = []
    relaxed_any = []

    for r in rows:
        label = r.get("label")
        ltype = classify_label(label)
        cl = r.get("correct", []) or []
        n_preds_per[len(cl)] += 1

        first = first_correct(cl)
        anyc  = best_of_k(cl)
        allc  = all(cl) if cl else False
        by_type[ltype].append({"first": first, "any": anyc, "all": allc, "label": label, "preds": r.get("predictions", []), "rew": r.get("reward_accuracy")})

        relaxed_first.append(first)
        relaxed_any.append(anyc)

        fc = r.get("format_compliance", {}) or {}
        fmt_compl.append(bool(fc.get("fully_compliant")))

        pf = r.get("parsed_first", {}) or {}
        parsed_first.append(bool(pf.get("parse_success")))
        table_parse_strict.append(bool(pf.get("table_parse_success_strict")))

        div = r.get("diversity", {}) or {}
        diversity_c_table.append(div.get("c_table"))
        diversity_d_reason.append(div.get("d_reason"))
        diversity_coherence.append(div.get("coherence"))

        t = r.get("time_seconds")
        if t is not None:
            times.append(t)

    # overall pass@1 (mean of first_correct), pass@k (any) — k varies
    pass1 = mean(relaxed_first)
    passk = mean(relaxed_any)

    print(f"\n=== {name}  (n={n}, k={dict(n_preds_per)}) ===")
    print(f"  pass@1 (first pred correct)      : {pass1:.4f}")
    print(f"  pass@k (any of k correct)        : {passk:.4f}")
    print(f"  format_compliance (fully)        : {mean([1 if x else 0 for x in fmt_compl]):.4f}")
    print(f"  parsed_first.parse_success       : {mean([1 if x else 0 for x in parsed_first]):.4f}")
    print(f"  parsed_first.table_parse_strict  : {mean([1 if x else 0 for x in table_parse_strict]):.4f}")
    print(f"  diversity.c_table  (mean)        : {mean(diversity_c_table):.4f}")
    print(f"  diversity.d_reason (mean)        : {mean(diversity_d_reason):.4f}")
    print(f"  diversity.coherence (mean)       : {mean(diversity_coherence):.4f}")
    if times:
        print(f"  time_seconds  mean / median      : {mean(times):.2f} / {statistics.median(times):.2f}")

    # per-type
    print(f"  --- by label type (count, pass@1, pass@k) ---")
    for ltype in ["boolean", "numeric", "list", "string", "missing"]:
        items = by_type.get(ltype, [])
        if not items:
            continue
        p1 = mean([1 if x["first"] else 0 for x in items])
        pk = mean([1 if x["any"]   else 0 for x in items])
        print(f"    {ltype:8s}  n={len(items):4d}  pass@1={p1:.4f}  pass@k={pk:.4f}")

    return {
        "name": name,
        "n": n,
        "pass1": pass1,
        "passk": passk,
        "by_type": {
            t: {
                "n": len(items),
                "pass1": mean([1 if x["first"] else 0 for x in items]),
                "passk": mean([1 if x["any"]   else 0 for x in items]),
            }
            for t, items in by_type.items()
        },
        "fmt_compliance": mean([1 if x else 0 for x in fmt_compl]),
        "parse_success": mean([1 if x else 0 for x in parsed_first]),
        "table_parse_strict": mean([1 if x else 0 for x in table_parse_strict]),
        "div_c_table": mean(diversity_c_table),
        "div_d_reason": mean(diversity_d_reason),
        "div_coherence": mean(diversity_coherence),
        "n_preds_distribution": dict(n_preds_per),
    }


def main():
    results = {}
    for name, path in RUNS.items():
        rows = load(path)
        results[name] = analyze(name, rows)

    out = Path(__file__).parent / "metrics_recomputed.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
