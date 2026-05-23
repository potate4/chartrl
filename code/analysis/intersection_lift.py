"""Intersection Δ_tbl: condition on samples where ALL compared methods produced
a parseable table at rollout 0 (per-sample table_parse_success_strict==True).

Removes the subset confound where different methods get different denominators.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "final_outputs"

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


def has_table(r):
    p = (r.get("parsed_first", {}) or {})
    return bool(p.get("table_parse_success_strict"))


for bench, runs in GROUPS.items():
    print(f"\n========== {bench}: full-set Δ_tbl ==========")
    data = {name: load(p) for name, p in runs.items()}
    idxs = sorted(next(iter(data.values())).keys())

    print(f"  {'method':<10s}  P(table)  P(ans|tbl)  P(ans|~tbl)  Δ_tbl")
    for name, d in data.items():
        n_t = sum(1 for i in idxs if has_table(d[i]))
        n_n = sum(1 for i in idxs if not has_table(d[i]))
        ans_t = sum(1 for i in idxs if has_table(d[i]) and first_correct(d[i]))
        ans_n = sum(1 for i in idxs if not has_table(d[i]) and first_correct(d[i]))
        p_t = n_t / len(idxs)
        p_at = ans_t / max(1, n_t)
        p_an = ans_n / max(1, n_n)
        print(f"  {name:<10s}  {p_t:>7.3f}   {p_at:>8.3f}    {p_an:>8.3f}     {p_at-p_an:+.3f}")

    # intersection: samples where the named pair both produced a parseable table
    print(f"\n  ---- intersection Δ_tbl on samples where BOTH methods parsed table ----")
    names = list(data)
    base_name = "grpo" if "grpo" in names else names[0]
    for other in names:
        if other == base_name:
            continue
        intersect = [i for i in idxs if has_table(data[base_name][i]) and has_table(data[other][i])]
        non = [i for i in idxs if not has_table(data[base_name][i]) and not has_table(data[other][i])]
        if not intersect or not non:
            print(f"  {base_name} vs {other}: insufficient intersection (n_inter={len(intersect)}, n_non={len(non)})")
            continue
        # for each method, compute P(ans|table) on intersect, P(ans|no table) on non
        for m in [base_name, other]:
            p_at = sum(1 for i in intersect if first_correct(data[m][i])) / len(intersect)
            p_an = sum(1 for i in non       if first_correct(data[m][i])) / len(non)
            print(f"    {m:<10s} on intersect: P(ans|tbl)={p_at:.3f} (n={len(intersect)})  "
                  f"P(ans|~tbl)={p_an:.3f} (n={len(non)})  Δ={p_at-p_an:+.3f}")
        print()

    # Same-sample-population Δ_tbl: for each sample, compute whether the GRPO
    # answer is correct conditional on each method having a table - paired.
    print(f"\n  ---- paired-per-sample Δ_tbl: each method evaluated on common subset ----")
    common_tbl = [i for i in idxs if all(has_table(data[m][i]) for m in names)]
    common_no = [i for i in idxs if all(not has_table(data[m][i]) for m in names)]
    print(f"  n_common_table={len(common_tbl)}  n_common_no_table={len(common_no)}")
    if common_tbl and common_no:
        for m in names:
            p_at = sum(1 for i in common_tbl if first_correct(data[m][i])) / len(common_tbl)
            p_an = sum(1 for i in common_no  if first_correct(data[m][i])) / len(common_no)
            print(f"    {m:<10s}  P(ans|tbl)={p_at:.3f}  P(ans|~tbl)={p_an:.3f}  Δ_common={p_at-p_an:+.3f}")
