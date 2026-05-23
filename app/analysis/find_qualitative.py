"""Find a qualitative sample where:
- GRPO produces well-formed output with table + reasoning
- NSR fails format (no <table> tag in rollout 0)
- NSR+HCPC partially recovers (has <table> but maybe not <think>)
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "final_outputs"


def load(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["idx"]] = r
    return out


def has(text, tag):
    return f"<{tag}>" in text and f"</{tag}>" in text


grpo = load(ROOT / "grpo_chartqa" / "per_sample.jsonl")
nsr  = load(ROOT / "nsr_baseline_chartqa" / "per_sample.jsonl")
nh   = load(ROOT / "nsr_hcpc_chartqa" / "per_sample.jsonl")

# We want all 3 to have the same idx
common = set(grpo) & set(nsr) & set(nh)
print(f"common idx count: {len(common)}")

# Find candidates
candidates = []
for i in sorted(common):
    g_out = grpo[i]["raw_outputs"][0] if grpo[i]["raw_outputs"] else ""
    n_out = nsr[i]["raw_outputs"][0] if nsr[i]["raw_outputs"] else ""
    h_out = nh[i]["raw_outputs"][0] if nh[i]["raw_outputs"] else ""
    g_correct = bool(grpo[i]["correct"][0]) if grpo[i]["correct"] else False
    n_correct = bool(nsr[i]["correct"][0])  if nsr[i]["correct"]  else False
    h_correct = bool(nh[i]["correct"][0])   if nh[i]["correct"]   else False

    g_format = has(g_out, "think") and has(g_out, "table") and has(g_out, "answer")
    n_format = has(n_out, "think") and has(n_out, "table") and has(n_out, "answer")
    h_format = has(h_out, "think") and has(h_out, "table") and has(h_out, "answer")

    g_has_table = has(g_out, "table")
    n_has_table = has(n_out, "table")
    h_has_table = has(h_out, "table")

    # Ideal: GRPO formatted + correct; NSR unformatted; HCPC recovers table
    if g_format and g_correct and not n_has_table and h_has_table:
        candidates.append((i, len(g_out), len(n_out), len(h_out)))

print(f"\nFound {len(candidates)} candidates where:")
print("  GRPO full format + correct, NSR no <table>, NSR+HCPC has <table>")
print("\nTop 5 shortest (for paper):")
for i, lg, ln, lh in sorted(candidates, key=lambda x: x[1]+x[2]+x[3])[:5]:
    print(f"  idx={i}  GRPO_len={lg}  NSR_len={ln}  NSR+HCPC_len={lh}")
    print(f"    Q: {grpo[i]['question']}")
    print(f"    A_gt: {grpo[i]['label']}")
