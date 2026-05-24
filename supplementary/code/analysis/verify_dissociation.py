"""Verify the dissociation claim.

Claim: NSR+HCPC on ChartFC reaches GRPO-level Pass@1 at near-zero format compliance.

Question: where do the correct answers come from? If most correct answers are from
formatted rollouts, then format and accuracy are still linked. If correct answers
come from both formatted AND unformatted rollouts, then they're truly dissociable.
"""
import json
from pathlib import Path

_SUPPL = Path(__file__).resolve().parents[2]
ROOT = _SUPPL / "eval_outputs"


def load(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def per_rollout_breakdown(name, path):
    data = load(path)
    # Each sample has 4 rollouts. For each rollout, check (correct, fully_formatted).
    # Sample-level format_compliance.fully_compliant: was this rollout (rollout 0) fully formatted?
    # But the diversity block and format block report rollout-0 properties.
    # The correct list has correctness for all 4 rollouts.
    # We need per-rollout format compliance. Let me check what's in the file.

    # First, what's available?
    if not data:
        print(f"  {name}: empty")
        return
    r0 = data[0]
    print(f"  {name}: keys = {list(r0.keys())}")
    fmt_keys = list((r0.get('format_compliance', {}) or {}).keys())
    print(f"    format_compliance subkeys: {fmt_keys}")
    div_keys = list((r0.get('diversity', {}) or {}).keys())
    print(f"    diversity subkeys: {div_keys}")
    print(f"    parsed_first keys: {list((r0.get('parsed_first', {}) or {}).keys())[:8]}")


print("Schema check:")
per_rollout_breakdown("nsr_hcpc_chartfc",
                     ROOT / "nsr_hcpc_chartfc" / "per_sample.jsonl")
print()

# OK now build a clean dissociation table per method.
# For each rollout in each sample: was it correct? was it well-formed (proper tags)?
# Approximation: format_compliance.fully_compliant in the JSONL is for rollout 0 only.
# To get per-rollout format, we'd need to parse raw_outputs. Let me do that.

import re
FULL_PAT = re.compile(
    r"^<think>\n<type>.*?</type>\n<table>.*?</table>.*?</think>\n<answer>.*?</answer>$",
    re.DOTALL | re.MULTILINE
)


def is_full_format(text):
    if not text:
        return False
    return bool(FULL_PAT.match(text))


def is_partial_format(text):
    """Has all required tags but may not be in strict order."""
    if not text:
        return False
    needed = ["<think>", "</think>", "<answer>", "</answer>",
              "<table>", "</table>", "<type>", "</type>"]
    return all(t in text for t in needed)


print("=" * 70)
print("Per-rollout breakdown on ChartFC: where does correctness come from?")
print("=" * 70)

CHARTFC = {
    "grpo":     ROOT / "grpo_chartfc"         / "per_sample.jsonl",
    "hcpc":     ROOT / "hcpc_chartfc"         / "per_sample.jsonl",
    "nsr":      ROOT / "nsr_baseline_chartfc" / "per_sample.jsonl",
    "nsr_hcpc": ROOT / "nsr_hcpc_chartfc"     / "per_sample.jsonl",
}

print(f"\n  {'method':<10s}  {'rollouts':>9s}  {'%fmt-full':>10s}  "
      f"{'%fmt-any':>10s}  {'P(corr|fmt)':>12s}  {'P(corr|~fmt)':>13s}")
for name, path in CHARTFC.items():
    data = load(path)
    n_roll = 0
    n_fmt = 0
    n_part = 0
    correct_when_fmt = 0
    correct_when_no_fmt = 0
    correct_when_part = 0
    correct_when_no_part = 0
    for s in data:
        raws = s.get("raw_outputs", []) or []
        corrs = s.get("correct", []) or []
        for r, c in zip(raws, corrs):
            n_roll += 1
            full = is_full_format(r)
            part = is_partial_format(r)
            if full:
                n_fmt += 1
                if c: correct_when_fmt += 1
            else:
                if c: correct_when_no_fmt += 1
            if part:
                n_part += 1
                if c: correct_when_part += 1
            else:
                if c: correct_when_no_part += 1
    pct_full = n_fmt / max(1, n_roll)
    pct_part = n_part / max(1, n_roll)
    p_c_fmt = correct_when_fmt / max(1, n_fmt) if n_fmt else 0.0
    p_c_no  = correct_when_no_fmt / max(1, n_roll - n_fmt) if (n_roll - n_fmt) else 0.0
    print(f"  {name:<10s}  {n_roll:>9d}  {pct_full*100:>9.1f}%  "
          f"{pct_part*100:>9.1f}%  {p_c_fmt:>11.3f}   {p_c_no:>11.3f}")

print()
print("Interpretation:")
print("  - %fmt-full: fraction of rollouts emitting all tags in strict order")
print("  - %fmt-any:  fraction emitting all required tags (any order)")
print("  - P(corr|fmt): P(answer correct | rollout is fully formatted)")
print("  - P(corr|~fmt): P(answer correct | rollout is NOT fully formatted)")
print()
print("If accuracy is dissociable from format, P(corr|~fmt) should be high.")
print("If accuracy depends on format, P(corr|~fmt) << P(corr|fmt).")
