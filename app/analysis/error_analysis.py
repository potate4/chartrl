"""Where are the gains coming from?

For each method, find:
- samples it gets right that the others don't (its 'unique wins')
- samples everyone gets wrong (hard set)
- typical errors

We focus on first-prediction-correct (pass@1) since that's the practical metric.
"""
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1] / "final_outputs"

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


def load_full(path):
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["idx"]] = r
    return out


def first_correct(rec):
    cl = rec.get("correct", []) or []
    return bool(cl[0]) if cl else False


for group, runs in GROUPS.items():
    print(f"\n========== {group} ==========")
    full = {name: load_full(p) for name, p in runs.items()}
    names = list(full)
    indices = list(full[names[0]])

    # uniqueness counts - which method uniquely solves a sample
    unique_wins = {n: 0 for n in names}
    everyone_right = 0
    everyone_wrong = 0
    rl_unique_wins = {}  # for chartqa: which RL methods solve when base fails

    for idx in indices:
        flags = {n: first_correct(full[n][idx]) for n in names}
        n_right = sum(flags.values())

        if n_right == len(names):
            everyone_right += 1
        elif n_right == 0:
            everyone_wrong += 1
        elif n_right == 1:
            for n, f in flags.items():
                if f:
                    unique_wins[n] += 1

    print(f"  Total samples: {len(indices)}")
    print(f"  Everyone correct (first pred): {everyone_right}")
    print(f"  Everyone wrong (first pred):   {everyone_wrong}")
    print(f"  Unique-win counts (only this method got first pred right):")
    for n in names:
        print(f"    {n}: {unique_wins[n]}")

    # ChartQA: methods that base fails on
    if "base" in full:
        rl_methods = [n for n in names if n != "base"]
        base_fails = [idx for idx in indices if not first_correct(full["base"][idx])]
        print(f"\n  Of {len(base_fails)} samples base fails on:")
        for n in rl_methods:
            saves = sum(1 for idx in base_fails if first_correct(full[n][idx]))
            print(f"    {n} fixes: {saves} ({saves/len(base_fails)*100:.1f}%)")

        # Samples where base wins but RL all lose (regression)
        rl_all_fail = [idx for idx in indices
                       if first_correct(full["base"][idx])
                       and all(not first_correct(full[n][idx]) for n in rl_methods)]
        print(f"  Samples base solves but ALL RL methods fail (RL regressions): {len(rl_all_fail)}")

    # show 5 unique-win examples for top method
    top = max(unique_wins, key=unique_wins.get)
    print(f"\n  Sample of {top}'s unique wins (first 3):")
    shown = 0
    for idx in indices:
        flags = {n: first_correct(full[n][idx]) for n in names}
        if flags[top] and sum(flags.values()) == 1:
            r = full[top][idx]
            preds_others = {n: full[n][idx]["predictions"][0] for n in names if n != top}
            print(f"    [idx {idx}] Q: {r['question'][:80]!r}")
            print(f"      label    = {r['label']!r}")
            print(f"      {top}     = {r['predictions'][0]!r}")
            for n, p in preds_others.items():
                print(f"      {n:6s}   = {p!r}")
            shown += 1
            if shown >= 3:
                break
