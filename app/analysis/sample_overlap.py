"""Are the runs comparing the same 500 samples? If not, headline accuracy
comparisons are bogus.

We use (question, label) tuples as a sample fingerprint.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0] / "final_outputs"

GROUPS = {
    "chartqa": {
        "base_chartqa":  ROOT / "base_chartqa"  / "per_sample.jsonl",
        "grpo_chartqa":  ROOT / "grpo_chartqa"  / "per_sample.jsonl",
        "hcpc_chartqa":  ROOT / "hcpc_chartqa"  / "per_sample.jsonl",
        "nsr_chartqa":   ROOT / "nsr_chartqa"   / "per_sample.jsonl",
    },
    "chartfc": {
        "grpo_chartfc":  ROOT / "grpo_chartfc"  / "per_sample.jsonl",
        "hcpc_chartfc":  ROOT / "hcpc_chartfc"  / "per_sample.jsonl",
        "nsr_chartfc":   ROOT / "nsr_chartfc"   / "per_sample (2).jsonl",
    },
}


def fingerprints(path):
    fps = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            fps.append((r.get("question", ""), str(r.get("label", ""))))
    return fps


def by_idx(path):
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["idx"]] = (r.get("question", ""), str(r.get("label", "")))
    return out


for group, runs in GROUPS.items():
    print(f"\n=== {group} ===")
    fps_by_run = {name: fingerprints(p) for name, p in runs.items()}
    sets = {name: set(fps) for name, fps in fps_by_run.items()}

    common = set.intersection(*sets.values())
    print(f"  shared samples: {len(common)}")
    for name, s in sets.items():
        only = s - common
        print(f"    {name}: total={len(s)}, only-here={len(only)}")

    # also check ordering â€” are idx 0..499 the same questions?
    idx_maps = {name: by_idx(p) for name, p in runs.items()}
    if len(idx_maps) > 1:
        names = list(idx_maps)
        first = idx_maps[names[0]]
        same_order = True
        for name in names[1:]:
            for idx, fp in first.items():
                if idx_maps[name].get(idx) != fp:
                    same_order = False
                    break
            if not same_order:
                break
        print(f"  same idx ordering across runs: {same_order}")
