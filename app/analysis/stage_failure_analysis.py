"""Stage-by-stage failure analysis across all eval runs.

Inputs: per_sample.jsonl files in app/final_outputs/. We trust ONLY the
raw_outputs[] and correct[] arrays per sample. Everything stage-related
is recomputed by re-parsing raw_outputs.

Outputs:
- Per-run joint distribution over (format_ok, type_ok, table_ok, answer_ok)
  at the rollout level (4 rollouts per sample, ~2000 rollouts per run).
- Lucky-correct rate: P(answer_ok | NOT table_ok).
- Conditional accuracies: P(answer_ok | table_ok), P(answer_ok | NOT table_ok).
- Provenance against predicted table: does the answer trace back to a number
  the model wrote in <table>?
- Reward-variant scoring: how would SCR (hard, soft, floor, discount) reshape
  the per-rollout reward signal?
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[0] / "final_outputs"

RUNS = {
    "base_chartqa":  ROOT / "base_chartqa"  / "per_sample.jsonl",
    "grpo_chartqa":  ROOT / "grpo_chartqa"  / "per_sample.jsonl",
    "hcpc_chartqa":  ROOT / "hcpc_chartqa"  / "per_sample.jsonl",
    "nsr_chartqa":   ROOT / "nsr_chartqa"   / "per_sample.jsonl",
    "grpo_chartfc":  ROOT / "grpo_chartfc"  / "per_sample.jsonl",
    "hcpc_chartfc":  ROOT / "hcpc_chartfc"  / "per_sample.jsonl",
    "nsr_chartfc":   ROOT / "nsr_chartfc"   / "per_sample (2).jsonl",
}

NUM_RE = re.compile(r"-?\d+(?:[.,]\d+)?(?:[eE][+-]?\d+)?")


def parse_numeric(s: str) -> Optional[float]:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    s = re.sub(r"[\$,]", "", s)
    s = re.sub(r"%\s*$", "", s)
    s = re.sub(r"\s*(million|billion|thousand)\s*$", "", s, flags=re.IGNORECASE)
    try:
        return float(s)
    except ValueError:
        return None


# ----- per-rollout parser ----------------------------------------------------

def extract_block(text: str, open_tag: str, close_tag: str) -> str:
    if open_tag not in text or close_tag not in text:
        return ""
    try:
        return text.split(open_tag, 1)[1].split(close_tag, 1)[0]
    except Exception:
        return ""


def has_format(raw: str) -> bool:
    """Full chart-RVR format compliance."""
    pat = re.compile(
        r"<think>\s*<type>.*?</type>\s*<table>.*?</table>.*?</think>\s*<answer>.*?</answer>",
        re.DOTALL,
    )
    return bool(pat.search(raw))


def parse_type(raw: str) -> Optional[str]:
    blk = extract_block(raw, "<type>", "</type>").strip()
    if not blk:
        return None
    blk = blk.lower()
    for cand in ["bar", "line", "pie", "scatter", "area", "histogram", "donut"]:
        if cand in blk:
            return cand
    return blk[:32]


def parse_table_strict(raw: str) -> Optional[Dict[str, Any]]:
    blk = extract_block(raw, "<table>", "</table>").strip()
    if not blk:
        return None
    if "```json" in blk:
        blk = blk.split("```json", 1)[1].split("```", 1)[0].strip()
    blk = blk.strip()
    try:
        obj = json.loads(blk)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    if "columns" not in obj or "rows" not in obj:
        return None
    return obj


def table_cells_numeric(tab: Dict[str, Any]) -> List[float]:
    nums: List[float] = []
    if not isinstance(tab, dict):
        return nums
    for row in tab.get("rows", []) or []:
        if isinstance(row, list):
            for cell in row:
                v = parse_numeric(str(cell))
                if v is not None:
                    nums.append(v)
        else:
            v = parse_numeric(str(row))
            if v is not None:
                nums.append(v)
    return nums


def parse_answer(raw: str) -> str:
    blk = extract_block(raw, "<answer>", "</answer>").strip()
    return blk


def stages(raw: str) -> Dict[str, Any]:
    """Return structural-stage flags for a single rollout."""
    fmt = has_format(raw)
    typ = parse_type(raw)
    tab = parse_table_strict(raw)
    ans = parse_answer(raw)
    return {
        "format_ok": fmt,
        "type_ok": typ is not None and len(typ) > 0,
        "table_ok": tab is not None,
        "answer_present": bool(ans),
        "predicted_type": typ,
        "predicted_table": tab,
        "predicted_answer": ans,
    }


# ----- provenance: answer-in-predicted-table --------------------------------

def closure_2(base: List[float], cap: int = 4096) -> set:
    seen = set(round(x, 6) for x in base)
    frontier = list({round(x, 6): x for x in base}.values())
    for _ in range(2):
        new = []
        for a in frontier:
            for b in frontier:
                for op_name, op in (("+", lambda x,y: x+y),
                                    ("-", lambda x,y: x-y),
                                    ("*", lambda x,y: x*y)):
                    try:
                        v = op(a, b)
                    except Exception:
                        continue
                    k = round(v, 6)
                    if k in seen: continue
                    seen.add(k); new.append(v)
                    if len(seen) >= cap: return seen
                if abs(b) > 1e-9:
                    try:
                        v = a / b; k = round(v, 6)
                        if k not in seen:
                            seen.add(k); new.append(v)
                            if len(seen) >= cap: return seen
                    except Exception:
                        pass
        # reductions
        if frontier:
            try:
                for v in (sum(frontier), sum(frontier)/len(frontier),
                          max(frontier), min(frontier), float(len(frontier))):
                    k = round(v, 6)
                    if k not in seen: seen.add(k); new.append(v)
            except Exception:
                pass
        if not new: break
        frontier = new
    return seen


def answer_traces_to_table(answer: str, table: Optional[Dict[str, Any]],
                           tolerance: float = 0.01) -> Optional[bool]:
    """Returns True if answer is numeric and matches a cell or a depth-2
    arithmetic combination of cells. Returns None if the question is not
    numeric (we treat string/boolean answers as out-of-scope here)."""
    a = parse_numeric(answer)
    if a is None:
        return None
    if not table:
        return False
    cells = table_cells_numeric(table)
    if not cells:
        return False
    cl = closure_2(cells)
    if abs(a) < 1e-9:
        return any(abs(c) < 1e-9 for c in cl)
    for c in cl:
        denom = max(abs(c), abs(a), 1e-6)
        if abs(c - a) / denom <= tolerance:
            return True
    return False


# ----- main analysis --------------------------------------------------------

def analyze_run(name: str, path: Path) -> Dict[str, Any]:
    print(f"\n========== {name} ==========")
    rollouts = []  # list of dicts per rollout
    n_samples = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            n_samples += 1
            correct_list = r.get("correct", []) or []
            raw_list = r.get("raw_outputs", []) or []
            for k in range(min(len(raw_list), len(correct_list))):
                st = stages(raw_list[k])
                trace = answer_traces_to_table(
                    st["predicted_answer"], st["predicted_table"]
                )
                rollouts.append({
                    "idx": r["idx"],
                    "k": k,
                    "format_ok": st["format_ok"],
                    "type_ok": st["type_ok"],
                    "table_ok": st["table_ok"],
                    "answer_present": st["answer_present"],
                    "answer_ok": bool(correct_list[k]),
                    "trace": trace,  # True / False / None (not numeric)
                    "label": r.get("label", ""),
                })

    n_roll = len(rollouts)
    if n_roll == 0:
        return {}

    fmt_rate = sum(1 for x in rollouts if x["format_ok"]) / n_roll
    type_rate = sum(1 for x in rollouts if x["type_ok"]) / n_roll
    table_rate = sum(1 for x in rollouts if x["table_ok"]) / n_roll
    answer_rate = sum(1 for x in rollouts if x["answer_ok"]) / n_roll

    print(f"  rollouts: {n_roll}  ({n_samples} samples x {n_roll // n_samples})")
    print(f"  format_ok  : {fmt_rate:.4f}")
    print(f"  type_ok    : {type_rate:.4f}")
    print(f"  table_ok   : {table_rate:.4f}")
    print(f"  answer_ok  : {answer_rate:.4f}")

    # joint table_ok x answer_ok
    p_tt = sum(1 for x in rollouts if x["table_ok"] and x["answer_ok"]) / n_roll
    p_tf = sum(1 for x in rollouts if x["table_ok"] and not x["answer_ok"]) / n_roll
    p_ft = sum(1 for x in rollouts if not x["table_ok"] and x["answer_ok"]) / n_roll
    p_ff = sum(1 for x in rollouts if not x["table_ok"] and not x["answer_ok"]) / n_roll
    print(f"\n  table_ok x answer_ok joint:")
    print(f"    table=T, answer=T : {p_tt:.4f}  (good pipeline, correct)")
    print(f"    table=T, answer=F : {p_tf:.4f}  (good extraction, bad reasoning)")
    print(f"    table=F, answer=T : {p_ft:.4f}  (LUCKY-CORRECT)")
    print(f"    table=F, answer=F : {p_ff:.4f}  (broken everything)")

    # conditional accuracies
    n_table_ok = sum(1 for x in rollouts if x["table_ok"])
    n_table_ng = n_roll - n_table_ok
    cond_t = (sum(1 for x in rollouts if x["table_ok"] and x["answer_ok"]) / n_table_ok) if n_table_ok else 0.0
    cond_f = (sum(1 for x in rollouts if not x["table_ok"] and x["answer_ok"]) / n_table_ng) if n_table_ng else 0.0
    print(f"\n  P(answer_ok | table_ok)     = {cond_t:.4f}  (n={n_table_ok})")
    print(f"  P(answer_ok | NOT table_ok) = {cond_f:.4f}  (n={n_table_ng})")
    print(f"  acc lift from having a table: {cond_t - cond_f:+.4f}")

    # provenance: numeric-answer rollouts only
    numeric = [x for x in rollouts if x["trace"] is not None]
    n_num = len(numeric)
    if n_num:
        traced = sum(1 for x in numeric if x["trace"]) / n_num
        # split by answer correctness
        tr_corr = [x for x in numeric if x["answer_ok"]]
        tr_wrong = [x for x in numeric if not x["answer_ok"]]
        rate_tr_corr = (sum(1 for x in tr_corr if x["trace"]) / len(tr_corr)) if tr_corr else 0.0
        rate_tr_wrong = (sum(1 for x in tr_wrong if x["trace"]) / len(tr_wrong)) if tr_wrong else 0.0
        print(f"\n  Numeric-answer rollouts: {n_num} of {n_roll} ({n_num/n_roll:.1%})")
        print(f"  P(answer traces to predicted table)            : {traced:.4f}")
        print(f"    | answer_correct=T : {rate_tr_corr:.4f}  (n={len(tr_corr)})")
        print(f"    | answer_correct=F : {rate_tr_wrong:.4f}  (n={len(tr_wrong)})")

    return {
        "name": name,
        "n_rollouts": n_roll,
        "format_rate": fmt_rate,
        "type_rate": type_rate,
        "table_rate": table_rate,
        "answer_rate": answer_rate,
        "joint": {"TT": p_tt, "TF": p_tf, "FT_lucky": p_ft, "FF": p_ff},
        "cond_acc_table_ok": cond_t,
        "cond_acc_table_bad": cond_f,
        "lift_from_table": cond_t - cond_f,
        "rollouts": rollouts,
    }


def main():
    results = {name: analyze_run(name, p) for name, p in RUNS.items()}
    out = Path(__file__).parent / "stage_failures.json"
    save = {k: {kk: vv for kk, vv in v.items() if kk != "rollouts"} for k, v in results.items() if v}
    out.write_text(json.dumps(save, indent=2))
    print(f"\nSaved aggregates to {out}")
    return results


if __name__ == "__main__":
    main()
