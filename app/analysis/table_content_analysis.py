"""Compare predicted-table CONTENT against GT-table CONTENT, mined from
training logs. We trust the training logs as ground truth pairings; they
contain gt_table on every step.

For each rollout we recover:
  - gt_label, gt_table
  - predicted table (parsed from <table>...</table>)
  - predicted answer (parsed from <answer>...</answer>)
  - per-component rewards (from the rewards: line)

We then split the joint into:
  numeric_recall_high (>=0.8) + answer_correct
  numeric_recall_high (>=0.8) + answer_wrong   <- "good extraction, bad reasoning"
  numeric_recall_low  (<0.5)  + answer_correct  <- lucky-correct on hallucinated table
  numeric_recall_low  (<0.5)  + answer_wrong   <- broken everywhere
  middle band (0.5-0.8)        ...

We also report the correlation between content-correctness and the current
table_reward score, to see how well the existing reward signals content quality.
"""
from __future__ import annotations

import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

LOGS = {
    "grpo_baseline": Path(r"app/outputs/grpo_baseline/run_20260218_125045/train.log"),
    "grpo_hcpc":     Path(r"app/outputs/grpo_hcpc/run_20260219_191830/train.log"),
}

NUM_RE = re.compile(r"-?\d+(?:[.,]\d+)?(?:[eE][+-]?\d+)?")


def parse_numeric(s: Any) -> Optional[float]:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    s = re.sub(r"[\$,]", "", s)
    s = re.sub(r"%\s*$", "", s)
    s = re.sub(r"\s*(million|billion|thousand)\s*$", "", s, flags=re.I)
    try:
        return float(s)
    except ValueError:
        return None


def table_numerics(tab: Any) -> List[float]:
    if not isinstance(tab, dict):
        return []
    out: List[float] = []
    for row in tab.get("rows", []) or []:
        if isinstance(row, list):
            for cell in row:
                v = parse_numeric(cell)
                if v is not None:
                    out.append(v)
        else:
            v = parse_numeric(row)
            if v is not None:
                out.append(v)
    return out


def numeric_recall(pred_tab: Any, gt_tab: Any, tol: float = 0.01) -> Optional[float]:
    """Fraction of GT numeric cells that appear (within tolerance) in predicted
    table. None if GT has no numeric cells."""
    gt_nums = table_numerics(gt_tab)
    if not gt_nums:
        return None
    pred_nums = table_numerics(pred_tab)
    if not pred_nums:
        return 0.0
    hits = 0
    for g in gt_nums:
        denom = max(abs(g), 1e-6)
        if any(abs(p - g) / denom <= tol for p in pred_nums):
            hits += 1
    return hits / len(gt_nums)


def numeric_precision(pred_tab: Any, gt_tab: Any, tol: float = 0.01) -> Optional[float]:
    """Fraction of predicted numeric cells that are present in GT (within tol).
    Catches hallucinated extra numbers."""
    pred_nums = table_numerics(pred_tab)
    if not pred_nums:
        return None
    gt_nums = table_numerics(gt_tab)
    if not gt_nums:
        return 0.0
    hits = 0
    for p in pred_nums:
        denom = max(abs(p), 1e-6)
        if any(abs(g - p) / denom <= tol for g in gt_nums):
            hits += 1
    return hits / len(pred_nums)


def parse_table_from_completion(completion: str) -> Optional[dict]:
    if "<table>" not in completion or "</table>" not in completion:
        return None
    blk = completion.split("<table>", 1)[1].split("</table>", 1)[0].strip()
    if "```json" in blk:
        blk = blk.split("```json", 1)[1].split("```", 1)[0].strip()
    blk = blk.strip()
    try:
        obj = json.loads(blk)
    except Exception:
        return None
    if isinstance(obj, dict) and "columns" in obj and "rows" in obj:
        return obj
    return None


def parse_answer_from_completion(completion: str) -> str:
    if "<answer>" not in completion or "</answer>" not in completion:
        return ""
    return completion.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()


def answers_match(pred: str, label: str) -> bool:
    """Mirror evaluation/metrics.py relaxed_accuracy."""
    if not label:
        return False
    label_num = parse_numeric(label)
    pred_for_num = pred
    if label_num is not None:
        s = pred.replace(",", "").replace("$", "")
        if "%" not in str(label):
            s = s.replace("%", "")
        pred_for_num = s
    pred_num = parse_numeric(pred_for_num)
    if pred_num is not None and label_num is not None:
        if re.fullmatch(r"-?\d+", str(label).strip()):
            return pred_num == label_num
        if label_num != 0:
            return abs(pred_num - label_num) / abs(label_num) <= 0.05
        return abs(pred_num - label_num) <= 0.05
    return pred.strip().lower() == str(label).strip().lower()


# Parse the train.log records. Each completion_log block is of the form:
#   [completion_log step=N idx=I]
#   prompt: ...
#   gt_label: ...
#   gt_chart_type: ...
#   gt_table: {...as python repr...}
#   gt_reasoning: ...
#
#   completion: <think>...</think><answer>...</answer>
#   rewards: base_format=..., base_accuracy=..., ..., base_total=..., hcpc=..., ...

import ast


_HDR = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \| ", re.MULTILINE)


def parse_log(path: Path) -> Iterable[Dict[str, Any]]:
    """Yield one dict per completion_log entry."""
    text = path.read_text(encoding="utf-8", errors="replace")
    # Find all completion_log markers and slice between them.
    starts = [(m.start(), int(m.group(1)), int(m.group(2)))
              for m in re.finditer(r"\[completion_log step=(\d+) idx=(\d+)\]", text)]
    starts.append((len(text), -1, -1))

    for i in range(len(starts) - 1):
        a, step, idx = starts[i]
        b, _, _ = starts[i + 1]
        body = text[a:b]

        gt_table = None
        m_t = re.search(r"^gt_table:\s*(.*?)$\ngt_reasoning:", body, re.MULTILINE | re.DOTALL)
        if m_t:
            try:
                gt_table = ast.literal_eval(m_t.group(1).strip())
            except Exception:
                gt_table = None
        m_l = re.search(r"^gt_label:\s*(.*?)$", body, re.MULTILINE)
        gt_label = m_l.group(1).strip() if m_l else ""

        # Completion runs from the line after "completion: " until the
        # "rewards: ..." line. We don't strip log-prefix lines yet because
        # the model outputs are clean (no log prefixes inside them).
        m_c = re.search(r"^completion:\s*(.*?)(?=^rewards:)", body, re.MULTILINE | re.DOTALL)
        completion = m_c.group(1).strip() if m_c else ""

        rewards: Dict[str, float] = {}
        m_r = re.search(r"^rewards:\s*(.+?)$", body, re.MULTILINE)
        if m_r:
            for kv in m_r.group(1).split(","):
                kv = kv.strip()
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    try:
                        rewards[k.strip()] = float(v.strip())
                    except Exception:
                        pass

        yield {
            "step": step,
            "idx": idx,
            "gt_label": gt_label,
            "gt_table": gt_table,
            "completion": completion,
            "rewards": rewards,
        }


def analyze(name: str, log: Path, max_records: int = 8000) -> None:
    print(f"\n========== {name}  ({log.name}) ==========")
    rows: List[Dict[str, Any]] = []
    for i, rec in enumerate(parse_log(log)):
        if i >= max_records:
            break
        if rec["gt_table"] is None or not rec["completion"]:
            continue
        pred_tab = parse_table_from_completion(rec["completion"])
        pred_ans = parse_answer_from_completion(rec["completion"])
        nrec = numeric_recall(pred_tab, rec["gt_table"])
        nprec = numeric_precision(pred_tab, rec["gt_table"])
        ans_ok = answers_match(pred_ans, rec["gt_label"])
        rows.append({
            "step": rec["step"],
            "idx": rec["idx"],
            "table_struct_ok": pred_tab is not None,
            "n_recall": nrec,         # may be None when GT has no numbers
            "n_precision": nprec,     # may be None
            "answer_ok": ans_ok,
            "table_reward": rec["rewards"].get("base_table"),
            "accuracy_reward": rec["rewards"].get("base_accuracy"),
        })

    n = len(rows)
    print(f"  parsed completion_log entries: {n}")
    if not n:
        return

    # Restrict to rows where GT has numeric cells (so recall is defined).
    rs = [r for r in rows if r["n_recall"] is not None]
    print(f"  rows with numeric GT table   : {len(rs)}")
    print(f"  rows where pred table parsed : {sum(1 for r in rs if r['table_struct_ok'])}")

    # Bucket by recall band.
    def bucket(r):
        nr = r["n_recall"]
        if nr is None: return "no-gt-num"
        if nr >= 0.95: return "perfect"
        if nr >= 0.80: return "high"
        if nr >= 0.50: return "mid"
        if nr >= 0.20: return "low"
        return "near-zero"

    bcounts = Counter(bucket(r) for r in rs)
    bcorrect = defaultdict(list)
    for r in rs:
        bcorrect[bucket(r)].append(r["answer_ok"])

    print(f"\n  Numeric-recall band x answer correctness:")
    print(f"    {'band':<10s}  {'n':>6s}   {'P(answer_ok)':>14s}")
    for b in ["perfect", "high", "mid", "low", "near-zero"]:
        c = bcorrect.get(b, [])
        if not c:
            continue
        rate = sum(c) / len(c)
        print(f"    {b:<10s}  {len(c):>6d}   {rate:>14.4f}")

    # The crucial split: among `table_struct_ok=True, answer_ok=True`, what's the recall distribution?
    print(f"\n  Among 'structurally-OK + answer-correct' rollouts:")
    sub = [r for r in rs if r["table_struct_ok"] and r["answer_ok"]]
    if sub:
        recs = [r["n_recall"] for r in sub]
        print(f"    n={len(sub)}, recall mean={statistics.mean(recs):.4f}, median={statistics.median(recs):.4f}")
        below = sum(1 for x in recs if x < 0.5)
        print(f"    below 0.5 recall: {below} ({100*below/len(sub):.1f}%)  <- 'lucky-correct on bad table'")

    # The other crucial split: among `table_struct_ok=True, answer_ok=False`, what's the recall distribution?
    print(f"  Among 'structurally-OK + answer-WRONG' rollouts:")
    sub = [r for r in rs if r["table_struct_ok"] and not r["answer_ok"]]
    if sub:
        recs = [r["n_recall"] for r in sub]
        print(f"    n={len(sub)}, recall mean={statistics.mean(recs):.4f}, median={statistics.median(recs):.4f}")
        above = sum(1 for x in recs if x >= 0.8)
        below = sum(1 for x in recs if x < 0.5)
        print(f"    >= 0.8 recall: {above} ({100*above/len(sub):.1f}%)  <- 'truly bad reasoning' (good content, wrong answer)")
        print(f"    <  0.5 recall: {below} ({100*below/len(sub):.1f}%)  <- 'wrong content, wrong answer' (extraction problem)")

    # Pearson-ish correlation between current table_reward and numeric-recall
    pairs = [(r["table_reward"], r["n_recall"])
             for r in rs if r["table_reward"] is not None and r["n_recall"] is not None]
    if len(pairs) > 10:
        xs = [p[0] for p in pairs]; ys = [p[1] for p in pairs]
        mx, my = statistics.mean(xs), statistics.mean(ys)
        num = sum((x-mx)*(y-my) for x,y in pairs)
        dx = sum((x-mx)**2 for x in xs) ** 0.5
        dy = sum((y-my)**2 for y in ys) ** 0.5
        if dx > 0 and dy > 0:
            r_xy = num / (dx * dy)
            print(f"\n  Correlation (current table_reward, numeric_recall) over {len(pairs)} rows: r = {r_xy:.4f}")


for name, log in LOGS.items():
    analyze(name, log)
