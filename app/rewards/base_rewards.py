"""Base reward functions from Chart-RVR (commit-aligned)."""

import re
import json
import math
import difflib
from typing import Dict, Any, Optional, List

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


_TEXT_REWARD_MODEL = None


def _get_text_reward_model():
    global _TEXT_REWARD_MODEL
    if _TEXT_REWARD_MODEL is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _TEXT_REWARD_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
        _TEXT_REWARD_MODEL.eval()
    return _TEXT_REWARD_MODEL


def _text_sim(pred: str, gt: str) -> float:
    if not pred or not gt:
        return 0.0
    model = _get_text_reward_model()
    device = model.device
    p_emb = model.encode(pred, convert_to_tensor=True, device=device)
    gt_emb = model.encode(gt, convert_to_tensor=True, device=device)
    cos = F.cosine_similarity(p_emb, gt_emb, dim=-1)
    return cos.max(dim=0).values.mean().item()


def format_reward(completion: str) -> float:
    """
    Reward function that checks if the completion has the expected format.

    Full format (2.0 points):
        <think>
        <type>...</type>
        <table>...</table>
        ...reasoning...
        </think>
        <answer>...</answer>

    Partial rewards help the model learn the format incrementally.
    """
    # Full format match (strict)
    full_pattern = r"^<think>\n<type>.*?</type>\n<table>.*?</table>.*?</think>\n<answer>.*?</answer>$"
    if re.match(full_pattern, completion, re.DOTALL | re.MULTILINE):
        return 2.0

    # Partial rewards for learning the format incrementally
    reward = 0.0

    # Has think block
    if "<think>" in completion and "</think>" in completion:
        reward += 0.3

    # Has answer block
    if "<answer>" in completion and "</answer>" in completion:
        reward += 0.3

    # Has type tag (inside think)
    if "<type>" in completion and "</type>" in completion:
        reward += 0.2

    # Has table tag (inside think)
    if "<table>" in completion and "</table>" in completion:
        reward += 0.2

    # Correct order: think before answer
    think_pos = completion.find("</think>")
    answer_pos = completion.find("<answer>")
    if think_pos > 0 and answer_pos > think_pos:
        reward += 0.2

    # Type before table
    type_pos = completion.find("</type>")
    table_pos = completion.find("<table>")
    if type_pos > 0 and table_pos > type_pos:
        reward += 0.2

    # Cap partial reward at 1.4 (full format gets 2.0 bonus)
    return min(reward, 1.4)


def _try_float(value: str) -> Optional[float]:
    if value is None:
        return None
    try:
        cleaned = str(value).strip()
        cleaned = cleaned.replace(",", "")
        if cleaned.endswith("%"):
            cleaned = cleaned[:-1]
        return float(cleaned)
    except Exception:
        return None


def accuracy_reward(completion: str, label: str, tolerance: float = 0.05) -> float:
    """Continuous accuracy reward with smooth decay as deviation increases."""
    if not label:
        return 0.0

    try:
        if "<answer>" not in completion or "<think>" not in completion:
            pred = ""
        else:
            pred = completion.split("<answer>")[-1].strip().split("</answer>")[0].strip()
            pred = pred.rstrip(".") if pred.endswith(".") else pred
    except Exception:
        pred = ""

    if not pred:
        return 0.0

    label_num = _try_float(label)
    pred_num = _try_float(pred)
    if label_num is not None and pred_num is not None:
        denom = abs(label_num) if abs(label_num) > 1e-8 else 1.0
        rel_error = abs(pred_num - label_num) / denom
        # Smooth decay from 1.0 at exact match; no flat tolerance region.
        k = 10.0
        return float(max(0.0, min(1.0, math.exp(-k * rel_error))))

    try:
        ratio = difflib.SequenceMatcher(None, str(label).lower(), str(pred).lower()).ratio()
        return float(ratio)
    except Exception:
        return 0.0

def length_reward(completion: str) -> float:
    """Length reward (Chart-RVR)."""
    if not completion:
        return 0.0

    reward = 0.0
    try:
        rationale = completion.split("<think>")[-1].strip().split("</think>")[0].strip()
    except Exception:
        rationale = ""

    if len(rationale) > 150:
        reward += 1.0
    if len(rationale) > 250:
        reward -= 1.0
    if len(rationale) > 70:
        reward += 1.0
    if len(rationale) > 150:
        reward -= 1.0

    steps = rationale.split("<step-")
    reward += min(0.25 * len(steps), 1.5)

    if len(rationale) > 500:
        reward = 0.0
    return reward


def token_count_reward(completion: str) -> float:
    _PATTERNS = [
        re.compile(r"<type>"),
        re.compile(r"</type>"),
        re.compile(r"<think>"),
        re.compile(r"</think>"),
        re.compile(r"<answer>"),
        re.compile(r"</answer>"),
        re.compile(r"<table>"),
        re.compile(r"</table>"),
        re.compile(r"<think>\n<type>"),
        re.compile(r"</type>\n<table>"),
    ]
    return 2.0 * int(all(len(p.findall(completion)) == 1 for p in _PATTERNS))


def chart_type_reward(completion: str, chart_type: str) -> float:
    if not chart_type:
        return 0.0
    try:
        pred_type = completion.split("<type>")[-1].strip().split("</type>")[0].strip().lower()
        return 1.0 if pred_type == str(chart_type).strip().lower() else 0.0
    except Exception:
        return 0.0


def _extract_table_block(completion: str) -> str:
    try:
        return completion.split("<table>")[-1].strip().split("</table>")[0].strip()
    except Exception:
        return ""


def _parse_table_from_completion(completion: str) -> Optional[Dict[str, Any]]:
    tab_struct = _extract_table_block(completion)
    if not tab_struct:
        return None
    if "```json" in tab_struct:
        block = tab_struct.split("```json")[-1].split("```")[0].strip()
    else:
        block = tab_struct.replace("\n", "").strip("\n").strip()
    try:
        return json.loads(block, parse_int=str, parse_float=str, parse_constant=str)
    except Exception:
        return None


def _compare_tables(pred: Dict[str, Any], gt: Dict[str, Any]) -> float:
    try:
        gt["columns"] = sorted(gt["columns"], key=lambda x: str(x).lower())
        pred["columns"] = sorted(pred["columns"], key=lambda x: str(x).lower())
    except Exception:
        pass

    try:
        if all(isinstance(row, list) for row in gt.get("rows", [])):
            gt["rows"] = sorted([g for g in gt["rows"]])
        if all(isinstance(row, list) for row in pred.get("rows", [])):
            pred["rows"] = sorted([g for g in pred["rows"]])
    except Exception:
        pass

    reward = 0.0
    try:
        min_cols = min(len(pred.get("columns", [])), len(gt.get("columns", [])))
        for col in range(min_cols):
            if str(pred["columns"][col]).lower() == str(gt["columns"][col]).lower():
                reward += 0.5 * float(1 / len(pred["columns"]))
    except Exception:
        pass

    try:
        if all(isinstance(row, list) for row in pred.get("rows", [])) and all(
            isinstance(row, list) for row in gt.get("rows", [])
        ):
            min_rows = min(len(pred["rows"]), len(gt["rows"]))
            for row in range(min_rows):
                min_cols_in_row = min(len(pred["rows"][row]), len(gt["rows"][row]))
                for row_id in range(min_cols_in_row):
                    if pred["rows"][row][row_id] == gt["rows"][row][row_id]:
                        reward += 0.5 * float(1.0 / len(pred["rows"]))
    except Exception:
        pass

    return reward


def table_reward(completion: str, table: Dict[str, Any]) -> float:
    if not table:
        return 0.0
    reward = 0.0
    pred_table = _parse_table_from_completion(completion)
    if pred_table is None:
        return 0.0
    reward += 0.5  # parseable JSON
    try:
        reward += _compare_tables(pred_table, table)
    except Exception:
        pass
    try:
        if set(pred_table) == {"columns", "rows"}:
            reward += 0.25
    except Exception:
        pass
    return reward


def process_reward(completion: str, reasoning: str) -> float:
    if not reasoning:
        return 0.0
    try:
        steps = completion.split("</table>")[-1].strip().split("</think>")[0].strip()
    except Exception:
        steps = ""
    if not steps:
        return 0.0
    return _text_sim(steps, reasoning)


def compute_base_rewards(
    completion: str,
    ground_truth: Dict[str, Any],
    use_format: bool = True,
    use_accuracy: bool = True,
    use_length: bool = True,
    use_token_count: bool = True,
    use_chart_type: bool = True,
    use_table: bool = True,
    use_process: bool = True,
) -> Dict[str, float]:
    """
    Compute base rewards for a completion.

    Args:
        completion: Model output string
        ground_truth: Dict with label, table, chart_type, reasoning
        use_format: Whether to compute format reward
        use_accuracy: Whether to compute accuracy reward
        use_length: Whether to compute length reward
        use_token_count: Whether to compute token count reward
        use_chart_type: Whether to compute chart type reward
        use_table: Whether to compute table reward
        use_process: Whether to compute process reward (GT reasoning similarity)
                     Set to False when using HCPC (which promotes diversity instead)

    Returns:
        Dict of reward component values
    """
    rewards = {}
    rewards["format"] = format_reward(completion) if use_format else 0.0
    rewards["accuracy"] = accuracy_reward(completion, ground_truth.get("label", "")) if use_accuracy else 0.0
    rewards["length"] = length_reward(completion) if use_length else 0.0
    rewards["token_count"] = token_count_reward(completion) if use_token_count else 0.0
    rewards["chart_type"] = chart_type_reward(completion, ground_truth.get("chart_type", "")) if use_chart_type else 0.0
    rewards["table"] = table_reward(completion, ground_truth.get("table", {})) if use_table else 0.0
    # process_reward measures similarity to GT reasoning
    # Disable when using HCPC which promotes diversity instead
    rewards["process"] = process_reward(completion, ground_truth.get("reasoning", "")) if use_process else 0.0
    rewards["total"] = sum(rewards.values())
    return rewards
