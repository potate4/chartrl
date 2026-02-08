"""Base reward functions from Chart-RVR (commit-aligned)."""

import re
import json
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


def accuracy_reward(completion: str, label: str, tolerance: float = 0.05) -> float:
    """Chart-RVR accuracy reward (numeric tolerance or exact text)."""
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

    try:
        sol = float(label) + 1e-6  # avoid zero division
        pred_num = float(pred)
        reward = int(float(abs(pred_num - sol) / sol) <= tolerance)
        return float(reward)
    except Exception:
        try:
            return float(int(str(label).lower() == str(pred).lower()))
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


def compute_base_rewards(completion: str, ground_truth: Dict[str, Any]) -> Dict[str, float]:
    rewards = {}
    rewards["format"] = format_reward(completion)
    rewards["accuracy"] = accuracy_reward(completion, ground_truth.get("label", ""))
    rewards["length"] = length_reward(completion)
    rewards["token_count"] = token_count_reward(completion)
    rewards["chart_type"] = chart_type_reward(completion, ground_truth.get("chart_type", ""))
    rewards["table"] = table_reward(completion, ground_truth.get("table", {}))
    rewards["process"] = process_reward(completion, ground_truth.get("reasoning", ""))
    rewards["total"] = sum(rewards.values())
    return rewards
