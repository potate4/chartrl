"""Parsing utilities for extracting structured components from model outputs."""

import re
import json
from typing import Dict, List, Optional, Any, Set


def parse_response(text: str) -> Dict[str, Any]:
    """
    Parse model output into structured components.

    Expected format:
        <think>
        <type>chart_type</type>
        <table>{"columns": [...], "rows": [...]}</table>
        reasoning steps...
        </think>
        <answer>final answer</answer>

    Args:
        text: Raw model output

    Returns:
        Dictionary with keys: type, table, reasoning, answer, raw
    """
    result = {
        "type": "",
        "table": {},
        "table_raw": "",
        "table_parse_success_strict": False,
        "reasoning": "",
        "answer": "",
        "raw": text,
        "parse_success": True,
    }

    # Extract chart type
    type_match = re.search(r"<type>(.*?)</type>", text, re.DOTALL)
    if type_match:
        result["type"] = type_match.group(1).strip().lower()

    # Extract table JSON
    table_match = re.search(r"<table>(.*?)</table>", text, re.DOTALL)
    if table_match:
        table_str = table_match.group(1).strip()
        result["table_raw"] = table_str
        # Strict JSON parse for format reward consistency
        try:
            json.loads(table_str)
            result["table_parse_success_strict"] = True
        except json.JSONDecodeError:
            result["table_parse_success_strict"] = False
        # Lenient parse for downstream table similarity
        result["table"] = parse_json_table(table_str)

    # Extract answer
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if answer_match:
        result["answer"] = answer_match.group(1).strip()

    # Extract reasoning (content in <think> after </table> and before </think>)
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        think_content = think_match.group(1)
        # Remove type and table tags to get reasoning
        reasoning = re.sub(r"<type>.*?</type>", "", think_content, flags=re.DOTALL)
        reasoning = re.sub(r"<table>.*?</table>", "", reasoning, flags=re.DOTALL)
        result["reasoning"] = reasoning.strip()

    # Check parse success
    if not result["type"] or not result["answer"]:
        result["parse_success"] = False

    return result


def _normalize_table_obj(obj: Any) -> Dict[str, Any]:
    """Return a canonical table dict or {} for malformed parsed JSON."""
    if not isinstance(obj, dict):
        return {}

    columns = obj.get("columns", [])
    rows = obj.get("rows", [])
    if not isinstance(columns, list) or not isinstance(rows, list):
        return {}

    return {"columns": columns, "rows": rows}


def parse_json_table(table_str: str) -> Dict[str, Any]:
    """
    Parse JSON table string into dictionary.

    Handles common formatting issues in model outputs.

    Args:
        table_str: JSON string of table

    Returns:
        Parsed dictionary or empty dict on failure
    """
    if not table_str:
        return {}

    # Clean common issues
    table_str = table_str.strip()

    # If wrapped in ```json ... ``` take the inner block
    if "```json" in table_str:
        try:
            table_str = table_str.split("```json", 1)[-1].split("```", 1)[0].strip()
        except Exception:
            pass

    # Try direct parse
    try:
        return _normalize_table_obj(json.loads(table_str))
    except json.JSONDecodeError:
        pass

    # Try fixing common issues
    try:
        # Replace single quotes with double quotes
        fixed = table_str.replace("'", '"')
        return _normalize_table_obj(json.loads(fixed))
    except json.JSONDecodeError:
        pass

    # Try extracting just the JSON object
    try:
        match = re.search(r"\{.*\}", table_str, re.DOTALL)
        if match:
            return _normalize_table_obj(json.loads(match.group()))
    except json.JSONDecodeError:
        pass

    return {}


def extract_numbers(text: str) -> Set[float]:
    """
    Extract all numeric values from text.

    Handles integers, floats, percentages, and negative numbers.

    Args:
        text: Input text

    Returns:
        Set of numeric values found
    """
    if not text:
        return set()

    numbers = set()

    # Pattern for numbers (handles decimals, negatives, percentages)
    pattern = r"-?\d+\.?\d*%?"

    for match in re.finditer(pattern, text):
        num_str = match.group()

        # Handle percentage
        is_percentage = num_str.endswith("%")
        if is_percentage:
            num_str = num_str[:-1]

        try:
            value = float(num_str)
            if is_percentage:
                value = value / 100
            numbers.add(value)
        except ValueError:
            continue

    return numbers


def extract_table_values(table: Dict[str, Any]) -> Set[float]:
    """
    Extract all numeric values from a table dictionary.

    Args:
        table: Table dictionary with 'columns' and 'rows' keys

    Returns:
        Set of numeric values in the table
    """
    values = set()

    if not table or not isinstance(table, dict):
        return values

    def extract_from_value(v):
        """Recursively extract numbers from a value."""
        if isinstance(v, (int, float)):
            values.add(float(v))
        elif isinstance(v, str):
            values.update(extract_numbers(v))
        elif isinstance(v, list):
            for item in v:
                extract_from_value(item)
        elif isinstance(v, dict):
            for val in v.values():
                extract_from_value(val)

    # Extract from rows
    rows = table.get("rows", [])
    for row in rows:
        extract_from_value(row)

    # Extract from columns (in case they contain numeric data)
    columns = table.get("columns", [])
    for col in columns:
        extract_from_value(col)

    return values


def normalize_answer(answer: str) -> str:
    """
    Normalize answer string for comparison.

    Args:
        answer: Raw answer string

    Returns:
        Normalized lowercase string
    """
    if not answer:
        return ""

    # Strip and lowercase
    normalized = answer.strip().lower()

    # Drop common lead-in phrases
    normalized = re.sub(
        r"^(the\s+answer\s+is|answer\s+is|it\s+is|it's|approximately|about)\s+",
        "",
        normalized,
    )

    # Normalize british/american spelling variants
    normalized = normalized.replace("grey", "gray")

    # Remove currency words/symbols and commas
    normalized = normalized.replace(",", "")
    normalized = re.sub(r"\b(usd|dollar|dollars)\b", "", normalized)
    normalized = normalized.replace("$", "").strip()

    # Remove trailing punctuation
    normalized = re.sub(r"[.,;:!?]+$", "", normalized)

    # Normalize whitespace
    normalized = " ".join(normalized.split())

    return normalized


def _parse_numeric_with_units(text: str) -> Optional[float]:
    """
    Parse a numeric value with optional magnitude suffixes/words.

    Examples:
        "42 million" -> 42000000
        "$42m" -> 42000000
        "3.5 billion" -> 3500000000
        "12%" -> 0.12
    """
    if not text:
        return None

    cleaned = text.strip().lower()
    cleaned = cleaned.replace(",", "")
    cleaned = cleaned.replace("$", "")

    # Handle percent word
    cleaned = cleaned.replace("percent", "%")

    pattern = re.compile(
        r"(-?\d+(?:\.\d+)?)\s*(%|k|m|b|t|thousand|million|billion|trillion)?"
    )

    match = pattern.search(cleaned)
    if not match:
        return None

    number = float(match.group(1))
    suffix = match.group(2)

    if suffix == "%":
        return number

    # Treat magnitude suffixes/words as cosmetic by default
    # (e.g., "1.5 billion" -> 1.5)
    if suffix in {"k", "m", "b", "t", "thousand", "million", "billion", "trillion"}:
        return number

    return number


def try_parse_numeric(value: str) -> Optional[float]:
    """
    Try to parse a string as a numeric value.

    Args:
        value: String to parse

    Returns:
        Float value or None if not numeric
    """
    if not value:
        return None

    # First try parsing with unit suffixes/words
    parsed = _parse_numeric_with_units(value)
    if parsed is not None:
        return parsed

    # Clean the string
    cleaned = value.strip()

    # Remove common prefixes/suffixes
    cleaned = re.sub(r"^[$%]", "", cleaned)
    cleaned = re.sub(r"[$%]$", "", cleaned)
    cleaned = cleaned.replace(",", "")  # Remove thousand separators

    try:
        return float(cleaned)
    except ValueError:
        return None


def split_reasoning_steps(reasoning: str) -> List[str]:
    """
    Split reasoning into individual steps.

    Handles various step markers: numbers, bullets, newlines.

    Args:
        reasoning: Reasoning text

    Returns:
        List of reasoning steps
    """
    if not reasoning:
        return []

    # Try splitting by numbered steps (1., 2., etc.)
    numbered = re.split(r"\n?\d+[.)]\s*", reasoning)
    if len(numbered) > 1:
        return [s.strip() for s in numbered if s.strip()]

    # Try splitting by bullet points
    bulleted = re.split(r"\n?[-*]\s*", reasoning)
    if len(bulleted) > 1:
        return [s.strip() for s in bulleted if s.strip()]

    # Try splitting by "Step" markers
    stepped = re.split(r"\n?Step\s*\d*[:.]\s*", reasoning, flags=re.IGNORECASE)
    if len(stepped) > 1:
        return [s.strip() for s in stepped if s.strip()]

    # Try splitting by <step-1>: style tags
    tagged = re.split(r"\n?<step-\d+>:\s*", reasoning, flags=re.IGNORECASE)
    if len(tagged) > 1:
        return [s.strip() for s in tagged if s.strip()]

    # Fall back to sentence splitting
    sentences = re.split(r"(?<=[.!?])\s+", reasoning)
    return [s.strip() for s in sentences if s.strip()]


def check_format_compliance(text: str) -> Dict[str, bool]:
    """
    Check if output follows expected format.

    Args:
        text: Model output

    Returns:
        Dictionary with compliance flags
    """
    checks = {
        "has_think_tags": bool(re.search(r"<think>.*</think>", text, re.DOTALL)),
        "has_answer_tags": bool(re.search(r"<answer>.*</answer>", text, re.DOTALL)),
        "has_type_tags": bool(re.search(r"<type>.*</type>", text, re.DOTALL)),
        "has_table_tags": bool(re.search(r"<table>.*</table>", text, re.DOTALL)),
        "proper_order": False,
        "single_tags": True,
    }

    # Check order: <think> before <answer>
    think_pos = text.find("<think>")
    answer_pos = text.find("<answer>")
    if think_pos >= 0 and answer_pos >= 0:
        checks["proper_order"] = think_pos < answer_pos

    # Check for duplicate tags
    for tag in ["<think>", "</think>", "<answer>", "</answer>"]:
        if text.count(tag) > 1:
            checks["single_tags"] = False
            break

    checks["fully_compliant"] = all(checks.values())

    return checks
