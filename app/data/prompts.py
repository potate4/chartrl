"""Prompt templates for HCPC-RLVR training and evaluation."""

# System prompt that defines the output format
SYSTEM_PROMPT = """You are an expert in understanding and reasoning over charts. Given a chart image and a question, your task is to provide accurate answers with detailed reasoning.

Your response MUST follow this exact format:

<think>
<type>chart_type</type>
<table>{"columns": [...], "rows": [...]}</table>
Step 1: [First reasoning step]
Step 2: [Second reasoning step]
...
</think>
<answer>your final answer</answer>

Guidelines:
1. First identify the chart type (bar, line, pie, scatter, etc.)
2. Extract the data table from the chart in JSON format
3. Show your reasoning step by step
4. Provide a concise final answer

Important:
- The table must be valid JSON with "columns" and "rows" keys
- Each reasoning step should reference actual values from the table
- The final answer should be precise and match the question format
"""

# Chart types we recognize
CHART_TYPES = [
    "bar",
    "line",
    "pie",
    "scatter",
    "area",
    "histogram",
    "box",
    "heatmap",
    "treemap",
    "radar",
    "bubble",
    "waterfall",
    "funnel",
    "other",
]


def format_prompt(question: str, include_system: bool = True) -> str:
    """
    Format a question into a prompt.

    Args:
        question: The question about the chart
        include_system: Whether to include system prompt

    Returns:
        Formatted prompt string
    """
    if include_system:
        return f"{SYSTEM_PROMPT}\n\nQuestion: {question}"
    return f"Question: {question}"


def format_training_example(
    question: str,
    chart_type: str,
    table: dict,
    reasoning: str,
    answer: str,
) -> str:
    """
    Format a complete training example with expected output.

    Args:
        question: The question
        chart_type: Type of chart
        table: Data table as dict
        reasoning: Reasoning steps
        answer: Final answer

    Returns:
        Formatted expected output
    """
    import json

    # Ensure table is properly formatted
    if isinstance(table, dict):
        table_str = json.dumps(table, ensure_ascii=False)
    else:
        table_str = str(table)

    return f"""<think>
<type>{chart_type}</type>
<table>{table_str}</table>
{reasoning}
</think>
<answer>{answer}</answer>"""


def format_conversation(question: str, image_token: str = "<image>") -> list:
    """
    Format as conversation for chat models.

    Args:
        question: The question
        image_token: Token representing the image

    Returns:
        List of message dicts
    """
    # Use list-of-dicts content for all roles to keep schema consistent
    return [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": SYSTEM_PROMPT},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        },
    ]


def get_few_shot_examples() -> list:
    """
    Get few-shot examples for prompting.

    Returns:
        List of (question, response) tuples
    """
    examples = [
        {
            "question": "What is the total sales in 2020 and 2021?",
            "response": """<think>
<type>bar</type>
<table>{"columns": ["Year", "Sales"], "rows": [["2020", 100], ["2021", 150]]}</table>
Step 1: From the chart, I can see sales for 2020 is 100 and for 2021 is 150.
Step 2: Adding these values: 100 + 150 = 250
</think>
<answer>250</answer>""",
        },
        {
            "question": "Which category has the highest percentage?",
            "response": """<think>
<type>pie</type>
<table>{"columns": ["Category", "Percentage"], "rows": [["A", 35], ["B", 25], ["C", 40]]}</table>
Step 1: Looking at the pie chart, I can identify three categories: A (35%), B (25%), and C (40%).
Step 2: Comparing the percentages, C has 40% which is the highest.
</think>
<answer>C</answer>""",
        },
    ]
    return examples


def build_prompt_with_examples(question: str, n_examples: int = 1) -> str:
    """
    Build prompt with few-shot examples.

    Args:
        question: The question to answer
        n_examples: Number of examples to include

    Returns:
        Prompt with examples
    """
    examples = get_few_shot_examples()[:n_examples]

    prompt_parts = [SYSTEM_PROMPT, "\nHere are some examples:\n"]

    for i, ex in enumerate(examples, 1):
        prompt_parts.append(f"\nExample {i}:")
        prompt_parts.append(f"Question: {ex['question']}")
        prompt_parts.append(f"Response: {ex['response']}")

    prompt_parts.append(f"\nNow answer this question:")
    prompt_parts.append(f"Question: {question}")

    return "\n".join(prompt_parts)
