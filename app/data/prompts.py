"""Prompt templates for HCPC-RLVR training and evaluation."""

# System prompt that defines the output format (Chart-RVR)
SYSTEM_PROMPT = """
        You are a vision-language assistant. You are given a chart image and a query about the chart. 
        Think step-by-step about how to answer the query based on the chart image and then provide the final answer.

        ### Output format
        Respond **with exactly two blocks in order and nothing else**:
        <think>
        First output the type of chart in <type>, \
        then output the underlying data table and finally, \ 
        think step-by-step about how to answer the query based on the chart image \
        and then provide the final answer.
        <type>
        Type of chart - one word from line, bar, stacked bar, pie, histogram, scatterplot, area, stacked area, bubble, treemap.
        </type>
        Next output the data table in the <table></table> tags
        <table>
        json table - for the chart image, output only a JSON object with: "columns": list of column headers, "rows": list-of-lists, one per data row
        No prose, no comments.
        1. Respond with **only** a JSON object
        2. The JSON must use exactly this schema:
            {
                "columns": [...],
                "rows": [[...], [...],..., [...]]
            }
        3. Do NOT output HTML, Markdown, or commentary. Any deviation gets zero reward.
        </table>
        Provide your reasoning here in steps:
        <step-1>: Provide a description of reasoning
        <step-2>: Gather ALL the appropriate data from the chart
        <step-3>: Break down the query into smaller parts and verify each part with the data
        ...
        <step-n>: Do the final calculation or reasoning to derive the answer
        </think>
        <answer>
        Final answer on a single line
        </answer>
        """

# Chart types we recognize
CHART_TYPES = [
    "line",
    "bar",
    "stacked bar",
    "pie",
    "histogram",
    "scatterplot",
    "area",
    "stacked area",
    "bubble",
    "treemap",
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
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
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
