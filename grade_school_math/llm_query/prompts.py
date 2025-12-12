"""
Shared prompt strings for generating and distilling MathQA rationales.
"""

from __future__ import annotations

GSM8K_FEW_SHOT_EXAMPLES = [
    (
        "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        """```json
{"rationale": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6.", "ans": 6}
```""",
    ),
    (
        "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        """```json
{"rationale": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.", "ans": 5}
```""",
    ),
    (
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        """```json
{"rationale": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.", "ans": 39}
```""",
    ),
    (
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        """```json
{"rationale": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.", "ans": 8}
```""",
    ),
    (
        "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        """```json
{"rationale": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9.", "ans": 9}
```""",
    ),
    (
        "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        """```json
{"rationale": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29.", "ans": 29}
```""",
    ),
    (
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        """```json
{"rationale": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33.", "ans": 33}
```""",
    ),
    (
        "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        """```json
{"rationale": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8.", "ans": 8}
```""",
    ),
]

DEFAULT_NORMAL_FEW_SHOT_COUNT = len(GSM8K_FEW_SHOT_EXAMPLES)

PART_ONE_ROLE = """You are a rigorous but concise math tutor.
You solve math problems carefully and explain your reasoning briefly and clearly.
Avoid unnecessary prose; show only the key steps needed to reach the answer."""

PART_TWO_TASK = """
Your task:
Solve the question and return ONLY a valid JSON object.
Do NOT include extra text, markdown, explanations, preambles, or trailing commentary.
JSON only.
Ensure the JSON is valid (use a comma between "rationale" and "ans" fields).
"""

NORMAL_BASE_INSTRUCTIONS = """
Think step by step in a few precise steps (no more than six sentences) to solve the problem.

Then output ONLY a compact JSON object of the form:
{
  "rationale": "<explanation>",
  "ans": <numeric_answer>
}

Rules:
- "ans" must be a number, not a string.
- No additional text before or after the JSON.
"""

def _format_normal_examples(examples: list[tuple[str, str]]) -> str:
    lines = []
    for idx, (question, answer) in enumerate(examples, start=1):
        lines.append(f"Example {idx}:")
        lines.append(f"Question: {question}")
        lines.append(f"Reasoning and answer: {answer}")
        lines.append("")
    return "\n".join(lines).strip()


def NORMAL_PART_THREE(num_shots: int | None = None) -> str:
    """Few-shot Chain-of-Thought instructions for the normal strategy."""
    if num_shots is None:
        count = DEFAULT_NORMAL_FEW_SHOT_COUNT
    else:
        count = max(0, int(num_shots))

    prompt_sections = [NORMAL_BASE_INSTRUCTIONS.strip()]
    if count:
        selected_examples = GSM8K_FEW_SHOT_EXAMPLES[:count]
        examples_block = _format_normal_examples(selected_examples)
        prompt_sections.append(
            f"Here are {len(selected_examples)} worked examples to mirror:"
            f"\n{examples_block}"
        )
        prompt_sections.append(
            "After the examples, solve the question above in the same concise reasoning style and "
            "return only the JSON object described."
        )

    return "\n\n".join(prompt_sections).strip()

STRUCTURED_NOISE_PART_THREE = """
Reason in three labeled parts:
- Understanding: Briefly state what the problem asks (one sentence).
- Derivation: Show the essential math steps (no more than four short sentences).
- Calculation: Provide the final numerical computation (one short sentence).

Then output ONLY a valid JSON object:
{
  "rationale": {
    "Understanding": "<one short sentence>",
    "Derivation": "<up to four short sentences>",
    "Calculation": "<one short sentence>"
  },
  "ans": <numeric final answer>
}

Rules:
- JSON only.
- All rationale fields must be concise.
- "ans" must be numeric.
"""

STRUCTURED_STEP_PART_THREE = """
Think through the solution step by step. Use a small number of labeled steps.
Each step may contain up to three short sentences.

Output ONLY this JSON format:
{
  "rationale": {
    "step_one": "<up to three concise sentences>",
    "step_two": "<up to three concise sentences>",
    "step_three": "... as needed ..."
  },
  "ans": <numeric final answer>
}

Rules:
- JSON only.
- Steps must be sequentially named (step_one, step_two, step_three, ...).
- "ans" must be numeric.
"""


STRUCTURED_FIXED_PART_THREE = """
Reason in two labeled parts:
- Understanding: Briefly state what the problem asks (one sentence).
- Derivation: Provide the key math steps (up to six short sentences).

Output ONLY this JSON format:
{
  "rationale": {
    "Understanding": "<one short sentence>",
    "Derivation": "<up to six short sentences>"
  },
  "ans": <numeric final answer>
}

Rules:
- JSON only.
- "ans" must be numeric.
"""

STRUCTURED_FREE_FORM_PART_THREE = """
Reason freely using up to three fields under "rationale".
Choose field names appropriate to the problem (e.g., "Setup", "Logic", "Compute").

Output ONLY this JSON format:
{
  "rationale": {
    "<field_name_1>": "<short sentences>",
    "<field_name_2>": "<short sentences>",
    "<field_name_3_optional>": "<short sentences>"
  },
  "ans": <numeric final answer>
}

Rules:
- Use no more than three rationale fields.
- Field names must be human-readable.
- Each field must contain short, essential reasoning.
- Do not use abbreviations.
- JSON only.
- "ans" must be numeric.
"""
