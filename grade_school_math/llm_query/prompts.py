"""
Shared prompt strings for generating and distilling MathQA rationales.
"""

from __future__ import annotations

PART_ONE_ROLE = """You are a rigorous but concise math tutor.
You solve math problems carefully and explain your reasoning briefly and clearly.
Avoid unnecessary prose; show only the key steps needed to reach the answer."""

PART_TWO_TASK = """
Your task:
Solve the question and return ONLY a valid JSON object.
Do NOT include extra text, markdown, explanations, preambles, or trailing commentary.
JSON only.
"""

NORMAL_PART_THREE = """
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
