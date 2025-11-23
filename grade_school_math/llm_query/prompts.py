"""
Shared prompt strings for generating and distilling MathQA rationales.
"""

from __future__ import annotations

PART_ONE_ROLE = """You are a rigorous but concise math tutor. 
You solve math problems carefully and explain your reasoning briefly and clearly.
Avoid unnecessary prose or speculation; show only the key steps needed to reach the answer."""

PART_TWO_TASK = """
Your task:
Solve the question and give a numeric answer in json format.
"""

NORMAL_PART_THREE = """
Think step by step in a few precise steps (No more than six sentences) to solve the problem.
Then output ONLY a compact JSON string with the following keys:
  {"rationale": "<explanation>", "ans": <numeric_answer>}
Do not include anything else before or after the JSON. The "ans" must be a number, not a string.
"""

STRUCTURED_NOISE_PART_THREE = """
Think through the solution by reasoning in three labeled parts:
- Understanding: Briefly interpret the problem and identify what is being asked (one sentence).
- Derivation: Show the essential mathematical steps or transformations needed to solve it, no more than four short sentences (you may include simple equations).
- Calculation: Provide the final numerical computation in one short sentence.

Then return ONLY a valid JSON object with the following structure:

{
  "rationale": {
    "Understanding": "<one or two short sentence>",
    "Derivation": "<no more than four short sentences>",
    "Calculation": "<one or two short sentence>"
  },
  "ans": <numeric final answer>
}

Requirements:
- Output valid JSON only.
- The rationale fields must be concise but cover the key reasoning steps.
- The "ans" field must be a number, not a string.
"""

STRUCTURED_STEP_PART_THREE = """
Think through the solution step by step. Break the reasoning into a small number of concise labeled steps 
(The number of steps depends on question difficulty but no more than five steps). Each step should contain no more than three short sentences.

Then return ONLY a valid JSON object with the following structure:

{
  "rationale": {
    "step_one": "<no more than three sentences>",
    "step_two": "<no more than three sentences>",
    "...": "... as needed ..."
  },
  "ans": <numeric final answer>
}

Requirements:
- Output valid JSON only.
- Number steps sequentially as step_one, step_two, step_three, etc.
- Each step must be logically incremental and relevant.
- The "ans" field must be a number, not a string.
"""


STRUCTURED_FIXED_PART_THREE = """
Think through the solution by reasoning in two labeled parts:
- Understanding: Briefly interpret the problem and identify what is being asked (one sentence).
- Derivation: Show the essential mathematical steps or transformations needed to solve it, no more than four short sentences (you may include simple equations).

Then return ONLY a valid JSON object with the following structure:

{
  "rationale": {
    "Understanding": "<one or two short sentence>",
    "Derivation": "<No more than six short sentences>"
  },
  "ans": <numeric final answer>
}

Requirements:
- Output valid JSON only.
- The rationale fields must be concise but cover the key reasoning steps.
- The "ans" field must be a number, not a string.
"""

STRUCTURED_FREE_FORM_PART_THREE = """
Think through the solution by reasoning, you can set the field_name according to the question details.
Then return ONLY a valid JSON object with the following structure:

{
  "rationale": {
    "<field_name_1>": "<short sentences>",
    "<field_name_2>": "<short sentences>",
    "<field_name_3_optional>": "<short sentences>"
  },
  "ans": <numeric final answer>
}

Follow these rules:
- Under "rationale", use no more than three fields in total.
- You may choose any human-readable field names that reflect their purpose.
- Each field value must be very concise, using short, minimal-word sentences.
- Keep only the important reasoning steps needed to solve the question.
- Do not use abbreviations.
- The "ans" field must be a number, not a string.
- Respond with valid JSON only. Do not include any extra text, explanation, or markdown before or after the JSON.
"""
