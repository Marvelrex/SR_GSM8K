from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATASET_PATH = PROJECT_ROOT / "grade_school_math/data/train.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "structure_rationale"
DEFAULT_STRATEGY = "step"

PROMPT_STRATEGIES = {
    "normal": "NORMAL_PART_THREE",
    "fixed": "STRUCTURED_FIXED_PART_THREE",
    "step": "STRUCTURED_STEP_PART_THREE",
    "freeform": "STRUCTURED_FREE_FORM_PART_THREE",
    "noisy": "STRUCTURED_NOISE_PART_THREE",
}

ID_FIELDS: Sequence[str] = ("id", "question_id", "sample_id")
SANITIZE_RE = re.compile(r"[^0-9A-Za-z_.-]+")


def _entry_key(entry: Dict[str, Any]) -> str:
    return str(entry.get("index", ""))


def load_existing_results(path: Path) -> Tuple[List[Dict[str, Any]], set]:
    entries: List[Dict[str, Any]] = []
    processed_ids: set = set()
    if not path.exists():
        return entries, processed_ids
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    entries.append(obj)
                    idx = obj.get("index")
                    if idx is not None:
                        processed_ids.add(str(idx))
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"Warning: Could not read {path}: {exc}")
    entries.sort(key=_entry_key)
    return entries, processed_ids


def write_results(path: Path, entries: List[Dict[str, Any]]) -> None:
    """Rewrite the JSONL file with the provided entries."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in entries:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def _coerce_number(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        candidate = value.strip().replace(",", "")
        try:
            if candidate:
                num = float(candidate)
                if num.is_integer():
                    return int(num)
                return num
        except ValueError:
            return value
    return value


def load_samples(dataset_path: Path, start_index: int, num_samples: int) -> List[Tuple[int, dict]]:
    """Load a consecutive slice of GSM8K samples."""
    if start_index < 0 or num_samples <= 0:
        raise ValueError("sample index must be >= 0 and num-samples must be positive.")
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Could not find dataset at {dataset_path}. "
            "Use --dataset-path to point to a jsonl file."
        )

    collected: List[Tuple[int, dict]] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_idx, raw_line in enumerate(handle):
            if line_idx < start_index or not raw_line.strip():
                continue
            sample = json.loads(raw_line)
            collected.append((line_idx, sample))
            if len(collected) >= num_samples:
                break

    if len(collected) < num_samples:
        raise IndexError(
            f"Requested {num_samples} samples starting at index {start_index}, "
            f"but {dataset_path} ended early."
        )
    return collected


def pick_part_three_prompt(
    strategy: str, prompt_module, prompt_strategies: Dict[str, str] = PROMPT_STRATEGIES
) -> Tuple[str, str]:
    """Fetch the part-three instructions for a given strategy."""
    try:
        prompt_name = prompt_strategies[strategy]
    except KeyError as exc:
        valid = ", ".join(prompt_strategies)
        raise SystemExit(f"Unknown prompt strategy '{strategy}'. Valid options: {valid}") from exc

    try:
        prompt_text = getattr(prompt_module, prompt_name)
    except AttributeError as exc:
        raise SystemExit(
            f"Prompt constant '{prompt_name}' configured for strategy '{strategy}' "
            "was not found inside utils.prompts."
        ) from exc

    if not isinstance(prompt_text, str):
        raise TypeError(f"Prompt '{prompt_name}' is not a string.")

    return prompt_name, prompt_text.strip()


def build_prompt_texts(system_prompt: str, user_prompt: str, question: str) -> Tuple[str, str]:
    """Create labeled SYSTEM/USER prompt strings."""
    stripped_system = system_prompt.strip() or "You are a helpful math tutor."
    stripped_user = user_prompt.strip()
    question_text = question.strip()

    labeled_system = f"\n{stripped_system}"

    user_sections = [""]
    if question_text:
        user_sections.append(f"Question:\n{question_text}")
    if stripped_user:
        user_sections.append(stripped_user)

    labeled_user = "\n\n".join(user_sections)
    return labeled_system, labeled_user


def parse_model_output(response_text: str) -> Tuple[Any, Any]:
    """Extract rationale and answer from a model's JSON output."""
    if isinstance(response_text, str):
        # Find the JSON block, which might be wrapped in markdown
        match = re.search(r"```json\n({.*?})\n```", response_text, re.DOTALL)
        if match:
            json_text = match.group(1)
        else:
            # Fallback for plain JSON or JSON-like text
            brace_start = response_text.find("{")
            brace_end = response_text.rfind("}")
            if brace_start != -1 and brace_end != -1 and brace_start < brace_end:
                json_text = response_text[brace_start : brace_end + 1]
            else:
                return None, None  # No JSON found
    else:
        return None, None  # Input is not a string

    try:
        data = json.loads(json_text)
    except (TypeError, json.JSONDecodeError):
        return None, None

    rationale = data.get("rationale")
    answer = _coerce_number(data.get("ans"))
    return rationale, answer


def _content_chunks(response_obj) -> List[str]:
    """Yield text chunks from different model response shapes."""
    chunks: List[str] = []
    output_text = getattr(response_obj, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        chunks.append(output_text.strip())
        return chunks

    choices = getattr(response_obj, "choices", None)
    if choices:
        for choice in choices:
            message = getattr(choice, "message", None) or getattr(choice, "delta", None)
            if message is None and isinstance(choice, dict):
                message = choice.get("message") or choice.get("delta")

            content = None
            if message is not None:
                content = getattr(message, "content", None)
                if content is None and isinstance(message, dict):
                    content = message.get("content")
            if content is None:
                content = getattr(choice, "content", None)
                if content is None and isinstance(choice, dict):
                    content = choice.get("content")

            if isinstance(content, list):
                for chunk in content:
                    chunk_type = getattr(chunk, "type", None) or chunk.get("type")
                    if chunk_type and "text" not in chunk_type and chunk_type != "output_text":
                        continue
                    text_value = getattr(chunk, "text", None) or chunk.get("text")
                    if isinstance(text_value, str):
                        chunks.append(text_value)
            elif isinstance(content, str):
                chunks.append(content)
        if chunks:
            return chunks

    output = getattr(response_obj, "output", None)
    if output:
        for item in output:
            content = getattr(item, "content", None) or getattr(item, "text", None)
            if isinstance(content, list):
                for chunk in content:
                    chunk_type = getattr(chunk, "type", None) or chunk.get("type")
                    if chunk_type and "text" not in chunk_type and chunk_type != "output_text":
                        continue
                    text_value = getattr(chunk, "text", None) or chunk.get("text")
                    if isinstance(text_value, str):
                        chunks.append(text_value)
            elif isinstance(content, str):
                chunks.append(content)
        return chunks

    content = getattr(response_obj, "content", None)
    if isinstance(content, list):
        for chunk in content:
            text_value = getattr(chunk, "text", None) or chunk.get("text")
            if isinstance(text_value, str):
                chunks.append(text_value)
    return chunks


def extract_response_text(response) -> str:
    """Convert a model response payload into plain text."""
    chunks = _content_chunks(response)
    if chunks:
        return "\n".join(chunk.strip() for chunk in chunks if chunk.strip())
    return str(response)


def sanitize_identifier(value: str, dataset_name: str, absolute_index: int) -> str:
    """Create a filesystem-safe identifier for saving responses."""
    candidate = SANITIZE_RE.sub("_", value).strip("._")
    if candidate:
        return candidate
    return f"{dataset_name}_{absolute_index:05d}"


def sample_identifier(sample: dict, dataset_path: Path, absolute_index: int) -> str:
    """Determine a stable identifier for a dataset row."""
    dataset_name = dataset_path.stem or "sample"
    for key in ID_FIELDS:
        if key in sample and sample[key] not in (None, ""):
            return sanitize_identifier(str(sample[key]), dataset_name, absolute_index)
    return f"{dataset_name}_{absolute_index:05d}"

def split_answer_and_rationale(text: str) -> tuple[str, str]:
    """Split a string into a rationale and a final answer."""
    if not isinstance(text, str):
        return "", ""
    parts = text.split("####")
    if len(parts) == 1:
        return "", parts[0].strip()
    rationale = parts[0].strip()
    answer = parts[1].strip()
    return rationale, answer
