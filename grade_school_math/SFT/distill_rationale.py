#!/usr/bin/env python3
"""
Distill the filtered rationale datasets (normal or structured) into a compact Llama 3.2 1B model.

Datasets live in this repo:
  - Normal rationales: filtered_gpt5_rationales/filtered_normal.jsonl
  - Structured rationales (step/fixed/freeform/noisy): data/structure_rationale/gpt5/third_run/<strategy>/results.jsonl

Use --mode {normal, structured} and (for structured) --strategy to pick the prompt & dataset.
Structured mode defaults to filtered_gpt5_rationales/filtered_<strategy>.jsonl and further
filters rows to the IDs listed in filtered_gpt5_rationales/filtered_intersection_ids.txt.
You can cap training data with --train-size. LoRA is enabled by default; pass --no-lora to disable.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import sys
import re

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:  # pragma: no cover - import shim for flexible entrypoints
    from llm_query import prompts as prompt_module
except ImportError:  # pragma: no cover
    from grade_school_math.llm_query import prompts as prompt_module  # type: ignore

from datasets import Dataset  # type: ignore
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

try:
    from peft import LoraConfig, get_peft_model
except ImportError:  # pragma: no cover - peft is optional until LoRA is requested.
    LoraConfig = None  # type: ignore
    get_peft_model = None  # type: ignore

DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

NORMAL_DATA_FILE = REPO_ROOT / "filtered_gpt5_rationales" / "filtered_normal.jsonl"
STRUCT_BASE_DIR = REPO_ROOT / "filtered_gpt5_rationales"
INTERSECTION_IDS = REPO_ROOT / "filtered_gpt5_rationales" / "filtered_intersection_ids.txt"

DEFAULT_OUTPUT_DIRS = {
    "normal": REPO_ROOT / "SFT" / "outputs" / "distilled-normal",
    "structured": REPO_ROOT / "SFT" / "outputs" / "distilled-structured",
}

STRATEGY_TO_DIR: Dict[str, str] = {
    "step": "step",
    "fixed": "fixed",
    "freeform": "freeform",
    "noisy": "noisy",
}

STRATEGY_TO_PROMPT: Dict[str, str] = {
    "step": prompt_module.STRUCTURED_STEP_PART_THREE.strip(),
    "fixed": prompt_module.STRUCTURED_FIXED_PART_THREE.strip(),
    "freeform": prompt_module.STRUCTURED_FREE_FORM_PART_THREE.strip(),
    "noisy": prompt_module.STRUCTURED_NOISE_PART_THREE.strip(),
}


@dataclass
class DistillConfig:
    data_file: Path
    output_dir: Path
    mode: str
    strategy: str
    intersection_file: Optional[Path]
    model_name: str
    max_samples: Optional[int]
    train_size: Optional[int]
    max_length: int
    learning_rate: float
    weight_decay: float
    num_epochs: float
    batch_size: int
    grad_accum: int
    warmup_ratio: float
    logging_steps: int
    save_steps: int
    bf16: bool
    use_lora: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    pad_to_max: bool
    generate: bool
    test_file: Path
    gen_output_file: Path
    max_gen_samples: Optional[int]
    max_new_tokens: int
    train_size: Optional[int]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Distill MathQA rationales into Llama 3.2 1B Instruct.")
    parser.add_argument("--mode", choices=["normal", "structured"], default="normal", help="Which rationale dataset to use.")
    parser.add_argument(
        "--strategy",
        choices=list(STRATEGY_TO_DIR.keys()),
        default="step",
        help="Structured strategy to use when --mode=structured.",
    )
    parser.add_argument("--data-file", type=Path, help="Path to MathQA rationale JSONL (overrides mode default).")
    parser.add_argument("--output-dir", type=Path, help="Directory to store the finetuned model (overrides mode default).")
    parser.add_argument("--model-name", default=DEFAULT_MODEL, help="HF model id to finetune (default: Llama 3.2 1B Instruct).")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on number of training samples.")
    parser.add_argument("--train-size", type=int, default=None,
                        help="Number of examples to actually use for training after filtering/formatting.")
    parser.add_argument("--max-length", type=int, default=2048, help="Maximum sequence length for training examples.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Peak learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--num-epochs", type=float, default=2.0, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size.")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps.")
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="Fraction of steps used for LR warmup.")
    parser.add_argument("--logging-steps", type=int, default=25, help="Logging interval (in steps).")
    parser.add_argument("--save-steps", type=int, default=250, help="Checkpoint interval (in steps).")
    parser.add_argument("--bf16", action="store_true", help="Train using bfloat16 (otherwise fp16).")
    parser.add_argument("--no-lora", dest="use_lora", action="store_false", help="Disable LoRA adapters and finetune full model.")
    parser.set_defaults(use_lora=True)
    parser.add_argument("--lora-r", type=int, default=64, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=128, help="LoRA alpha scaling.")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument(
        "--pad-to-max-length",
        action="store_true",
        help="Pad all sequences to max_length (default: pad to longest in batch).",
    )
    parser.add_argument(
        "--intersection-file",
        type=Path,
        default=INTERSECTION_IDS,
        help="Optional list of IDs to keep (used by default for structured mode).",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="After training, run generation on a test file.",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=REPO_ROOT / "data" / "test.jsonl",
        help="Test JSONL path for generation.",
    )
    parser.add_argument(
        "--gen-output-file",
        type=Path,
        default=None,
        help="Where to write generations (default: <output_dir>/predictions.jsonl).",
    )
    parser.add_argument(
        "--max-gen-samples",
        type=int,
        default=300,
        help="Limit number of test rows to generate (default: 300).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max new tokens to generate (default: 256).",
    )
    return parser


def slugify(text: str) -> str:
    text = text.replace("/", "_").replace("\\", "_")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "model"


def load_jsonl(path: Path, limit: Optional[int] = None) -> List[dict]:
    rows: List[dict] = []
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if limit is not None and len(rows) >= limit:
                break
    return rows


def build_instructions(mode: str, strategy: str) -> str:
    if mode == "normal":
        part_three = prompt_module.NORMAL_PART_THREE.strip()
    else:
        if strategy not in STRATEGY_TO_PROMPT:
            raise ValueError(f"Unknown strategy '{strategy}'. Options: {list(STRATEGY_TO_PROMPT)}")
        part_three = STRATEGY_TO_PROMPT[strategy]
    instructions = "\n".join(
        [
            prompt_module.PART_TWO_TASK.strip(),
            part_three,
        ]
    ).strip()
    return instructions


def load_id_sequence(path: Optional[Path]) -> Optional[List[str]]:
    """Return an ordered list of IDs to keep; preserves file order."""
    if path is None:
        return None
    if not path.exists():
        print(f"Warning: intersection file {path} not found; skipping ID filtering.", flush=True)
        return None
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return ids or None


def build_messages(example: dict, instructions: str) -> List[Dict[str, str]]:
    system = prompt_module.PART_ONE_ROLE.strip()
    user = f"Question:\n{example.get('question','').strip()}\n\n{instructions}"
    target = json.dumps(
        {
            "rationale": example.get("response_rationale", ""),
            "ans": example.get("response_ans", example.get("option", "")),
        },
        ensure_ascii=False,
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": target},
    ]


def tokenize_examples(
    examples: dict,
    tokenizer: AutoTokenizer,
    max_length: int,
    instructions: str,
    pad_to_max: bool,
) -> dict:
    chats = [
        build_messages(
            {key: examples[key][i] for key in examples.keys()},
            instructions,
        )
        for i in range(len(next(iter(examples.values()))))
    ]
    text_batch = tokenizer.apply_chat_template(
        chats, tokenize=False, add_generation_prompt=False
    )
    padding = "max_length" if pad_to_max else "longest"
    tokenized = tokenizer(
        text_batch,
        truncation=True,
        max_length=max_length,
        padding=padding,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def build_messages_for_inference(example: dict, instructions: str) -> List[Dict[str, str]]:
    system = prompt_module.PART_ONE_ROLE.strip()
    user = f"Question:\n{example.get('question','').strip()}\n\n{instructions}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_answer_field(answer_text: str) -> tuple[str, str]:
    if not answer_text:
        return "", ""
    if "####" in answer_text:
        parts = answer_text.split("####", 1)
        rationale = parts[0].strip()
        ans = parts[1].strip()
    else:
        rationale = answer_text.strip()
        ans = ""
    return rationale, ans


def maybe_apply_lora(model: AutoModelForCausalLM, cfg: DistillConfig):
    if not cfg.use_lora:
        return model
    if LoraConfig is None or get_peft_model is None:
        raise ImportError("peft is required for LoRA training. Install it or run with --no-lora.")
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def run_distillation(cfg: DistillConfig) -> None:
    rows = load_jsonl(cfg.data_file, cfg.max_samples)
    id_sequence = load_id_sequence(cfg.intersection_file if cfg.mode == "structured" else None)
    if id_sequence is not None:
        row_map = {row.get("index"): row for row in rows}
        ordered_rows = [row_map[i] for i in id_sequence if i in row_map]
        missing = len(id_sequence) - len(ordered_rows)
        if missing:
            print(f"Warning: {missing} ids from {cfg.intersection_file} not found in dataset.", flush=True)
        rows = ordered_rows
    if cfg.train_size is not None:
        rows = rows[: cfg.train_size]
    raw_ds = Dataset.from_list(rows)

    instructions = build_instructions(cfg.mode, cfg.strategy)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    def _tokenize(batch: dict) -> dict:
        return tokenize_examples(
            batch,
            tokenizer=tokenizer,
            max_length=cfg.max_length,
            instructions=instructions,
            pad_to_max=cfg.pad_to_max,
        )

    tokenized_ds = raw_ds.map(
        _tokenize,
        batched=True,
        remove_columns=raw_ds.column_names,
    )

    has_cuda = torch.cuda.is_available()
    if cfg.bf16 and not has_cuda:
        print("Warning: --bf16 requested but CUDA is unavailable. Falling back to float32.", flush=True)
    dtype = (
        torch.bfloat16
        if cfg.bf16 and has_cuda
        else torch.float16
        if has_cuda
        else torch.float32
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        device_map="auto",
        dtype=dtype,
    )
    model = maybe_apply_lora(model, cfg)

    use_gradient_checkpointing = has_cuda
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    else:
        print("Gradient checkpointing disabled (requires CUDA for efficiency).", flush=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        save_strategy="steps",
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=3,
        bf16=cfg.bf16 and has_cuda,
        fp16=(not cfg.bf16) and has_cuda,
        gradient_checkpointing=use_gradient_checkpointing,
        dataloader_pin_memory=has_cuda,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(str(cfg.output_dir))
    tokenizer.save_pretrained(str(cfg.output_dir))

    if cfg.generate:
        run_generation(
            model,
            tokenizer,
            Path(cfg.test_file),
            Path(cfg.gen_output_file),
            cfg.max_length,
            cfg.max_new_tokens,
            instructions,
            cfg.max_gen_samples,
        )


def run_generation(
    model,
    tokenizer,
    test_path: Path,
    out_path: Path,
    max_len: int,
    max_new_tokens: int,
    instructions: str,
    max_rows: Optional[int],
) -> None:
    model.eval()
    rows = load_jsonl(test_path, limit=max_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(rows):
            messages = build_messages_for_inference(row, instructions)
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(
                prompt_text, return_tensors="pt", truncation=True, max_length=max_len
            ).to(model.device)
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                )
            gen_text = tokenizer.decode(
                generated[0][inputs["input_ids"].shape[-1] :],
                skip_special_tokens=True,
            )
            gold_ans_text = row.get("answer") or ""
            rationale_part, ans_part = parse_answer_field(gold_ans_text)
            handle.write(
                json.dumps(
                    {
                        "index": row.get("index", f"test_{idx:05d}"),
                        "question": row.get("question"),
                        "gold_rationale": rationale_part,
                        "gold_ans": ans_part,
                        "model_response": gen_text.strip(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    mode = args.mode
    if mode not in {"normal", "structured"}:
        raise ValueError("Mode must be 'normal' or 'structured'.")

    if mode == "structured":
        strategy = args.strategy.lower()
        if strategy not in STRATEGY_TO_DIR:
            raise SystemExit(f"Unknown strategy '{strategy}'. Options: {list(STRATEGY_TO_DIR)}")
    else:
        strategy = "normal"

    if args.data_file:
        data_file = args.data_file
    else:
        if mode == "normal":
            data_file = NORMAL_DATA_FILE
        else:
            dir_name = STRATEGY_TO_DIR[strategy]
            data_file = STRUCT_BASE_DIR / f"filtered_{dir_name}.jsonl"

    model_slug = slugify(args.model_name)
    scratch_root = Path("/home/jyang001/scratch")
    strategy_label = strategy if mode == "structured" else "normal"
    default_output = scratch_root / f"{strategy_label}_{model_slug}"
    output_dir = args.output_dir or default_output
    intersection_file = args.intersection_file if mode == "structured" else None
    dataset_label = (args.test_file or Path("test")).stem
    size_label = args.train_size if args.train_size is not None else "all"
    default_pred = output_dir / f"{strategy_label}_{model_slug}_{dataset_label}_{size_label}.jsonl"
    gen_output_file = args.gen_output_file if args.gen_output_file else default_pred

    cfg = DistillConfig(
        data_file=data_file,
        output_dir=output_dir,
        mode=mode,
        strategy=strategy,
        intersection_file=intersection_file,
        model_name=args.model_name,
        max_samples=args.max_samples,
        train_size=args.train_size,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=args.bf16,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        pad_to_max=args.pad_to_max_length,
        generate=args.generate,
        test_file=args.test_file,
        gen_output_file=gen_output_file,
        max_gen_samples=args.max_gen_samples,
        max_new_tokens=args.max_new_tokens,
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    run_distillation(cfg)


if __name__ == "__main__":
    main()
