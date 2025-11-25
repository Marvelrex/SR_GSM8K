#!/usr/bin/env python3
"""
Weighted distillation of MathQA rationales into a chat Llama model.

Differences vs the simpler distill_rationale.py:
- Only the assistant JSON is supervised; system/user tokens are masked.
- Per-token weights: answer > rationale > JSON scaffolding.
- Generation guards against empty or punctuation-only outputs (min_new_tokens + fallback).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import re

import torch
import torch.nn.functional as F
from datasets import Dataset  # type: ignore
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

# Ensure repo root on sys.path so llm_query imports work regardless of cwd
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:  # pragma: no cover - import shim for flexible entrypoints
    from llm_query import prompts as prompt_module
except ImportError:  # pragma: no cover
    from grade_school_math.llm_query import prompts as prompt_module  # type: ignore

try:
    from peft import LoraConfig, get_peft_model
except ImportError:  # pragma: no cover
    LoraConfig = None  # type: ignore
    get_peft_model = None  # type: ignore


SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
NORMAL_DATA_FILE = REPO_ROOT / "filtered_gpt5_rationales" / "filtered_normal.jsonl"
STRUCT_BASE_DIR = REPO_ROOT / "filtered_gpt5_rationales"
INTERSECTION_IDS = REPO_ROOT / "filtered_gpt5_rationales" / "filtered_intersection_ids.txt"


PROMPT_STRATEGIES = {
    "normal": "NORMAL_PART_THREE",
    "fixed": "STRUCTURED_FIXED_PART_THREE",
    "step": "STRUCTURED_STEP_PART_THREE",
    "freeform": "STRUCTURED_FREE_FORM_PART_THREE",
    "noisy": "STRUCTURED_NOISE_PART_THREE",
}


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


def load_id_sequence(path: Optional[Path]) -> Optional[List[str]]:
    if path is None:
        return None
    if not path.exists():
        return None
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return ids or None


def build_instructions(mode: str, strategy: str) -> str:
    if mode == "normal":
        part_three = prompt_module.NORMAL_PART_THREE.strip()
    else:
        prompt_name = PROMPT_STRATEGIES[strategy]
        part_three = getattr(prompt_module, prompt_name).strip()
    return "\n".join([prompt_module.PART_TWO_TASK.strip(), part_three]).strip()


def build_messages(example: dict, instructions: str) -> List[Dict[str, str]]:
    system = prompt_module.PART_ONE_ROLE.strip()
    user = f"Question:\n{example.get('question','').strip()}\n\n{instructions}"
    target = json.dumps(
        {"rationale": example.get("response_rationale", ""), "ans": example.get("response_ans", example.get("option", ""))},
        ensure_ascii=False,
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": target},
    ]


def format_chat(tokenizer, chat: List[Dict[str, str]], add_generation_prompt: bool) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=add_generation_prompt)
    system = chat[0]["content"] if chat and chat[0]["role"] == "system" else ""
    user = chat[1]["content"] if len(chat) > 1 and chat[1]["role"] == "user" else ""
    assistant = chat[2]["content"] if len(chat) > 2 and chat[2]["role"] == "assistant" else ""
    if add_generation_prompt:
        return f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]"
    return f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST] {assistant}</s>"


def _span(text: str, substring: str, base: int) -> Optional[Tuple[int, int]]:
    pos = text.find(substring)
    if pos == -1:
        return None
    return (base + pos, base + pos + len(substring))


def tokenize_examples(
    example: dict,
    tokenizer: AutoTokenizer,
    max_length: int,
    instructions: str,
    answer_weight: float,
    rationale_weight: float,
    json_weight: float,
) -> dict:
    chat = build_messages(example, instructions)
    assistant_json = str(chat[-1]["content"])
    chat_text = format_chat(tokenizer, chat, add_generation_prompt=False)

    tokenized = tokenizer(
        chat_text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_offsets_mapping=True,
    )
    input_ids = tokenized["input_ids"]
    labels = input_ids.copy()
    offsets = tokenized["offset_mapping"]

    rat_val = example.get("response_rationale", "")
    rat_text = json.dumps(rat_val, ensure_ascii=False) if isinstance(rat_val, (dict, list)) else str(rat_val or "")
    ans_val = example.get("response_ans", example.get("option", ""))
    ans_text = json.dumps(ans_val, ensure_ascii=False) if isinstance(ans_val, (dict, list)) else str(ans_val or "")

    assistant_start = chat_text.rfind(assistant_json)
    rat_range = ans_range = json_range = None
    if assistant_start != -1:
        json_range = (assistant_start, assistant_start + len(assistant_json))
        if rat_text:
            rat_range = _span(assistant_json, rat_text, assistant_start)
        if ans_text:
            ans_range = _span(assistant_json, ans_text, assistant_start)
        if ans_range is None and rat_range is None:
            ans_range = json_range

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    weights = [0.0 for _ in labels]

    if assistant_start == -1:
        for i in range(len(labels)):
            if labels[i] == pad_id:
                labels[i] = -100
            else:
                weights[i] = 1.0
    else:
        for i, (start, end) in enumerate(offsets):
            if labels[i] == pad_id:
                labels[i] = -100
                continue
            if ans_range and start < ans_range[1] and end > ans_range[0]:
                weights[i] = answer_weight
            elif rat_range and start < rat_range[1] and end > rat_range[0]:
                weights[i] = rationale_weight
            elif json_range and start < json_range[1] and end > json_range[0]:
                weights[i] = json_weight
            else:
                labels[i] = -100

    if not any(w > 0 for w in weights):
        for i in range(len(labels)):
            if labels[i] == pad_id:
                labels[i] = -100
            else:
                weights[i] = 1.0

    tokenized["labels"] = labels
    tokenized["loss_weights"] = weights
    tokenized.pop("offset_mapping", None)
    return tokenized


class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        weights = inputs.pop("loss_weights", None)
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if labels is None:
            loss = outputs.loss
        else:
            vocab_size = logits.size(-1)
            loss_flat = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), reduction="none")
            if weights is not None:
                w_flat = torch.tensor(weights, device=logits.device, dtype=loss_flat.dtype).view_as(labels).view(-1)
                valid = labels.view(-1) != -100
                loss = (loss_flat[valid] * w_flat[valid]).sum() / (w_flat[valid].sum() + 1e-8)
            else:
                loss = loss_flat[labels.view(-1) != -100].mean()
        return (loss, outputs) if return_outputs else loss


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
    rationale_weight: float
    answer_weight: float
    json_weight: float


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


def parse_answer_field(answer_text: str) -> tuple[str, str]:
    if not answer_text:
        return "", ""
    if "####" in answer_text:
        parts = answer_text.split("####", 1)
        return parts[0].strip(), parts[1].strip()
    return answer_text.strip(), ""


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

    eos_id = tokenizer.eos_token_id
    eot_id = None
    try:
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    except Exception:
        pass
    eos_ids = []
    if isinstance(eos_id, list):
        eos_ids.extend(eos_id)
    elif eos_id is not None:
        eos_ids.append(eos_id)
    if eot_id is not None and eot_id not in eos_ids and eot_id != -1:
        eos_ids.append(eot_id)
    eos_arg = eos_ids if eos_ids else eos_id
    pad_id = tokenizer.pad_token_id or eos_id

    base_gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        min_new_tokens=8,
        repetition_penalty=1.05,
        eos_token_id=eos_arg,
        pad_token_id=pad_id,
    )
    fallback_gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=1.0,
        top_p=0.95,
        min_new_tokens=8,
        repetition_penalty=1.1,
        eos_token_id=eos_arg,
        pad_token_id=pad_id,
    )

    completed_ids: set[str] = set()
    if out_path.exists():
        try:
            with out_path.open("r", encoding="utf-8") as existing:
                for line in existing:
                    try:
                        obj = json.loads(line)
                        if "index" in obj:
                            completed_ids.add(str(obj["index"]))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            pass

    with out_path.open("a", encoding="utf-8") as handle:
        for idx, row in enumerate(rows):
            row_id = str(row.get("index", f"test_{idx:05d}"))
            if row_id in completed_ids:
                continue
            messages = [
                {"role": "system", "content": prompt_module.PART_ONE_ROLE.strip()},
                {"role": "user", "content": f"Question:\n{row.get('question','').strip()}\n\n{instructions}"},
            ]
            prompt_text = format_chat(tokenizer, messages, add_generation_prompt=True)
            inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_len).to(model.device)
            with torch.no_grad():
                generated = model.generate(**inputs, **base_gen_kwargs)
            gen_text = tokenizer.decode(generated[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True).strip()
            if not gen_text or generated.shape[1] == inputs["input_ids"].shape[1]:
                with torch.no_grad():
                    generated = model.generate(**inputs, **fallback_gen_kwargs)
                gen_text = tokenizer.decode(generated[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True).strip()
            if not gen_text:
                gen_text = tokenizer.decode(generated[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=False).strip()
            gold_ans_text = row.get("answer") or ""
            rationale_part, ans_part = parse_answer_field(gold_ans_text)
            handle.write(
                json.dumps(
                    {
                        "index": row_id,
                        "question": row.get("question"),
                        "gold_rationale": rationale_part,
                        "gold_ans": ans_part,
                        "model_response": gen_text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            handle.flush()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Weighted distillation of MathQA rationales into Llama chat models.")
    parser.add_argument("--mode", choices=["normal", "structured"], default="normal")
    parser.add_argument("--strategy", choices=list(PROMPT_STRATEGIES.keys()), default="step")
    parser.add_argument("--data-file", type=Path, help="Override dataset path.")
    parser.add_argument("--output-dir", type=Path, help="Override output dir.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--no-lora", dest="use_lora", action="store_false")
    parser.set_defaults(use_lora=True)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--pad-to-max-length", action="store_true")
    parser.add_argument("--intersection-file", type=Path, default=INTERSECTION_IDS)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--test-file", type=Path, default=REPO_ROOT / "data" / "test.jsonl")
    parser.add_argument("--gen-output-file", type=Path, default=None)
    parser.add_argument("--max-gen-samples", type=int, default=300)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--rationale-weight", type=float, default=0.4)
    parser.add_argument("--answer-weight", type=float, default=1.0)
    parser.add_argument("--json-weight", type=float, default=0.3)
    return parser


def run_distillation(cfg: DistillConfig) -> None:
    rows = load_jsonl(cfg.data_file, cfg.max_samples)
    id_sequence = load_id_sequence(cfg.intersection_file if cfg.mode == "structured" else None)
    if id_sequence is not None:
        row_map = {row.get("index"): row for row in rows}
        ordered_rows = [row_map[i] for i in id_sequence if i in row_map]
        rows = ordered_rows
    if cfg.train_size is not None:
        rows = rows[: cfg.train_size]
    raw_ds = Dataset.from_list(rows)

    instructions = build_instructions(cfg.mode, cfg.strategy)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    def _tokenize(example: dict) -> dict:
        return tokenize_examples(
            example,
            tokenizer=tokenizer,
            max_length=cfg.max_length,
            instructions=instructions,
            answer_weight=cfg.answer_weight,
            rationale_weight=cfg.rationale_weight,
            json_weight=cfg.json_weight,
        )

    tokenized_ds = raw_ds.map(_tokenize, batched=False, remove_columns=raw_ds.column_names, desc="Tokenizing")

    has_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if cfg.bf16 and has_cuda else (torch.float16 if has_cuda else torch.float32)

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, device_map="auto", torch_dtype=dtype)
    model = maybe_apply_lora(model, cfg)

    use_gradient_checkpointing = has_cuda
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    data_collator = default_data_collator

    training_args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=2,
        bf16=cfg.bf16 and has_cuda,
        fp16=(not cfg.bf16) and has_cuda,
        gradient_checkpointing=use_gradient_checkpointing,
        dataloader_pin_memory=has_cuda,
        report_to="none",
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=data_collator,
    )
    resume_ckpt = None
    last_ckpt = get_last_checkpoint(str(cfg.output_dir)) if cfg.output_dir.exists() else None
    if last_ckpt:
        resume_ckpt = last_ckpt
    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(str(cfg.output_dir))
    tokenizer.save_pretrained(str(cfg.output_dir))

    if cfg.generate:
        out_path = cfg.gen_output_file if cfg.gen_output_file else cfg.output_dir / "predictions.jsonl"
        run_generation(
            model,
            tokenizer,
            Path(cfg.test_file),
            Path(out_path),
            cfg.max_length,
            cfg.max_new_tokens,
            instructions,
            cfg.max_gen_samples,
        )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    mode = args.mode
    strategy = args.strategy if mode == "structured" else "normal"

    data_file = args.data_file or (NORMAL_DATA_FILE if mode == "normal" else STRUCT_BASE_DIR / f"filtered_{strategy}.jsonl")
    model_slug = slugify(args.model_name)
    strategy_label = strategy if mode == "structured" else "normal"
    default_output = REPO_ROOT / "SFT" / "outputs" / f"{strategy_label}_{model_slug}"
    output_dir = args.output_dir or default_output
    output_dir.mkdir(parents=True, exist_ok=True)

    gen_output_file = args.gen_output_file
    if gen_output_file is None:
        dataset_label = (args.test_file or Path("test")).stem
        size_label = args.train_size if args.train_size is not None else "all"
        gen_output_file = output_dir / f"{strategy_label}_{model_slug}_{dataset_label}_{size_label}.jsonl"

    cfg = DistillConfig(
        data_file=data_file,
        output_dir=output_dir,
        mode=mode,
        strategy=strategy,
        intersection_file=args.intersection_file if mode == "structured" else None,
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
        rationale_weight=args.rationale_weight,
        answer_weight=args.answer_weight,
        json_weight=args.json_weight,
    )
    run_distillation(cfg)


if __name__ == "__main__":
    main()
