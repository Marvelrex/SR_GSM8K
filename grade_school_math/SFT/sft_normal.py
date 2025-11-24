#!/usr/bin/env python3
"""Fine-tune Llama-3.2-3B on normal rationales and run inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset
try:
    from peft import LoraConfig, get_peft_model
except ImportError:  # pragma: no cover - optional
    LoraConfig = None  # type: ignore
    get_peft_model = None  # type: ignore
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_query import prompts as prompt_module


def load_jsonl(path: Path, limit: int | None = None) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            if limit is not None and len(rows) >= limit:
                break
    return rows


def build_messages(example: Dict) -> List[Dict[str, str]]:
    system = prompt_module.PART_ONE_ROLE.strip()
    instructions = "\n".join(
        [
            prompt_module.PART_TWO_TASK.strip(),
            prompt_module.NORMAL_PART_THREE.strip(),
        ]
    ).strip()
    user = f"Question:\n{example.get('question','').strip()}\n\n{instructions}"

    rationale = example.get("response_rationale")
    ans = example.get("response_ans")
    target = json.dumps({"rationale": rationale, "ans": ans}, ensure_ascii=False)

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": target},
    ]


def tokenize_examples(examples, tokenizer, max_len: int, pad_to_max: bool):
    if isinstance(examples, dict):
        keys = list(examples.keys())
        length = len(examples[keys[0]]) if keys else 0
        example_list = [{k: examples[k][i] for k in keys} for i in range(length)]
    else:
        example_list = examples

    # Ensure each example is a dict
    normalized_examples = []
    for ex in example_list:
        if isinstance(ex, dict):
            normalized_examples.append(ex)
        elif isinstance(ex, str):
            normalized_examples.append({"question": ex})
        else:
            continue  # skip malformed rows

    chats = [build_messages(ex) for ex in normalized_examples]
    texts = tokenizer.apply_chat_template(
        chats, tokenize=False, add_generation_prompt=False
    )
    padding = "max_length" if pad_to_max else "longest"
    tokenized = tokenizer(
        texts,
        padding=padding,
        truncation=True,
        max_length=max_len,
        return_tensors=None,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def main():
    parser = argparse.ArgumentParser(description="SFT on normal rationales.")
    parser.add_argument(
        "--train-file",
        type=Path,
        default=PROJECT_ROOT / "filtered_gpt5_rationales" / "filtered_normal.jsonl",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "test.jsonl",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "SFT" / "outputs" / "normal")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 (default: fp16 on CUDA, fp32 on CPU).")
    parser.add_argument(
        "--pad-to-max-length",
        action="store_true",
        help="Pad sequences to max_seq_len (default: pad to longest in batch).",
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        default=False,
        help="Enable LoRA adapters (default: off).",
    )
    parser.add_argument("--lora-r", type=int, default=64, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=128, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument(
        "--generate",
        action="store_true",
        help="After training, run generation on the test file.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint to resume from (or 'auto' to pick the last in output_dir).",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    train_rows = load_jsonl(args.train_file, limit=args.max_train_samples)
    train_ds = Dataset.from_list(train_rows)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    has_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if has_cuda else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        dtype=dtype,
    )
    if args.use_lora:
        if LoraConfig is None or get_peft_model is None:
            raise ImportError("peft is required for --use-lora. Install it or rerun without --use-lora.")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    def _tokenize(batch):
        return tokenize_examples(batch, tokenizer, args.max_seq_len, args.pad_to_max_length)

    tokenized_ds = train_ds.map(
        _tokenize,
        batched=True,
        remove_columns=train_ds.column_names,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.lr,
        weight_decay=0.0,
        logging_steps=10,
        save_strategy="epoch",
        bf16=args.bf16 and torch.cuda.is_available(),
        fp16=(not args.bf16) and torch.cuda.is_available(),
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=data_collator,
    )

    resume_arg = None
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint.lower() == "auto":
            from transformers.trainer_utils import get_last_checkpoint

            resume_arg = get_last_checkpoint(str(args.output_dir))
        else:
            resume_arg = args.resume_from_checkpoint

    trainer.train(resume_from_checkpoint=resume_arg)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    if args.generate:
        run_generation(
            model,
            tokenizer,
            Path(args.test_file),
            args.output_dir / "predictions.jsonl",
            args.max_seq_len,
        )


def run_generation(model, tokenizer, test_path: Path, out_path: Path, max_len: int):
    model.eval()
    rows = load_jsonl(test_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            messages = build_messages_for_inference(row)
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(
                prompt_text, return_tensors="pt", truncation=True, max_length=max_len
            ).to(model.device)
            with torch.no_grad():
                generated = model.generate(
                    **inputs, max_new_tokens=256, do_sample=False, temperature=0.0
                )
            gen_text = tokenizer.decode(
                generated[0][inputs["input_ids"].shape[-1] :],
                skip_special_tokens=True,
            )
            handle.write(
                json.dumps(
                    {
                        "index": row.get("index"),
                        "question": row.get("question"),
                        "model_response": gen_text.strip(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def build_messages_for_inference(example: Dict) -> List[Dict[str, str]]:
    system = prompt_module.PART_ONE_ROLE.strip()
    instructions = "\n".join(
        [
            prompt_module.PART_TWO_TASK.strip(),
            prompt_module.NORMAL_PART_THREE.strip(),
        ]
    ).strip()
    user = f"Question:\n{example.get('question','').strip()}\n\n{instructions}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


if __name__ == "__main__":
    main()
