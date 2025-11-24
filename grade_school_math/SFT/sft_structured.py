#!/usr/bin/env python3
"""Fine-tune Llama-3.2-3B on structured rationales (fixed/step/freeform/noisy)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset, concatenate_datasets
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

STRATEGY_TO_DIR = {
    "fixed": "fixed",
    "step": "step",
    "freeform": "free_form",
    "noisy": "noise",
}

STRATEGY_TO_PROMPT = {
    "fixed": prompt_module.STRUCTURED_FIXED_PART_THREE.strip(),
    "step": prompt_module.STRUCTURED_STEP_PART_THREE.strip(),
    "freeform": prompt_module.STRUCTURED_FREE_FORM_PART_THREE.strip(),
    "noisy": prompt_module.STRUCTURED_NOISE_PART_THREE.strip(),
}


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


def build_messages(example: Dict, strategy: str) -> List[Dict[str, str]]:
    system = prompt_module.PART_ONE_ROLE.strip()
    part_three = STRATEGY_TO_PROMPT[strategy]
    instructions = "\n".join(
        [
            prompt_module.PART_TWO_TASK.strip(),
            part_three,
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


def tokenize_examples(examples, tokenizer, max_len: int, strategy: str):
    if isinstance(examples, dict):
        keys = list(examples.keys())
        length = len(examples[keys[0]]) if keys else 0
        example_list = [{k: examples[k][i] for k in keys} for i in range(length)]
    else:
        example_list = examples

    normalized_examples = []
    for ex in example_list:
        if isinstance(ex, dict):
            normalized_examples.append(ex)
        elif isinstance(ex, str):
            normalized_examples.append({"question": ex})
        else:
            continue

    chats = [build_messages(ex, strategy) for ex in normalized_examples]
    texts = tokenizer.apply_chat_template(
        chats, tokenize=False, add_generation_prompt=False
    )
    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors=None,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def main():
    parser = argparse.ArgumentParser(
        description="SFT on structured rationales (fixed/step/freeform/noisy)."
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["fixed", "step", "freeform", "noisy"],
        help="Which strategies to include. Options: fixed step freeform noisy.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "structure_rationale" / "gpt5" / "third_run",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "SFT" / "outputs" / "structured")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--generate",
        action="store_true",
        help="After training, run generation on the test file.",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "test.jsonl",
    )
    parser.add_argument(
        "--gen-strategy",
        type=str,
        default="step",
        help="Strategy prompt to use for generation (fixed/step/freeform/noisy).",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    selected = []
    for name in args.strategies:
        key = name.lower()
        if key not in STRATEGY_TO_DIR:
            raise SystemExit(f"Unknown strategy '{name}'. Options: {list(STRATEGY_TO_DIR)}")
        selected.append(key)

    datasets = []
    for strat in selected:
        dir_name = STRATEGY_TO_DIR[strat]
        file_name = f"filtered_{dir_name}_results.jsonl"
        path = args.base_dir / dir_name / file_name
        if not path.exists():
            raise SystemExit(f"Missing file for strategy '{strat}': {path}")
        rows = load_jsonl(path, limit=args.max_train_samples)
        ds = Dataset.from_list(rows)
        ds = ds.add_column("sft_strategy", [strat] * len(ds))
        datasets.append(ds)

    if not datasets:
        raise SystemExit("No data loaded; check strategy names and paths.")
    train_ds = concatenate_datasets(datasets)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    def _tokenize(batch):
        # batch["sft_strategy"] is uniform per example even when batched
        strategy = batch["sft_strategy"][0] if isinstance(batch["sft_strategy"], list) else batch["sft_strategy"]
        return tokenize_examples(batch, tokenizer, args.max_seq_len, strategy)

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
        bf16=torch.cuda.is_available(),
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    if args.generate:
        run_generation(
            model,
            tokenizer,
            Path(args.test_file),
            args.output_dir / "predictions.jsonl",
            args.max_seq_len,
            strategy=args.gen_strategy.lower(),
        )


def run_generation(model, tokenizer, test_path: Path, out_path: Path, max_len: int, strategy: str):
    if strategy not in STRATEGY_TO_PROMPT:
        raise SystemExit(f"Unknown gen-strategy '{strategy}'. Options: {list(STRATEGY_TO_PROMPT)}")
    model.eval()
    rows = load_jsonl(test_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            messages = build_messages_for_inference(row, strategy)
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
                        "strategy": strategy,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def build_messages_for_inference(example: Dict, strategy: str) -> List[Dict[str, str]]:
    system = prompt_module.PART_ONE_ROLE.strip()
    part_three = STRATEGY_TO_PROMPT[strategy]
    instructions = "\n".join(
        [
            prompt_module.PART_TWO_TASK.strip(),
            part_three,
        ]
    ).strip()
    user = f"Question:\n{example.get('question','').strip()}\n\n{instructions}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


if __name__ == "__main__":
    main()
