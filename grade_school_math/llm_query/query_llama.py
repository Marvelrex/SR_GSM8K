#!/usr/bin/env python3
"""Query a local Hugging Face Llama model with GSM8K prompts and store responses."""

from __future__ import annotations

import argparse
import sys
import time
from bisect import bisect_left
from pathlib import Path
from typing import Any, Dict, List

import torch as th
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import prompts as prompt_module
from query_common import (
    DEFAULT_DATASET_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_STRATEGY,
    PROMPT_STRATEGIES,
    build_prompt_texts,
    load_existing_results,
    load_samples,
    parse_model_output,
    pick_part_three_prompt,
    sample_identifier,
    split_answer_and_rationale,
    write_results, _entry_key,
)

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def ask_model(
    system_prompt: str,
    user_prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: th.device,
) -> str:
    """Send the prompt text to the target model and return its response."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    # Create the full prompt string from the chat template
    prompt_string = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Tokenize the prompt string to get input_ids and attention_mask
    inputs = tokenizer(prompt_string, return_tensors="pt").to(device)

    # Generate a response, passing both input_ids and attention_mask
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.2,
    )
    
    # Decode only the newly generated tokens, skipping the prompt
    response_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    
    return response.strip()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Combine GSM8K questions with prompts and query a local Llama model."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Path to the GSM8K jsonl file (default: {DEFAULT_DATASET_PATH})",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Zero-based row index to start from (default: 0).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of consecutive samples to process (default: 1).",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=DEFAULT_STRATEGY,
        choices=list(PROMPT_STRATEGIES.keys()),
        help="Prompt strategy to use (default: Step).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Hugging Face model identifier to query (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to store responses (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--show-only",
        action="store_true",
        help="Print the samples and prompt but do not run the model.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    # --- Model and Tokenizer Initialization ---
    th.manual_seed(42)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(42)
    print(f"Loading model: {args.model}")
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=th.bfloat16,
    )
    
    # Set pad_token_id to eos_token_id to silence the warning
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    # Set generation temperature and sampling configuration
    model.generation_config.temperature = 0.2
    model.generation_config.do_sample = True

    model.to(device)
    model.eval()
    print(f"Model loaded on device: {device}")
    # --- End Initialization ---

    strategy = args.strategy or DEFAULT_STRATEGY
    request_retries = 3
    base_system_prompt = getattr(
        prompt_module,
        "PART_ONE_ROLE",
        "You are a rigorous but concise math tutor.",
    )
    system_prompt = str(base_system_prompt or "").strip() or "You are a rigorous but concise math tutor."
    part_two_prompt = str(getattr(prompt_module, "PART_TWO_TASK", "") or "").strip()
    part_three_name, part_three_prompt = pick_part_three_prompt(strategy, prompt_module)
    combined_prompt_parts = [part.strip() for part in (part_two_prompt, part_three_prompt) if part.strip()]
    if not combined_prompt_parts:
        raise SystemExit("PART_TWO_TASK and strategy prompt are both empty; cannot build user instructions.")
    combined_prompt = "\n\n".join(combined_prompt_parts)

    samples = load_samples(args.dataset_path, args.sample_index, args.num_samples)
    results_path = args.output_dir / "results.jsonl"
    existing_entries, processed_ids = load_existing_results(results_path)
    all_entries: List[Dict[str, Any]] = list(existing_entries)
    entry_ids: List[str] = [_entry_key(entry) for entry in all_entries]
    new_entries_added = 0

    print(f"Dataset: {args.dataset_path}")
    print(f"Strategy: {strategy} ({part_three_name})")
    print(f"Model: {args.model}")
    print(f"Output directory: {args.output_dir}")
    print("-" * 60)

    for offset, (absolute_index, sample) in enumerate(samples, start=1):
        question = sample.get("question", "").strip()
        gold_answer_text = sample.get("answer", "").strip()
        gold_rationale, gold_answer = split_answer_and_rationale(gold_answer_text)
        sample_id_value = sample.get("id")
        if not sample_id_value:
            sample_id_value = sample_identifier(sample, args.dataset_path, absolute_index)
        sample_id = str(sample_id_value)

        if sample_id in processed_ids:
            print(
                f"Sample {offset}/{args.num_samples}  |  Row #{absolute_index}  |  ID: {sample_id} already processed; skipping."
            )
            continue

        print(f"Sample {offset}/{args.num_samples}  |  Row #{absolute_index}  |  ID: {sample_id}")
        print(f"Question:\n{question}\n")
        if gold_answer_text:
            print("Ground-truth answer:")
            print(gold_answer_text)
            print()

        system_prompt_text, user_prompt_text = build_prompt_texts(system_prompt, combined_prompt, question)
        
        print("---- Prompt Start ----")
        print(system_prompt_text)
        print(user_prompt_text)
        print("---- Prompt End ----")
        print()

        if args.show_only:
            continue

        max_attempts = request_retries
        model_reply = None
        for attempt in range(1, max_attempts + 1):
            try:
                with th.no_grad():
                    model_reply = ask_model(
                        system_prompt_text,
                        user_prompt_text,
                        model,
                        tokenizer,
                        device,
                    )
                print("Model response:")
                print(model_reply)
                print()
                break
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                print(
                    f"Request failed for sample {sample_id} "
                    f"(attempt {attempt}/{max_attempts}): {exc.__class__.__name__}: {exc}"
                )
                if attempt < max_attempts:
                    sleep_seconds = min(2 * attempt, 10)
                    print(f"Retrying in {sleep_seconds} seconds...\n")
                    time.sleep(sleep_seconds)
                else:
                    print("Max attempts reached; moving to next sample.\n")

        if model_reply is None:
            print(
                f"Failed to obtain a response for sample {sample_id}; recording the error and continuing."
            )
            continue

        rationale, parsed_ans = parse_model_output(model_reply)
        if rationale is None:
            rationale_value = model_reply
        else:
            rationale_value = rationale

        sample_record = {
            "index": sample_id,
            "question": question,
            "answer_from_dataset": gold_answer_text,
            "gold_rationale": gold_rationale,
            "gold": gold_answer,
            "prompt_strategy": strategy,
            "model": args.model,
            "response_rationale": rationale_value,
            "response_ans": parsed_ans,
        }
        insert_at = bisect_left(entry_ids, sample_id)
        entry_ids.insert(insert_at, sample_id)
        all_entries.insert(insert_at, sample_record)
        write_results(results_path, all_entries)
        processed_ids.add(sample_id)
        new_entries_added += 1
        print(f"Saved progress to {results_path}\n")

    if args.show_only:
        print("--show-only flag set; no API requests were made and no files were written.")
        return

    if new_entries_added:
        print(f"Processing complete. Wrote {new_entries_added} new entries to {results_path}")
    else:
        if results_path.exists():
            print(f"No new samples processed; existing entries remain in {results_path}")
        else:
            print("No samples were processed; results file was not created.")


if __name__ == "__main__":
    main()
