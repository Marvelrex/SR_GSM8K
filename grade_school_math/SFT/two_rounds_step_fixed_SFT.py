#!/usr/bin/env python3
"""
Batch launcher (similar to run_distill_experiments.py) for SFTing Llama-3.1-8B on the
filtered step and fixed rationale datasets. Runs multiple train sizes and orders the
step/fixed strategies as requested, reusing the intersection file and generating after
each run.

Examples:
  # Default: structured, strategies step then fixed, models=[Llama-3.1-8B], train sizes 300/500/700/900
  python SFT/sft_llama31_8b.py

  # Run fixed first, only sizes 500 and 700, custom output root
  python SFT/sft_llama31_8b.py --strategy-order fixed step --train-sizes 500 700 --output-root SFT/outputs/llama31

  # Use two models and a custom test file
  python SFT/sft_llama31_8b.py --models meta-llama/Llama-3.1-8B-Instruct meta-llama/Llama-3.1-8B --test-file data/test.jsonl
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
SCRIPT = SCRIPTS_DIR / "distill_rationale.py"

DEFAULT_TRAIN_SIZES = [300, 500, 700, 900]
DEFAULT_MODELS = ["meta-llama/Llama-3.1-8B-Instruct"]
INTERSECTION_IDS = REPO_ROOT / "filtered_gpt5_rationales" / "filtered_intersection_ids.txt"
STEP_FILE = REPO_ROOT / "filtered_gpt5_rationales" / "filtered_step.jsonl"
FIXED_FILE = REPO_ROOT / "filtered_gpt5_rationales" / "filtered_fixed.jsonl"
NORMAL_FILE = REPO_ROOT / "filtered_gpt5_rationales" / "filtered_normal.jsonl"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "SFT" / "outputs"
DEFAULT_PRED_ROOT = REPO_ROOT / "SFT" / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch SFT launcher for Llama-3.1-8B on filtered step/fixed datasets."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=f"Model ids to finetune (default: {DEFAULT_MODELS}).",
    )
    parser.add_argument(
        "--mode",
        choices=["normal", "structured"],
        default="structured",
        help="Rationale mode for distillation (default: structured).",
    )
    parser.add_argument(
        "--strategy-order",
        nargs="+",
        default=["step", "fixed"],
        choices=["step", "fixed"],
        help="Order to run structured strategies; ignored in normal mode (default: step fixed).",
    )
    parser.add_argument(
        "--train-sizes",
        nargs="+",
        type=int,
        default=DEFAULT_TRAIN_SIZES,
        help=f"Train sizes to iterate over (default: {DEFAULT_TRAIN_SIZES}).",
    )
    parser.add_argument(
        "--step-file",
        type=Path,
        default=STEP_FILE,
        help=f"Path to filtered_step.jsonl (default: {STEP_FILE}).",
    )
    parser.add_argument(
        "--fixed-file",
        type=Path,
        default=FIXED_FILE,
        help=f"Path to filtered_fixed.jsonl (default: {FIXED_FILE}).",
    )
    parser.add_argument(
        "--intersection-file",
        type=Path,
        default=INTERSECTION_IDS,
        help="Intersection file to pass through for structured mode.",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        default=True,
        help="Generate after each run (default: on).",
    )
    parser.add_argument(
        "--no-generate",
        dest="generate",
        action="store_false",
        help="Disable generation after training.",
    )
    parser.add_argument(
        "--max-gen-samples",
        type=int,
        default=300,
        help="Max number of test rows to generate (default: 300).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max new tokens during generation (default: 256).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature (default: 0.0).",
    )
    parser.add_argument(
        "--do-sample",
        dest="do_sample",
        action="store_true",
        help="Enable sampling for generation (default: off).",
    )
    parser.add_argument(
        "--no-do-sample",
        dest="do_sample",
        action="store_false",
        help="Disable sampling for generation.",
    )
    parser.set_defaults(do_sample=False)
    parser.add_argument(
        "--test-file",
        type=Path,
        default=None,
        help="Optional test JSONL path (default: distill_rationale.py default).",
    )
    parser.add_argument(
        "--gen-output-file",
        type=Path,
        default=None,
        help="Optional explicit predictions path (default: auto per run).",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4).",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=250,
        help="Checkpoint save frequency (steps) (default: 250).",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=25,
        help="Logging frequency (steps) (default: 25).",
    )
    parser.add_argument(
        "--num-epochs",
        type=float,
        default=4.0,
        help="Number of training epochs to pass through (default: 4.0).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device batch size (default: 2).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Root directory for checkpoints (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument(
        "--pred-root",
        type=Path,
        default=DEFAULT_PRED_ROOT,
        help=f"Root directory for prediction files (default: {DEFAULT_PRED_ROOT}).",
    )
    parser.add_argument(
        "--flatten-targets",
        action="store_true",
        help="Flatten rationale/ans fields before training.",
    )
    parser.add_argument(
        "--print-chat",
        action="store_true",
        help="Print formatted chat for every example.",
    )
    return parser.parse_args()


def default_data_file(mode: str, strategy: Optional[str], args: argparse.Namespace) -> Path:
    if mode == "normal":
        return NORMAL_FILE
    if strategy == "step":
        return args.step_file
    if strategy == "fixed":
        return args.fixed_file
    raise SystemExit(f"Unknown strategy '{strategy}'. Expected 'step' or 'fixed'.")


def build_base_cmd(
    model_name: str,
    mode: str,
    strategy: Optional[str],
    data_file: Path,
    intersection_file: Optional[Path],
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--bf16",
        "--batch-size",
        str(args.batch_size),
        "--grad-accum",
        str(args.grad_accum),
        "--num-epochs",
        str(args.num_epochs),
        "--mode",
        mode,
        "--data-file",
        str(data_file),
        "--model-name",
        model_name,
        "--save-steps",
        str(args.save_steps),
        "--logging-steps",
        str(args.logging_steps),
    ]
    if mode == "structured" and strategy:
        cmd += ["--strategy", strategy]
        if intersection_file:
            cmd += ["--intersection-file", str(intersection_file)]
    if args.generate:
        cmd += [
            "--generate",
            "--max-gen-samples",
            str(args.max_gen_samples),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--temperature",
            str(args.temperature),
        ]
        if args.do_sample:
            cmd.append("--do-sample")
        else:
            cmd.append("--no-do-sample")
        if args.test_file:
            cmd += ["--test-file", str(args.test_file)]
        if args.gen_output_file:
            cmd += ["--gen-output-file", str(args.gen_output_file)]
    if args.flatten_targets:
        cmd.append("--flatten-targets")
    if args.print_chat:
        cmd.append("--print-chat")
    return cmd


def run_experiment(
    train_size: int,
    base_cmd: list[str],
    model_name: str,
    mode: str,
    strategy: Optional[str],
    args: argparse.Namespace,
) -> None:
    suffix = "structured" if mode == "structured" else "normal"
    strat_suffix = f"_{strategy}" if mode == "structured" and strategy else ""
    model_slug = Path(model_name).name
    output_dir = args.output_root / model_slug / f"{suffix}{strat_suffix}_ts{train_size}"
    run_pred_dir = args.pred_root / model_slug / f"{suffix}{strat_suffix}_ts{train_size}"
    run_pred_dir.mkdir(parents=True, exist_ok=True)
    dataset_label = (args.test_file or Path("test")).stem
    default_pred = run_pred_dir / f"{suffix}{strat_suffix}_{model_slug}_{dataset_label}_{train_size}.jsonl"

    cmd = list(base_cmd)
    if args.generate and not args.gen_output_file:
        cmd += ["--gen-output-file", str(default_pred)]
    cmd += [
        "--train-size",
        str(train_size),
        "--output-dir",
        str(output_dir),
    ]
    print(
        f"\n=== Running model={model_slug} mode={mode} strategy={strategy or 'n/a'} "
        f"train_size={train_size} -> {output_dir} ===",
        flush=True,
    )
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main() -> None:
    args = parse_args()
    strategies = ["step", "fixed"] if args.mode == "structured" else [None]
    if args.mode == "structured":
        strategies = args.strategy_order
        bad = [s for s in strategies if s not in {"step", "fixed"}]
        if bad:
            raise SystemExit(f"Invalid strategies in --strategy-order: {bad}. Allowed: step, fixed.")

    for model_name in args.models:
        for strat in strategies:
            data_path = default_data_file(args.mode, strat, args)
            base_cmd = build_base_cmd(
                model_name,
                args.mode,
                strat,
                data_path,
                args.intersection_file if args.mode == "structured" else None,
                args,
            )
            for size in args.train_sizes:
                run_experiment(size, base_cmd, model_name, args.mode, strat, args)


if __name__ == "__main__":
    main()
